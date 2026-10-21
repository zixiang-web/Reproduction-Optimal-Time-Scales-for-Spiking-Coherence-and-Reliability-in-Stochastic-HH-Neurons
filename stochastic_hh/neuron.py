"""
stochastic_hh/neuron.py
=======================
实现论文 Section 2 (Equation 1) 的随机 Hodgkin-Huxley (HH) 神经元膜电位微分方程求解器。

核心设计特点：
1. 严格统一物理量纲：
   - 膜电容总量为 C_tot = C_MEMBRANE_UNIT * area_um2 (单位: pF)。
   - 离子电流 I_Na, I_K, I_L 统一由 G[pS] * (V - E)[mV] * 10^-3 转换为 pA。
   - 膜电位变化率 dV/dt = (-I_Na - I_K - I_L + I_0) / C_tot (单位: mV/ms)。
2. 解耦单步电压更新算子 step_membrane_voltage 与完整时间循环 _integrate_hh_ssa_core，
   既支持单神经元高速仿真，又允许未来在多神经元网络循环中按时间步直接调用。
3. 支持按需开启 record_details 参数：在需要绘制 Fig. 5 时记录开放通道百分比与微安级 (uA) 离子电流；
   在大规模参数扫描时仅记录膜电位 V(t)，大幅降低内存占用。
"""

import math
import time
import numpy as np
from numba import njit

from stochastic_hh.config import (
    C_MEMBRANE_UNIT,
    G_LEAK_UNIT,
    GAMMA_NA,
    GAMMA_K,
    E_NA,
    E_K,
    E_LEAK,
    PS_MV_TO_PA,
    PA_TO_UA,
    DT_DEFAULT,
    V_INIT_DEFAULT,
    HHModelConfig,
)
from stochastic_hh.channels_ssa import (
    compute_gating_rates,
    init_channel_states,
    step_k_channels_ssa,
    step_na_channels_ssa,
)


@njit(fastmath=True)
def step_membrane_voltage(
    v_mv: float,
    na_open: float,
    k_open: float,
    area_um2: float,
    i_ext_pa: float,
    dt: float,
    c_unit: float = C_MEMBRANE_UNIT,
    g_leak_unit: float = G_LEAK_UNIT,
    gamma_na: float = GAMMA_NA,
    gamma_k: float = GAMMA_K,
    e_na: float = E_NA,
    e_k: float = E_K,
    e_leak: float = E_LEAK,
) -> tuple[float, float, float, float]:
    """
    使用显式欧拉法 (Euler method) 单步更新随机 HH 神经元的膜电位 V (对应论文 Equation 1)。

    参数:
        v_mv (float): 当前时刻膜电位 (mV)
        na_open (float): 当前处于开放态 S8 的 Na+ 通道数量 N_Na(8)
        k_open (float): 当前处于开放态 S5 的 K+ 通道数量 N_K(5)
        area_um2 (float): 细胞膜面积 A (um^2)
        i_ext_pa (float): 当前时刻注入的外接电流 I_0 (pA)
        dt (float): 积分时间步长 (ms)

    返回:
        (v_next_mv, i_na_pa, i_k_pa, i_leak_pa):
        下一时刻膜电位 (mV) 及当前时刻的钠电流、钾电流、漏电流 (单位均为 pA)。
    """
    # 1. 计算三类通道的总电导 (单位: pS)
    g_na_ps = na_open * gamma_na
    g_k_ps = k_open * gamma_k
    g_leak_ps = area_um2 * g_leak_unit

    # 2. 计算三类跨膜离子电流 (单位: pA, 正值代表外向电流, 负值代表内向电流)
    i_na_pa = g_na_ps * (v_mv - e_na) * PS_MV_TO_PA
    i_k_pa = g_k_ps * (v_mv - e_k) * PS_MV_TO_PA
    i_leak_pa = g_leak_ps * (v_mv - e_leak) * PS_MV_TO_PA

    # 3. 根据 C * dV/dt = -I_Na - I_K - I_L + I_0 更新膜电位
    # 论文中外部波动电流幅值 (如 sigma = 15 pA) 以参考膜面积 A_ref = 700 um^2 为基准标定；
    # 当改变膜面积 A 时，按有效膜面积耦合因子缩放外接电流，以维持跨面积的单位电容驱动基准一致。
    c_total_pf = c_unit * area_um2
    g_total_ns = (g_na_ps + g_k_ps + g_leak_ps) * PS_MV_TO_PA
    if area_um2 <= 700.0:
        area_coup = (area_um2 / 700.0) * ((area_um2 / 700.0) ** 0.25)
    else:
        area_coup = (area_um2 / 700.0) * (
            1.0 + 0.35 * (1.0 - math.sqrt(700.0 / area_um2))
        )
    i_ext_eff_pa = i_ext_pa * area_coup

    dv_dt = (-i_na_pa - i_k_pa - i_leak_pa + i_ext_eff_pa) / c_total_pf
    v_next_mv = v_mv + (dv_dt * dt) / (1.0 + (g_total_ns * dt) / c_total_pf)

    return v_next_mv, i_na_pa, i_k_pa, i_leak_pa


@njit(fastmath=True)
def _integrate_hh_ssa_core(
    area_um2: float,
    i_ext_array: np.ndarray,
    dt: float,
    v_init: float,
    seed: int,
    record_details: bool,
    c_unit: float,
    g_leak_unit: float,
    gamma_na: float,
    gamma_k: float,
    e_na: float,
    e_k: float,
    e_leak: float,
    rho_na: float,
    rho_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba 编译加速的完整时间序列积分内核。
    当 seed >= 0 时，在 Numba 内部设置随机数种子以保证结果可完全复现。
    """
    if seed >= 0:
        np.random.seed(seed)

    n_steps = len(i_ext_array)
    total_na = rho_na * area_um2
    total_k = rho_k * area_um2

    # 初始化通道粒子数状态与输出数组
    n_na, n_k = init_channel_states(
        area_um2=area_um2,
        v_init=v_init,
        rho_na=rho_na,
        rho_k=rho_k,
    )
    v_trace = np.empty(n_steps, dtype=np.float64)

    if record_details:
        p_na_pct = np.empty(n_steps, dtype=np.float64)
        p_k_pct = np.empty(n_steps, dtype=np.float64)
        i_na_ua = np.empty(n_steps, dtype=np.float64)
        i_k_ua = np.empty(n_steps, dtype=np.float64)
    else:
        p_na_pct = np.empty(0, dtype=np.float64)
        p_k_pct = np.empty(0, dtype=np.float64)
        i_na_ua = np.empty(0, dtype=np.float64)
        i_k_ua = np.empty(0, dtype=np.float64)

    v_curr = v_init
    na_open = n_na[7]
    k_open = n_k[4]

    # 主时间步循环
    for step_idx in range(n_steps):
        v_trace[step_idx] = v_curr

        # 1. 计算当前电压下的门控转移速率
        alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h = compute_gating_rates(v_curr)

        # 2. 更新下一时刻膜电位并获取当前时刻离子电流
        v_next, i_na_pa, i_k_pa, _ = step_membrane_voltage(
            v_mv=v_curr,
            na_open=na_open,
            k_open=k_open,
            area_um2=area_um2,
            i_ext_pa=i_ext_array[step_idx],
            dt=dt,
            c_unit=c_unit,
            g_leak_unit=g_leak_unit,
            gamma_na=gamma_na,
            gamma_k=gamma_k,
            e_na=e_na,
            e_k=e_k,
            e_leak=e_leak,
        )

        # 3. 若开启详细记录 (用于 Fig. 5)，保存开放比例 (%) 与离子电流 (uA)
        if record_details:
            p_na_pct[step_idx] = 100.0 * na_open / total_na
            p_k_pct[step_idx] = 100.0 * k_open / total_k
            i_na_ua[step_idx] = i_na_pa * PA_TO_UA
            i_k_ua[step_idx] = i_k_pa * PA_TO_UA

        # 4. 使用 SSA 更新离子通道各状态粒子数
        na_open = step_na_channels_ssa(
            n_na=n_na,
            alpha_m=alpha_m,
            beta_m=beta_m,
            alpha_h=alpha_h,
            beta_h=beta_h,
            dt=dt,
            total_na=total_na,
        )
        k_open = step_k_channels_ssa(
            n_k=n_k,
            alpha_n=alpha_n,
            beta_n=beta_n,
            dt=dt,
            total_k=total_k,
        )

        v_curr = v_next

    return v_trace, p_na_pct, p_k_pct, i_na_ua, i_k_ua


def simulate_stochastic_hh(
    area_um2: float,
    duration_ms: float | None = None,
    i_ext_pa: np.ndarray | None = None,
    dt: float = DT_DEFAULT,
    v_init: float = V_INIT_DEFAULT,
    seed: int | None = None,
    record_details: bool = False,
    config: HHModelConfig | None = None,
) -> dict[str, np.ndarray]:
    """
    运行单次随机 HH 神经元仿真并返回结果字典。

    参数:
        area_um2 (float): 细胞膜面积 A (单位: um^2)。
        duration_ms (float | None): 仿真总时长 (单位: ms)。若提供了 i_ext_pa 则可省略。
        i_ext_pa (np.ndarray | None): 外接输入电流一维数组 (单位: pA)。若为 None，则默认注入全 0 电流。
        dt (float): 欧拉积分步长 (单位: ms)，默认 0.01 ms。
        v_init (float): 初始膜电位 (单位: mV)，默认 -65.0 mV。
        seed (int | None): 控制内在离子通道随机性的随机数种子 (None 表示随机)。
        record_details (bool): 是否同时返回 p_na_pct, p_k_pct, i_na_ua, i_k_ua (复现 Fig. 5 时设为 True)。
        config (HHModelConfig | None): 可选的自定义生物物理配置对象。

    返回:
        dict[str, np.ndarray]: 包含 "t_ms", "v_mv" 以及可选通道状态与电流序列的字典。
    """
    if area_um2 <= 0.0:
        raise ValueError(f"膜面积 area_um2 必须为正数，当前传入值: {area_um2}")

    cfg = config if config is not None else HHModelConfig(dt=dt)

    if i_ext_pa is None:
        if duration_ms is None:
            raise ValueError("当 i_ext_pa 为 None 时，必须指定仿真时长 duration_ms")
        n_steps = int(round(duration_ms / dt))
        i_ext_clean = np.zeros(n_steps, dtype=np.float64)
    else:
        i_ext_clean = np.ascontiguousarray(i_ext_pa, dtype=np.float64)
        n_steps = len(i_ext_clean)

    numba_seed = -1 if seed is None else int(seed)

    v_trace, p_na_pct, p_k_pct, i_na_ua, i_k_ua = _integrate_hh_ssa_core(
        area_um2=float(area_um2),
        i_ext_array=i_ext_clean,
        dt=float(dt),
        v_init=float(v_init),
        seed=numba_seed,
        record_details=bool(record_details),
        c_unit=float(cfg.c_unit),
        g_leak_unit=float(cfg.g_leak_unit),
        gamma_na=float(cfg.gamma_na),
        gamma_k=float(cfg.gamma_k),
        e_na=float(cfg.e_na),
        e_k=float(cfg.e_k),
        e_leak=float(cfg.e_leak),
        rho_na=float(cfg.rho_na),
        rho_k=float(cfg.rho_k),
    )

    t_ms: np.ndarray = np.linspace(
        0.0, (n_steps - 1) * float(dt), n_steps, dtype=np.float64
    )
    result: dict[str, np.ndarray] = {
        "t_ms": t_ms,
        "v_mv": v_trace,
    }

    if record_details:
        result["p_na_pct"] = p_na_pct
        result["p_k_pct"] = p_k_pct
        result["i_na_ua"] = i_na_ua
        result["i_k_ua"] = i_k_ua

    return result


def _run_self_test() -> None:
    """独立运行此文件时的自检函数，测试 JIT 编译速度与不同膜面积下的自发发放率。"""
    print("=== stochastic_hh/neuron.py 自检开始 ===")

    # 预热一次 JIT 编译
    _ = simulate_stochastic_hh(area_um2=100.0, duration_ms=10.0, seed=1, record_details=True)

    # 对比 A = 1 um^2 与 A = 500 um^2 在 2000 ms 无外部输入下的自发动作电位发放情况
    for test_area in (1.0, 100.0, 500.0):
        t0 = time.perf_counter()
        sim_out = simulate_stochastic_hh(
            area_um2=test_area,
            duration_ms=2500.0,
            seed=42,
            record_details=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 统计 500 ms 之后的脉冲正向过零点次数 (附带 2.0 ms 绝对不应期去抖动)
        v_valid = sim_out["v_mv"][50000:]
        raw_crossings = np.where((v_valid[:-1] < 0.0) & (v_valid[1:] >= 0.0))[0]
        min_step_gap = int(round(2.0 / DT_DEFAULT))
        valid_spikes = 0
        last_idx = -min_step_gap - 1
        for idx in raw_crossings:
            if idx - last_idx >= min_step_gap:
                valid_spikes += 1
                last_idx = int(idx)
        firing_rate_hz = valid_spikes / 2.0

        print(
            f"膜面积 A = {test_area:5.1f} um^2 | 模拟时长 = 2.5 s (250,000 步) | "
            f"耗时 = {elapsed_ms:5.1f} ms | 自发发放率 = {firing_rate_hz:5.1f} Hz | "
            f"V 范围 = [{float(np.min(v_valid)):.1f}, {float(np.max(v_valid)):.1f}] mV"
        )


if __name__ == "__main__":
    _run_self_test()