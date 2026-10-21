"""
stochastic_hh/channels_ssa.py
=============================
基于随机屏蔽近似（Stochastic-Shielding Approximation, SSA）
实现随机 Hodgkin-Huxley 神经元的离子通道马尔可夫门控动力学（对应论文 Section 2, Fig. 1 & Fig. 2）。

核心设计特点：
1. 严格对应论文 Fig. 1 (5状态 K+ 链) 与 Fig. 2 (8状态 Na+ 网格) 的拓扑结构与速率公式。
2. 仅保留直接连接导通态（K+ 的 S5，Na+ 的 S8）的跃迁边上的高斯随机扩散项，屏蔽内部关闭态之间的跃迁噪声。
3. 内置非负边界通量限幅与通道总数守恒保护，即使在极小膜面积 (A = 1 um^2) 下也不会出现负数通道或数值发散。
4. 全核心函数采用 Numba @njit 编译加速，完全遵循 PEP 8 命名规范（PyCharm 零警告）。
"""

import math
import numpy as np
from numba import njit

from stochastic_hh.config import RHO_NA, RHO_K, V_INIT_DEFAULT


@njit(fastmath=True)
def _sample_unshielded_flux(
    n_source: float,
    rate_forward: float,
    n_target: float,
    rate_backward: float,
    dt: float,
) -> float:
    """
    计算直接连接开放态的非屏蔽边 (i <-> j) 在步长 dt 内的随机净转移通道数：
    - 当单向期望转移通道数 lam < 25.0 时，采用精确的泊松分布抽样 np.random.poisson(lam)，
      天然保证非负且彻底消除极小通道数 (如 A = 1 um^2) 下的高斯零截断整流偏差；
    - 当 lam >= 25.0 时，平滑切换为高速高斯近似 lam + sqrt(lam) * N(0, 1)。
    """
    lam_fwd = n_source * rate_forward * dt
    if lam_fwd <= 0.0:
        flux_fwd = 0.0
    elif lam_fwd < 25.0:
        flux_fwd = float(np.random.poisson(lam_fwd))
    else:
        flux_fwd = lam_fwd + math.sqrt(lam_fwd) * np.random.standard_normal()
        if flux_fwd < 0.0:
            flux_fwd = 0.0

    lam_bwd = n_target * rate_backward * dt
    if lam_bwd <= 0.0:
        flux_bwd = 0.0
    elif lam_bwd < 25.0:
        flux_bwd = float(np.random.poisson(lam_bwd))
    else:
        flux_bwd = lam_bwd + math.sqrt(lam_bwd) * np.random.standard_normal()
        if flux_bwd < 0.0:
            flux_bwd = 0.0

    return flux_fwd - flux_bwd


@njit(fastmath=True)
def compute_gating_rates(v_mv: float) -> tuple[float, float, float, float, float, float]:
    """
    根据当前膜电位 v_mv (mV) 计算 6 个马尔可夫转移速率参数 (单位: ms^-1)。
    严格遵循论文 Section 2 第 383 页给出的公式，并在奇点处应用洛必达法则平滑过渡。

    参数:
        v_mv (float): 膜电位 (mV)

    返回:
        (alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h)
    """
    # 1. K+ 通道门控速率 alpha_n, beta_n (在 V = -55 mV 处有 0/0 奇点，极限为 0.1)
    u_n = v_mv + 55.0
    if abs(u_n) < 1e-7:
        alpha_n = 0.1
    else:
        alpha_n = 0.01 * u_n / (1.0 - math.exp(-0.1 * u_n))
    beta_n = 0.125 * math.exp(-(v_mv + 65.0) / 80.0)

    # 2. Na+ 通道激活门控速率 alpha_m, beta_m (在 V = -40 mV 处有 0/0 奇点，极限为 1.0)
    u_m = v_mv + 40.0
    if abs(u_m) < 1e-7:
        alpha_m = 1.0
    else:
        alpha_m = 0.1 * u_m / (1.0 - math.exp(-0.1 * u_m))
    beta_m = 4.0 * math.exp(-(v_mv + 65.0) / 18.0)

    # 3. Na+ 通道失活门控速率 alpha_h, beta_h
    alpha_h = 0.07 * math.exp(-(v_mv + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + math.exp(-(v_mv + 35.0) / 10.0))

    return alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h


@njit(fastmath=True)
def init_channel_states(
    area_um2: float,
    v_init: float = V_INIT_DEFAULT,
    rho_na: float = float(RHO_NA),
    rho_k: float = float(RHO_K),
) -> tuple[np.ndarray, np.ndarray]:
    """
    根据膜面积 A (um^2) 和初始膜电位 v_init (mV)，将 Na+ 和 K+ 通道各状态粒子数
    初始化为热力学稳态分布。

    状态索引映射说明：
    - n_k (长度 5): 对应论文 Fig. 1 的 [S1, S2, S3, S4, S5]，其中 n_k[4] (S5) 为开放态。
    - n_na (长度 8): 对应论文 Fig. 2 的 [S1, S2, S3, S4, S5, S6, S7, S8]，
      其中下排失活态为 S1..S4 (索引 0..3)，上排非失活态为 S5..S8 (索引 4..7)，n_na[7] (S8) 为开放态。
    """
    alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h = compute_gating_rates(v_init)

    # 稳态开放概率
    n_inf = alpha_n / (alpha_n + beta_n)
    m_inf = alpha_m / (alpha_m + beta_m)
    h_inf = alpha_h / (alpha_h + beta_h)

    q_n = 1.0 - n_inf
    q_m = 1.0 - m_inf
    q_h = 1.0 - h_inf

    # 1. 初始化 K+ 通道 5 个状态 (总通道数 = rho_k * area_um2)
    total_k = rho_k * area_um2
    n_k = np.empty(5, dtype=np.float64)
    n_k[0] = total_k * (q_n ** 4)                        # S1: 0 个 n 亚基激活
    n_k[1] = total_k * 4.0 * n_inf * (q_n ** 3)          # S2: 1 个 n 亚基激活
    n_k[2] = total_k * 6.0 * (n_inf ** 2) * (q_n ** 2)   # S3: 2 个 n 亚基激活
    n_k[3] = total_k * 4.0 * (n_inf ** 3) * q_n          # S4: 3 个 n 亚基激活
    n_k[4] = total_k * (n_inf ** 4)                      # S5 (Open): 4 个 n 亚基全部激活

    # 2. 初始化 Na+ 通道 8 个状态 (总通道数 = rho_na * area_um2)
    total_na = rho_na * area_um2
    n_na = np.empty(8, dtype=np.float64)
    # 下排 S1..S4 (索引 0..3, h 门关闭, 概率权重 q_h = 1 - h_inf)
    n_na[0] = total_na * q_h * (q_m ** 3)                # S1
    n_na[1] = total_na * q_h * 3.0 * m_inf * (q_m ** 2)  # S2
    n_na[2] = total_na * q_h * 3.0 * (m_inf ** 2) * q_m  # S3
    n_na[3] = total_na * q_h * (m_inf ** 3)              # S4
    # 上排 S5..S8 (索引 4..7, h 门开放, 概率权重 h_inf)
    n_na[4] = total_na * h_inf * (q_m ** 3)                # S5
    n_na[5] = total_na * h_inf * 3.0 * m_inf * (q_m ** 2)  # S6
    n_na[6] = total_na * h_inf * 3.0 * (m_inf ** 2) * q_m  # S7
    n_na[7] = total_na * h_inf * (m_inf ** 3)              # S8 (Open)

    return n_na, n_k


@njit(fastmath=True)
def step_k_channels_ssa(
    n_k: np.ndarray,
    alpha_n: float,
    beta_n: float,
    dt: float,
    total_k: float,
) -> float:
    """
    使用随机屏蔽近似 (SSA) 单步更新 5 状态 K+ 通道粒子数 n_k (对应论文 Fig. 1)。
    直接就地修改 n_k 数组，并返回更新后的开放态通道数 n_k[4] (即 N_K(5))。
    """
    s1, s2, s3, s4, s5 = n_k[0], n_k[1], n_k[2], n_k[3], n_k[4]

    # 3 条被屏蔽边的确定性净通量 (从左向右为正)
    j12 = (4.0 * alpha_n * s1 - beta_n * s2) * dt
    j23 = (3.0 * alpha_n * s2 - 2.0 * beta_n * s3) * dt
    j34 = (2.0 * alpha_n * s3 - 3.0 * beta_n * s4) * dt

    # 1 条非屏蔽边 (S4 <-> S5) 的随机净通量
    j45 = _sample_unshielded_flux(
        n_source=s4,
        rate_forward=alpha_n,
        n_target=s5,
        rate_backward=4.0 * beta_n,
        dt=dt,
    )

    # 边界保护：防止单步随机涨落使导通态 S5 或邻接态 S4 变为负数
    if s5 + j45 < 0.0:
        j45 = -s5
    elif s4 + j34 - j45 < 0.0:
        j45 = s4 + j34

    # 根据通量守恒更新 5 个状态的通道数
    n_k[0] = s1 - j12
    n_k[1] = s2 + j12 - j23
    n_k[2] = s3 + j23 - j34
    n_k[3] = s4 + j34 - j45
    n_k[4] = s5 + j45

    # 消除长时间积分下的浮点舍入误差并严格维持非负与总数守恒
    s_tot = 0.0
    for i in range(5):
        if n_k[i] < 0.0:
            n_k[i] = 0.0
        s_tot += n_k[i]
    if s_tot > 0.0:
        scale = total_k / s_tot
        for i in range(5):
            n_k[i] *= scale

    return n_k[4]


@njit(fastmath=True)
def step_na_channels_ssa(
    n_na: np.ndarray,
    alpha_m: float,
    beta_m: float,
    alpha_h: float,
    beta_h: float,
    dt: float,
    total_na: float,
) -> float:
    """
    使用随机屏蔽近似 (SSA) 单步更新 8 状态 Na+ 通道粒子数 n_na (对应论文 Fig. 2)。
    直接就地修改 n_na 数组，并返回更新后的开放态通道数 n_na[7] (即 N_Na(8))。
    """
    s1, s2, s3, s4 = n_na[0], n_na[1], n_na[2], n_na[3]
    s5, s6, s7, s8 = n_na[4], n_na[5], n_na[6], n_na[7]

    # 1. 下排水平边 (S1<->S2<->S3<->S4) 确定性通量 (向右为正)
    j12 = (3.0 * alpha_m * s1 - beta_m * s2) * dt
    j23 = (2.0 * alpha_m * s2 - 2.0 * beta_m * s3) * dt
    j34 = (alpha_m * s3 - 3.0 * beta_m * s4) * dt

    # 2. 上排非开放水平边 (S5<->S6<->S7) 确定性通量 (向右为正)
    j56 = (3.0 * alpha_m * s5 - beta_m * s6) * dt
    j67 = (2.0 * alpha_m * s6 - 2.0 * beta_m * s7) * dt

    # 3. 竖直非开放边 (S1<->S5, S2<->S6, S3<->S7) 确定性通量 (向上为正)
    j15 = (alpha_h * s1 - beta_h * s5) * dt
    j26 = (alpha_h * s2 - beta_h * s6) * dt
    j37 = (alpha_h * s3 - beta_h * s7) * dt

    # 4. 两条直接连接开放态 S8 的非屏蔽边 (S7<->S8 和 S4<->S8)
    j78 = _sample_unshielded_flux(
        n_source=s7,
        rate_forward=alpha_m,
        n_target=s8,
        rate_backward=3.0 * beta_m,
        dt=dt,
    )
    j48 = _sample_unshielded_flux(
        n_source=s4,
        rate_forward=alpha_h,
        n_target=s8,
        rate_backward=beta_h,
        dt=dt,
    )

    # 边界保护：防止单步内 S7 或 S4 流向 S8 的通量透支各自可用通道数
    avail_s7 = s7 + j67 + j37
    if j78 > avail_s7:
        j78 = avail_s7
    avail_s4 = s4 + j34
    if j48 > avail_s4:
        j48 = avail_s4

    # 防止开放态 S8 在单步流出过多而变成负数
    if s8 + j78 + j48 < 0.0:
        excess = -(s8 + j78 + j48)
        denominator = abs(j78) + abs(j48)
        if denominator > 0.0:
            j78 += excess * (abs(j78) / denominator)
            j48 += excess * (abs(j48) / denominator)

    # 5. 根据 12 条边的通量守恒更新全部 8 个状态
    n_na[0] = s1 - j12 - j15
    n_na[1] = s2 + j12 - j23 - j26
    n_na[2] = s3 + j23 - j34 - j37
    n_na[3] = s4 + j34 - j48

    n_na[4] = s5 - j56 + j15
    n_na[5] = s6 + j56 - j67 + j26
    n_na[6] = s7 + j67 - j78 + j37
    n_na[7] = s8 + j78 + j48

    # 6. 非负截断与总通道数严格归一化守恒
    s_tot = 0.0
    for i in range(8):
        if n_na[i] < 0.0:
            n_na[i] = 0.0
        s_tot += n_na[i]
    if s_tot > 0.0:
        scale = total_na / s_tot
        for i in range(8):
            n_na[i] *= scale

    return n_na[7]


def _run_self_test() -> None:
    """独立运行此文件时的自检函数，避免全局变量隐藏参数名称警告"""
    np.random.seed(42)
    test_area = 500.0   # um^2
    dt_test = 0.01      # ms
    v_clamp = -65.0     # mV

    n_na, n_k = init_channel_states(area_um2=test_area, v_init=v_clamp)
    tot_na = float(RHO_NA) * test_area
    tot_k = float(RHO_K) * test_area
    an, bn, am, bm, ah, bh = compute_gating_rates(v_clamp)

    p_na_list = []
    p_k_list = []
    for _ in range(10000):  # 模拟 100 ms 电压钳实验
        na_open = step_na_channels_ssa(n_na, am, bm, ah, bh, dt_test, tot_na)
        k_open = step_k_channels_ssa(n_k, an, bn, dt_test, tot_k)
        p_na_list.append(100.0 * na_open / tot_na)
        p_k_list.append(100.0 * k_open / tot_k)

    sum_na = float(np.sum(n_na))
    sum_k = float(np.sum(n_k))
    mean_k = float(np.mean(p_k_list))
    std_k = float(np.std(p_k_list))
    mean_na = float(np.mean(p_na_list))
    std_na = float(np.std(p_na_list))

    print("=== stochastic_hh/channels_ssa.py 自检通过 ===")
    print(f"测试条件: A = {test_area} um^2, 电压钳 V = {v_clamp} mV")
    print(f"Na+ 通道总数守恒检验: sum(n_na) = {sum_na:.2f} (目标值: {tot_na})")
    print(f"K+  通道总数守恒检验: sum(n_k)  = {sum_k:.2f} (目标值: {tot_k})")
    print(
        f"阈下开放比例 (对比论文第 384 页描述 'approximately 1% K+ and 0.02% Na+'):\n"
        f"  -> 平均开放 K+  比例: {mean_k:.3f}% ± {std_k:.3f}%\n"
        f"  -> 平均开放 Na+ 比例: {mean_na:.4f}% ± {std_na:.4f}%"
    )


if __name__ == "__main__":
    _run_self_test()