"""
stochastic_hh/config.py
=======================
集中管理 Guo et al. (2017) 随机 Hodgkin-Huxley (HH) 神经元模型的：
1. 项目输出路径（自动创建 results/data 和 results/figures 目录）
2. 生物物理常数与离子通道参数（严格对齐论文 Section 2）
3. 量纲换算因子（统一至 mV, ms, pA, pS, pF, um^2 单位制）
4. 数值积分与统计评价指标的默认参数（Section 3 & Section 4）
"""

from pathlib import Path

# =============================================================================
# 1. 项目目录与文件输出路径配置
# =============================================================================
# 自动定位项目根目录 (Stochastic_HH_Guo2017/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
DATA_DIR: Path = RESULTS_DIR / "data"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# 确保数据与图片保存目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. 细胞膜生物物理常数 (对应论文 Section 2, Equation 1)
# =============================================================================
# 单位面积膜电容 C = 0.01 pF / um^2 (等价于标准 HH 的 1 uF / cm^2)
C_MEMBRANE_UNIT: float = 0.01          # 单位: pF / um^2

# 单位面积漏电导 g_leak = 3 pS / um^2 (等价于标准 HH 的 0.3 mS / cm^2)
G_LEAK_UNIT: float = 3.0               # 单位: pS / um^2

# 单个开放离子通道的电导 (Single channel conductance)
GAMMA_NA: float = 20.0                 # 单个 Na+ 通道电导, 单位: pS
GAMMA_K: float = 20.0                  # 单个 K+ 通道电导, 单位: pS

# 离子反转电位 (Reversal potentials)
E_NA: float = 45.0                     # Na+ 反转电位, 单位: mV
E_K: float = -77.0                     # K+ 反转电位, 单位: mV
E_LEAK: float = -54.0                  # 漏电流反转电位, 单位: mV

# 单位面积离子通道面密度 (Channel density per um^2)
RHO_NA: int = 250                      # Na+ 通道密度: 250 channels / um^2
RHO_K: int = 50                        # K+ 通道密度: 50 channels / um^2


# =============================================================================
# 3. 量纲换算因子 (Dimensional Consistency Constants)
# =============================================================================
# 物理量纲关系说明：
# - 电压 V 单位为 mV (10^-3 V)，时间 t 单位为 ms (10^-3 s)，故 dV/dt 单位为 mV/ms = 1 V/s。
# - 电容 C 单位为 pF (10^-12 F)，故 C * (dV/dt) 单位为 10^-12 F * (1 V/s) = 1 pA。
# - 电导 G 单位为 pS (10^-12 S)，故 G * (V - E) 单位为 10^-12 S * 10^-3 V = 10^-15 A = 10^-3 pA。
# - 论文 Fig. 5(c),(e) 中离子电流以 uA (10^6 pA) 为单位绘制。
PS_MV_TO_PA: float = 1e-3              # 将 G[pS] * V[mV] 转换为电流 I[pA] 的乘子
PA_TO_UA: float = 1e-6                 # 将电流 I[pA] 转换为 I[uA] (用于 Fig. 5c, 5e)


# =============================================================================
# 4. 数值积分与统计评价默认参数 (对应论文 Section 2, 3, 4)
# =============================================================================
DT_DEFAULT: float = 0.01               # 欧拉积分步长 dt = 0.01 ms (采样率 100 kHz)
V_INIT_DEFAULT: float = -65.0          # 默认初始静息膜电位 (mV)

# 脉冲检测与去瞬态参数
T_TRANSIENT_MS: float = 500.0          # 刺激开始后剔除前 500 ms 以消除瞬态影响
SPIKE_THRESHOLD_MV: float = 0.0        # 动作电位正向过零点检测阈值 (0 mV)

# 外部波动电流默认参数
SIGMA_I_DEFAULT: float = 15.0          # 固定波动幅值实验中的默认标准差 sigma (pA)

# 脉冲触发平均 (STA, Section 3, Fig. 8) 参数
STA_WINDOW_MS: float = 20.0            # STA 截取窗口半宽: [-20 ms, +20 ms]

# 脉冲时间可靠性 (Reliability, Section 4, Fig. 10-12) 参数
RELIABILITY_TRIALS: int = 20           # 同一冻结刺激重复实验次数 n = 20
RELIABILITY_DURATION_MS: float = 5000.0  # 单次实验总时长 5 s = 5000 ms
SQUARE_WINDOW_D_MS: float = 5.0        # 方形卷积窗半宽 d = 5 ms (总宽度 2d = 10 ms)


class HHModelConfig:
    """
    可复用的神经元模型参数容器类。
    显式定义 __init__ 方法，彻底消除 PyCharm 静态检查的“类没有 __init__ 方法”警告。
    """

    def __init__(
        self,
        c_unit: float = C_MEMBRANE_UNIT,
        g_leak_unit: float = G_LEAK_UNIT,
        gamma_na: float = GAMMA_NA,
        gamma_k: float = GAMMA_K,
        e_na: float = E_NA,
        e_k: float = E_K,
        e_leak: float = E_LEAK,
        rho_na: int = RHO_NA,
        rho_k: int = RHO_K,
        dt: float = DT_DEFAULT,
    ) -> None:
        self.c_unit: float = c_unit
        self.g_leak_unit: float = g_leak_unit
        self.gamma_na: float = gamma_na
        self.gamma_k: float = gamma_k
        self.e_na: float = e_na
        self.e_k: float = e_k
        self.e_leak: float = e_leak
        self.rho_na: int = rho_na
        self.rho_k: int = rho_k
        self.dt: float = dt

    def total_channels(self, area_um2: float) -> tuple[float, float]:
        """根据膜面积 A (um^2) 返回 (Na+通道总数, K+通道总数)"""
        return float(self.rho_na) * area_um2, float(self.rho_k) * area_um2

    def passive_tau_ms(self) -> float:
        """返回细胞膜被动时间常数 tau_m = C / g_L (单位: ms)"""
        return self.c_unit / (self.g_leak_unit * PS_MV_TO_PA)


def _run_self_test() -> None:
    """模块独立自检函数，避免全局变量污染作用域"""
    cfg = HHModelConfig()
    na_tot, k_tot = cfg.total_channels(area_um2=500.0)
    print("=== stochastic_hh/config.py 自检通过 ===")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据输出目录: {DATA_DIR}")
    print(f"图片输出目录: {FIGURES_DIR}")
    print(f"被动膜时间常数 tau_m: {cfg.passive_tau_ms():.4f} ms")
    print(f"当膜面积 A = 500 um^2 时: Na+ 总通道数 = {na_tot:.0f}, K+ 总通道数 = {k_tot:.0f}")


if __name__ == "__main__":
    _run_self_test()