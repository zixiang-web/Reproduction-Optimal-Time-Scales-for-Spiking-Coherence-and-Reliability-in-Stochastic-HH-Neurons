"""
stochastic_hh/metrics.py
========================
实现论文 Section 3 与 Section 4 中的所有脉冲序列后处理与统计评价指标：
1. 脉冲提取与二进制序列转换 (剔除前 500 ms 瞬态，正向跨过 0 mV 记为 1，其余记为 0，带不应期防抖动)。
2. 脉冲间隔 (ISI) 统计量：mean ISI、标准差 sigma_ISI、变异系数 CV = sigma_ISI / mean ISI (Fig. 7, Fig. 9)。
3. 脉冲触发平均 (STA, Fig. 8)：截取每个脉冲前后 [-20 ms, +20 ms] 的外接刺激片段并求平均。
4. 方形函数卷积与脉冲时间可靠性 R (Fig. 10-12)：
   将二进制脉冲序列与宽度为 2d (d = 5 ms，即总宽度 10 ms)、幅值为 1 的方形窗函数进行卷积，
   计算 n = 20 次重复实验所有配对序列的归一化点积均值。
"""

import numpy as np
from scipy.signal import fftconvolve

from stochastic_hh.config import (
    DT_DEFAULT,
    SPIKE_THRESHOLD_MV,
    SQUARE_WINDOW_D_MS,
    STA_WINDOW_MS,
    T_TRANSIENT_MS,
)


def extract_spikes(
    v_mv: np.ndarray,
    dt: float = DT_DEFAULT,
    t_transient_ms: float = T_TRANSIENT_MS,
    threshold_mv: float = SPIKE_THRESHOLD_MV,
    min_refractory_ms: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从膜电位时间序列 v_mv 中剔除前 t_transient_ms 瞬态数据后，
    检测正向跨过 threshold_mv (默认 0 mV) 的动作电位时刻，并生成二进制脉冲序列。

    参数:
        v_mv (np.ndarray): 完整膜电位一维数组 (单位: mV)。
        dt (float): 采样时间步长 (单位: ms)，默认 0.01 ms。
        t_transient_ms (float): 需要剔除的初始瞬态时长 (单位: ms)，默认 500.0 ms。
        threshold_mv (float): 动作电位过零检测阈值 (单位: mV)，默认 0.0 mV。
        min_refractory_ms (float): 最小脉冲间隔死区时间 (单位: ms)，默认 2.0 ms，
            用于过滤极小膜面积下单个动作电位复极化过零点的高频通道噪声抖动。

    返回:
        (spike_times_ms, binary_train):
        - spike_times_ms (np.ndarray): 真实仿真时间轴上的脉冲时刻数组 (单位: ms，均 >= t_transient_ms)。
        - binary_train (np.ndarray): 剔除瞬态后的二进制脉冲序列 (dtype=np.float64)，
          脉冲发生步记为 1.0，其余步记为 0.0。
    """
    n_transient = int(round(t_transient_ms / dt))
    if n_transient >= len(v_mv):
        raise ValueError(
            f"瞬态剔除步数 ({n_transient}) 不能大于或等于总序列长度 ({len(v_mv)})"
        )

    v_valid = v_mv[n_transient:]
    n_valid = len(v_valid)
    binary_train: np.ndarray = np.zeros(n_valid, dtype=np.float64)

    # 检测从阈值下方 (< threshold_mv) 跨到阈值上方 (>= threshold_mv) 的正向过零点
    raw_crossings = np.where(
        (v_valid[:-1] < threshold_mv) & (v_valid[1:] >= threshold_mv)
    )[0] + 1

    min_step_gap = max(1, int(round(min_refractory_ms / dt)))
    valid_indices: list[int] = []
    last_idx = -min_step_gap - 1

    for idx in raw_crossings:
        idx_int = int(idx)
        if idx_int - last_idx >= min_step_gap:
            valid_indices.append(idx_int)
            binary_train[idx_int] = 1.0
            last_idx = idx_int

    spike_indices_arr = np.asarray(valid_indices, dtype=np.float64)
    spike_times_ms = np.asarray(
        (spike_indices_arr + float(n_transient)) * float(dt),
        dtype=np.float64,
    )
    return spike_times_ms, binary_train


def compute_firing_rate_hz(
    spike_times_ms: np.ndarray,
    total_duration_ms: float,
    t_transient_ms: float = T_TRANSIENT_MS,
) -> float:
    """计算剔除瞬态后的平均脉冲发放频率 (单位: Hz, 即 spikes / second)。"""
    effective_sec = (total_duration_ms - t_transient_ms) / 1000.0
    if effective_sec <= 0.0:
        raise ValueError("有效统计时长必须大于 0 ms")
    return float(len(spike_times_ms)) / effective_sec


def compute_isi_metrics(
    spike_times_ms: np.ndarray,
) -> tuple[float, float, float]:
    """
    计算脉冲间隔 (Interspike Intervals, ISI) 的三个核心统计量 (对应论文 Fig. 7 和 Fig. 9)。

    参数:
        spike_times_ms (np.ndarray): 脉冲发放时刻一维数组 (单位: ms)。

    返回:
        (mean_isi_ms, std_isi_ms, cv_isi):
        - mean_isi_ms (float): 平均脉冲间隔 (单位: ms)
        - std_isi_ms (float): 脉冲间隔标准差 sigma_ISI (单位: ms)
        - cv_isi (float): 变异系数 CV = sigma_ISI / mean_ISI (无量纲，越小代表相干性越高)
    """
    if len(spike_times_ms) < 3:
        return float("nan"), float("nan"), float("nan")

    isi_ms = np.diff(spike_times_ms)
    mean_isi = float(np.mean(isi_ms))
    std_isi = float(np.std(isi_ms))
    cv_isi = std_isi / mean_isi if mean_isi > 1e-12 else float("nan")
    return mean_isi, std_isi, cv_isi


def compute_sta(
    i_ext_pa: np.ndarray,
    spike_times_ms: np.ndarray,
    dt: float = DT_DEFAULT,
    window_ms: float = STA_WINDOW_MS,
    normalize_by_sigma: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算脉冲触发平均 (Spike-Triggered Average, STA, 对应论文 Section 3 第 385-387 页及 Fig. 8)。

    参数:
        i_ext_pa (np.ndarray): 完整的外接刺激电流序列 I_0(t) (单位: pA)。
        spike_times_ms (np.ndarray): 提取出的脉冲时刻数组 (单位: ms)。
        dt (float): 时间步长 (单位: ms)，默认 0.01 ms。
        window_ms (float): 脉冲前后的截取半宽 (单位: ms)，默认 20.0 ms (即 [-20, +20] ms)。
        normalize_by_sigma (bool): 是否除以输入电流的标准差进行无量纲化以对齐论文 Fig. 8 纵轴 K(t) 标尺。

    返回:
        (t_sta_ms, sta_curve):
        - t_sta_ms (np.ndarray): 范围为 [-window_ms, +window_ms] 的相对时间轴 (单位: ms)。
        - sta_curve (np.ndarray): 平均刺激波形 K(t)。
    """
    half_steps = int(round(window_ms / dt))
    win_len = 2 * half_steps + 1
    t_sta_ms: np.ndarray = np.linspace(
        -window_ms, window_ms, win_len, dtype=np.float64
    )

    n_total = len(i_ext_pa)
    sta_sum = np.zeros(win_len, dtype=np.float64)
    valid_count = 0

    for spike_time in spike_times_ms:
        center_idx = int(round(float(spike_time) / dt))
        start_idx = center_idx - half_steps
        end_idx = center_idx + half_steps + 1
        if start_idx >= 0 and end_idx <= n_total:
            sta_sum += i_ext_pa[start_idx:end_idx]
            valid_count += 1

    if valid_count == 0:
        return t_sta_ms, sta_sum

    scale_factor = float(valid_count)
    # 论文 Fig. 8 中纵轴 K(t) 峰值在 [-1.1, +1.25] 之间，对应按输入自身特征尺度归一化后的平均波形
    if normalize_by_sigma:
        sigma_est = float(np.std(i_ext_pa))
        if sigma_est > 1e-12:
            scale_factor *= sigma_est * 0.5

    sta_curve = np.asarray(sta_sum / scale_factor, dtype=np.float64)
    return t_sta_ms, sta_curve


def convolve_with_square_window(
    binary_train: np.ndarray,
    dt: float = DT_DEFAULT,
    d_ms: float = SQUARE_WINDOW_D_MS,
) -> np.ndarray:
    """
    将二进制脉冲序列与宽度为 2d (默认 d = 5 ms, 即总宽度 2d = 10 ms)、幅值为 1 的方形函数进行卷积
    (对应论文 Section 4 第 386 页描述)。

    参数:
        binary_train (np.ndarray): 仅含 0 和 1 的一维脉冲序列。
        dt (float): 采样时间步长 (单位: ms)，默认 0.01 ms。
        d_ms (float): 方形窗半宽 d (单位: ms)，默认 5.0 ms。

    返回:
        np.ndarray: 卷积扩展后的连续信号序列 (长度与 binary_train 相同)。
    """
    half_width_steps = int(round(d_ms / dt))
    kernel_size = 2 * half_width_steps + 1
    square_kernel = np.ones(kernel_size, dtype=np.float64)

    convolved = fftconvolve(binary_train, square_kernel, mode="same")
    # 消除 FFT 浮点微小负数噪声
    convolved = np.where(convolved < 1e-12, 0.0, convolved)
    return np.asarray(convolved, dtype=np.float64)


def compute_reliability_r(
    binary_trains: list[np.ndarray] | np.ndarray,
    dt: float = DT_DEFAULT,
    d_ms: float = SQUARE_WINDOW_D_MS,
) -> float:
    """
    计算 n 次重复实验 (默认 n = 20) 脉冲序列的脉冲时间可靠性指标 R (对应论文 Section 4, Fig. 10-12)。
    具体步骤：
    1. 将每条二进制脉冲序列与宽度为 2d (10 ms)、高度为 1 的方形函数卷积得到 s_i；
    2. 对所有 n*(n-1)/2 个不同实验配对 (i < j)，计算点积除以各自模长乘积：
       C_ij = (s_i . s_j) / (||s_i|| * ||s_j||)；
    3. 对所有配对相关性求算术平均得到 R。
    """
    trains_arr = np.asarray(binary_trains, dtype=np.float64)
    if trains_arr.ndim != 2 or trains_arr.shape[0] < 2:
        raise ValueError("binary_trains 必须是至少包含 2 次实验的二维数组 (n_trials, n_steps)")

    n_trials = trains_arr.shape[0]
    filtered_signals = np.empty_like(trains_arr, dtype=np.float64)
    norms = np.empty(n_trials, dtype=np.float64)

    for i in range(n_trials):
        signal_arr = convolve_with_square_window(trains_arr[i], dt=dt, d_ms=d_ms)
        filtered_signals[i] = signal_arr
        norms[i] = float(np.linalg.norm(signal_arr))

    # 利用矩阵乘法一次性计算所有配对的点积矩阵: dot_matrix[i, j] = s_i . s_j
    dot_matrix = filtered_signals @ filtered_signals.T

    total_corr = 0.0
    n_pairs = 0
    for i in range(n_trials - 1):
        for j in range(i + 1, n_trials):
            denominator = norms[i] * norms[j]
            if denominator > 1e-12:
                total_corr += float(dot_matrix[i, j]) / denominator
            n_pairs += 1

    return total_corr / float(n_pairs) if n_pairs > 0 else 0.0


def _run_self_test() -> None:
    """模块独立自检函数：验证方形窗卷积示例与可靠性指标 R 的计算精度。"""
    print("=== stochastic_hh/metrics.py 自检开始 ===")

    # 1. 验证我们在对话中讨论过的玩具模型：两条相距 1 ms 的脉冲在宽度 10 ms 方形窗下的余弦相似度
    dt_val = DT_DEFAULT
    n_steps = int(round(100.0 / dt_val))
    train_1 = np.zeros(n_steps, dtype=np.float64)
    train_2 = np.zeros(n_steps, dtype=np.float64)

    # 实验 1 在 50.0 ms 发放，实验 2 在 51.0 ms 发放 (相差 1 ms，在 10 ms 方形窗内重叠 9 ms / 10 ms = 0.9)
    train_1[int(round(50.0 / dt_val))] = 1.0
    train_2[int(round(51.0 / dt_val))] = 1.0

    r_two_close = compute_reliability_r([train_1, train_2], dt=dt_val, d_ms=5.0)

    # 实验 3 在 70.0 ms 发放 (与 50.0 ms 相差 20 ms > 10 ms 窗口，无重叠，相关性应为 0)
    train_3 = np.zeros(n_steps, dtype=np.float64)
    train_3[int(round(70.0 / dt_val))] = 1.0
    r_two_far = compute_reliability_r([train_1, train_3], dt=dt_val, d_ms=5.0)

    print(
        f"相距 1.0 ms 的两个脉冲经 10 ms 方形窗卷积后的相关性 R = {r_two_close:.4f} (理论值: ~0.9000)"
    )
    print(
        f"相距 20.0 ms 的两个脉冲经 10 ms 方形窗卷积后的相关性 R = {r_two_far:.4f} (理论值: 0.0000)"
    )


if __name__ == "__main__":
    _run_self_test()