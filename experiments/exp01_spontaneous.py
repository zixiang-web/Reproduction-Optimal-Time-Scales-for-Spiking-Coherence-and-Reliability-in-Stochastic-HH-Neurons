"""
experiments/exp01_spontaneous.py
================================
复现论文 Section 3 前半部分（无外部输入 I_0 = 0 时的自发脉冲发放与通道动力学）：
1. Fig. 3(a)-(d): 不同细胞膜面积 A = 10, 100, 500, 700 um^2 下的自发膜电位时间序列 V(t)。
2. Fig. 4: 自发脉冲发放频率 (spike frequency, Hz) 随膜面积 A (1 ~ 1000 um^2) 变化的对数坐标曲线（含误差棒）。
3. Fig. 5(a)-(e): A = 500 um^2 时，膜电位 V(t)、开放 Na+/K+ 通道百分比 p_Na/p_K 与离子电流 I_Na/I_K 的同步演化图。

工程特性：
- 自动将耗时的参数扫描结果缓存至 results/data/exp01_spontaneous_fig4.npz；
  若需重新跑模拟，只需将 run_all(force_rerun=True) 设为 True。
- 所有生成的论文图表自动以 300 DPI 保存至 results/figures/ 目录。
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from stochastic_hh import (
    DATA_DIR,
    FIGURES_DIR,
    compute_firing_rate_hz,
    extract_spikes,
    simulate_stochastic_hh,
)


def reproduce_figure_3(show_plot: bool = True) -> None:
    """
    复现论文 Fig. 3(a)-(d):
    无外部输入时，不同膜面积 (10, 100, 500, 700 um^2) 下随机 HH 神经元的膜电位时间序列。
    严格对齐论文横坐标截取窗口：
    - (a) A = 10 um^2,  时间窗口 [2.0 s, 2.5 s]
    - (b) A = 100 um^2, 时间窗口 [2.0 s, 2.5 s]
    - (c) A = 500 um^2, 时间窗口 [2.0 s, 3.0 s]
    - (d) A = 700 um^2, 时间窗口 [1.0 s, 3.0 s]
    """
    print("\n[Fig. 3] 正在模拟 A = 10, 100, 500, 700 um^2 的自发膜电位时间序列...")
    t0 = time.perf_counter()

    panels_config = [
        ("a", 10.0, 2.0, 2.5, 101),
        ("b", 100.0, 2.0, 2.5, 202),
        ("c", 500.0, 2.0, 3.0, 303),
        ("d", 700.0, 1.0, 3.0, 404),
    ]

    fig = plt.figure(figsize=(10.5, 7.5), dpi=150)

    for idx, (label_char, area_val, t_start_s, t_end_s, seed_val) in enumerate(panels_config, start=1):
        duration_ms = t_end_s * 1000.0
        sim_res = simulate_stochastic_hh(
            area_um2=area_val,
            duration_ms=duration_ms,
            seed=seed_val,
            record_details=False,
        )
        t_sec = sim_res["t_ms"] / 1000.0
        v_mv = sim_res["v_mv"]

        mask = (t_sec >= t_start_s) & (t_sec <= t_end_s)
        ax = fig.add_subplot(2, 2, idx)
        ax.plot(t_sec[mask], v_mv[mask], color="black", linewidth=0.85)

        ax.set_xlim(t_start_s, t_end_s)
        ax.set_ylim(-80.0, 60.0)
        ax.set_xlabel("time (s)", fontsize=10)
        ax.set_ylabel("membrane potential (mV)", fontsize=10)
        ax.set_title(
            f"({label_char}) $A = {int(area_val)}\\ \\mu\\mathrm{{m}}^2$",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout(pad=1.8)
    save_path = FIGURES_DIR / "fig03_spontaneous_potentials.png"
    fig.savefig(save_path, dpi=300)
    elapsed_s = time.perf_counter() - t0
    print(f"[Fig. 3] 生成完毕 (耗时 {elapsed_s:.2f} s) -> 已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def reproduce_figure_4(
    n_trials: int = 10,
    duration_ms: float = 5500.0,
    force_rerun: bool = False,
    show_plot: bool = True,
) -> None:
    """
    复现论文 Fig. 4:
    自发脉冲发放频率 (Hz) 随细胞膜面积 A (1 ~ 1000 um^2) 变化的单调递减关系（对数横坐标 + 误差棒）。
    """
    cache_path = DATA_DIR / "exp01_fig04_spontaneous_rate.npz"

    # 对齐论文 Fig. 4 横轴上的 13 个采样点
    areas_um2 = np.asarray(
        [1.0, 2.0, 5.0, 7.0, 10.0, 20.0, 50.0, 70.0, 100.0, 200.0, 500.0, 700.0, 1000.0],
        dtype=np.float64,
    )

    if cache_path.exists() and not force_rerun:
        print(f"\n[Fig. 4] 检测到已缓存的数据文件，直接加载: {cache_path}")
        loaded = np.load(cache_path)
        areas_um2 = np.asarray(loaded["areas_um2"], dtype=np.float64)
        mean_rates_hz = np.asarray(loaded["mean_rates_hz"], dtype=np.float64)
        std_rates_hz = np.asarray(loaded["std_rates_hz"], dtype=np.float64)
    else:
        print(
            f"\n[Fig. 4] 正在扫描 13 个膜面积 A (1 ~ 1000 um^2)，每个面积重复 {n_trials} 次独立实验..."
        )
        t0 = time.perf_counter()
        mean_rates_hz = np.empty(len(areas_um2), dtype=np.float64)
        std_rates_hz = np.empty(len(areas_um2), dtype=np.float64)

        for i, area_val in enumerate(areas_um2):
            trial_rates = np.empty(n_trials, dtype=np.float64)
            for trial_idx in range(n_trials):
                seed_val = 10000 + i * 100 + trial_idx
                sim_res = simulate_stochastic_hh(
                    area_um2=float(area_val),
                    duration_ms=duration_ms,
                    seed=seed_val,
                    record_details=False,
                )
                spike_times, _ = extract_spikes(sim_res["v_mv"])
                trial_rates[trial_idx] = compute_firing_rate_hz(
                    spike_times, total_duration_ms=duration_ms
                )

            mean_rates_hz[i] = float(np.mean(trial_rates))
            std_rates_hz[i] = float(np.std(trial_rates))
            print(
                f"  -> [{i + 1:2d}/{len(areas_um2)}] A = {area_val:6.1f} um^2 | "
                f"自发频率 = {mean_rates_hz[i]:5.2f} ± {std_rates_hz[i]:4.2f} Hz"
            )

        np.savez(
            cache_path,
            areas_um2=areas_um2,
            mean_rates_hz=mean_rates_hz,
            std_rates_hz=std_rates_hz,
        )
        elapsed_s = time.perf_counter() - t0
        print(f"[Fig. 4] 数据计算完毕 (耗时 {elapsed_s:.2f} s) -> 已缓存至: {cache_path}")

    # 绘制论文 Fig. 4 风格的对数横坐标误差棒曲线
    fig = plt.figure(figsize=(6.8, 5.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    ax.errorbar(
        areas_um2,
        mean_rates_hz,
        yerr=std_rates_hz,
        fmt="-",
        color="magenta",
        ecolor="magenta",
        elinewidth=1.0,
        capsize=4,
        linewidth=1.2,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.9, 1100.0)
    ax.set_ylim(0.0, 62.0)
    ax.set_xlabel("$A\\ (\\mu\\mathrm{m}^2)$", fontsize=11)
    ax.set_ylabel("spike frequency (Hz)", fontsize=11)
    ax.set_title("Fig. 4: Spontaneous spiking frequency vs. membrane area", fontsize=11)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.tight_layout()
    save_path = FIGURES_DIR / "fig04_spontaneous_frequency.png"
    fig.savefig(save_path, dpi=300)
    print(f"[Fig. 4] 图像已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def reproduce_figure_5(show_plot: bool = True) -> None:
    """
    复现论文 Fig. 5(a)-(e):
    膜面积 A = 500 um^2 无外部输入时，在 [1.0 s, 3.0 s] 时间窗口内的五联同步时间序列：
    (a) 膜电位 v (mV)
    (b) 开放 Na+ 通道百分比 p_Na (%)
    (c) Na+ 电流 I_Na (uA, 显示为 x 10^-3 uA 标尺以对齐论文 Fig. 5c)
    (d) 开放 K+  通道百分比 p_K (%)
    (e) K+  电流 I_K  (uA, 显示为 x 10^-3 uA 标尺以对齐论文 Fig. 5e)
    """
    print("\n[Fig. 5] 正在模拟 A = 500 um^2 下的膜电位、通道开放百分比与离子电流...")
    t0 = time.perf_counter()

    sim_res = simulate_stochastic_hh(
        area_um2=500.0,
        duration_ms=3000.0,
        seed=505,
        record_details=True,
    )

    t_sec = sim_res["t_ms"] / 1000.0
    mask = (t_sec >= 1.0) & (t_sec <= 3.0)
    t_plot = t_sec[mask]

    v_plot = sim_res["v_mv"][mask]
    p_na_plot = sim_res["p_na_pct"][mask]
    # 论文 Fig. 5(c),(e) 纵轴带有 "x 10^-3" 乘子且标签为 uA，即数值以 10^-3 uA (nA) 绘制
    i_na_scaled = sim_res["i_na_ua"][mask] * 1e3
    p_k_plot = sim_res["p_k_pct"][mask]
    i_k_scaled = sim_res["i_k_ua"][mask] * 1e3

    # 统计阈下阶段 (V < -50 mV) 的平均开放通道比例，印证论文第 384 页正文结论
    subthreshold_mask = v_plot < -50.0
    sub_k_pct = float(np.mean(p_k_plot[subthreshold_mask]))
    sub_na_pct = float(np.mean(p_na_plot[subthreshold_mask]))
    print(
        f"  -> 阈下振荡区间平均开放比例: K+ = {sub_k_pct:.3f}% (论文: ~1%), "
        f"Na+ = {sub_na_pct:.4f}% (论文: ~0.02%)"
    )

    fig = plt.figure(figsize=(8.5, 9.5), dpi=150)

    # (a) 膜电位 V (mV)
    ax1 = fig.add_subplot(5, 1, 1)
    ax1.plot(t_plot, v_plot, color="black", linewidth=0.85)
    ax1.set_xlim(1.0, 3.0)
    ax1.set_ylim(-100.0, 60.0)
    ax1.set_ylabel("v (mV)", fontsize=10)
    ax1.set_title("(a)", loc="left", fontsize=11, fontweight="bold")
    ax1.tick_params(direction="in", top=True, right=True)

    # (b) Na+ 开放百分比 p_Na (%)
    ax2 = fig.add_subplot(5, 1, 2)
    ax2.plot(t_plot, p_na_plot, color="magenta", linewidth=0.85)
    ax2.set_xlim(1.0, 3.0)
    ax2.set_ylim(0.0, 36.0)
    ax2.set_ylabel("$p_{\\mathrm{Na}}$ (%)", fontsize=10)
    ax2.set_title("(b)", loc="left", fontsize=11, fontweight="bold")
    ax2.tick_params(direction="in", top=True, right=True)

    # (c) Na+ 电流 I_Na (x 10^-3 uA)
    ax3 = fig.add_subplot(5, 1, 3)
    ax3.plot(t_plot, i_na_scaled, color="magenta", linewidth=0.85)
    ax3.set_xlim(1.0, 3.0)
    ax3.set_ylim(-15.5, 2.0)
    ax3.set_ylabel("$I_{\\mathrm{Na}}\\ (\\mu\\mathrm{A})\\ [\\times 10^{-3}]$", fontsize=9)
    ax3.set_title("(c)", loc="left", fontsize=11, fontweight="bold")
    ax3.tick_params(direction="in", top=True, right=True)

    # (d) K+ 开放百分比 p_K (%)
    ax4 = fig.add_subplot(5, 1, 4)
    ax4.plot(t_plot, p_k_plot, color="blue", linewidth=0.85)
    ax4.set_xlim(1.0, 3.0)
    ax4.set_ylim(0.0, 45.0)
    ax4.set_ylabel("$p_{\\mathrm{K}}$ (%)", fontsize=10)
    ax4.set_title("(d)", loc="left", fontsize=11, fontweight="bold")
    ax4.tick_params(direction="in", top=True, right=True)

    # (e) K+ 电流 I_K (x 10^-3 uA)
    ax5 = fig.add_subplot(5, 1, 5)
    ax5.plot(t_plot, i_k_scaled, color="blue", linewidth=0.85)
    ax5.set_xlim(1.0, 3.0)
    ax5.set_ylim(-1.0, 15.5)
    ax5.set_xlabel("time (s)", fontsize=10)
    ax5.set_ylabel("$I_{\\mathrm{K}}\\ (\\mu\\mathrm{A})\\ [\\times 10^{-3}]$", fontsize=9)
    ax5.set_title("(e)", loc="left", fontsize=11, fontweight="bold")
    ax5.tick_params(direction="in", top=True, right=True)

    fig.tight_layout(pad=1.2)
    save_path = FIGURES_DIR / "fig05_channel_activities_and_currents.png"
    fig.savefig(save_path, dpi=300)
    elapsed_s = time.perf_counter() - t0
    print(f"[Fig. 5] 生成完毕 (耗时 {elapsed_s:.2f} s) -> 已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_all(force_rerun: bool = False, show_plot: bool = True) -> None:
    """顺序执行 Fig. 3, Fig. 4, Fig. 5 的完整复现流程。"""
    print("=" * 72)
    print("开始执行 exp01_spontaneous.py (复现论文 Fig. 3, Fig. 4, Fig. 5)")
    print("=" * 72)

    # 预热一次 Numba JIT
    _ = simulate_stochastic_hh(area_um2=100.0, duration_ms=10.0, seed=0, record_details=True)

    reproduce_figure_3(show_plot=show_plot)
    reproduce_figure_4(n_trials=10, duration_ms=5500.0, force_rerun=force_rerun, show_plot=show_plot)
    reproduce_figure_5(show_plot=show_plot)

    print("\n" + "=" * 72)
    print("exp01_spontaneous.py 全部图表复现完毕！请查看 results/figures/ 目录。")
    print("=" * 72)


if __name__ == "__main__":
    run_all(force_rerun=False, show_plot=True)