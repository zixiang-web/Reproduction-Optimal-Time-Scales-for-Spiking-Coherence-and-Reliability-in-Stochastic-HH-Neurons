"""
main_verify.py
==============
一键自检脚本：验证 stochastic_hh 核心包的所有模块连通性，并快速对标论文中的 3 个关键基准数值：
1. Section 3 (Fig. 4): 无外接输入时，A = 1 um^2 (~52 Hz) 与 A = 700 um^2 (~4 Hz) 的自发发放频率；
2. Section 3 (Fig. 7): A = 700 um^2 下，最优相关时间 tau = 1.4 ms 与快/慢波动下的变异系数 CV 对比；
3. Section 4 (Fig. 10): A = 700 um^2 下，5 次重复实验在 tau = 1.5 ms 下的脉冲时间可靠性指标 R。
"""

import time
import numpy as np

from stochastic_hh import (
    compute_firing_rate_hz,
    compute_isi_metrics,
    compute_reliability_r,
    extract_spikes,
    generate_alpha_colored_noise,
    simulate_stochastic_hh,
)


def run_verification() -> None:
    """运行三大核心实验场景的快速基准测试。"""
    print("=" * 72)
    print("Stochastic_HH_Guo2017 核心算法包全链路基准验证")
    print("=" * 72)

    # 预热 Numba JIT 编译器
    _ = simulate_stochastic_hh(area_um2=100.0, duration_ms=20.0, seed=0)

    # -------------------------------------------------------------------------
    # 测试 1：无外部输入下的自发发放频率 (对标论文 Section 3, Fig. 4)
    # -------------------------------------------------------------------------
    print("\n[测试 1] 无外部输入下的自发脉冲发放率 (对标论文第 383 页 & Fig. 4):")
    duration_spont_ms = 5500.0  # 500 ms 瞬态 + 5000 ms (5 s) 有效统计时长
    for area_val, paper_ref in ((1.0, "~52 Hz"), (100.0, "~32 Hz"), (700.0, "~4 Hz")):
        t0 = time.perf_counter()
        out = simulate_stochastic_hh(
            area_um2=area_val,
            duration_ms=duration_spont_ms,
            seed=100 + int(area_val),
        )
        spk_times, _ = extract_spikes(out["v_mv"])
        rate_hz = compute_firing_rate_hz(spk_times, total_duration_ms=duration_spont_ms)
        cost_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"  -> A = {area_val:5.1f} um^2 | 模拟自发频率 = {rate_hz:5.2f} Hz "
            f"(论文参考值: {paper_ref}) | 耗时 = {cost_ms:5.1f} ms"
        )

    # -------------------------------------------------------------------------
    # 测试 2：不同相关时间 tau 下的相干共振 CV 对比 (对标论文 Section 3, Fig. 7a-c)
    # -------------------------------------------------------------------------
    print("\n[测试 2] A = 700 um^2 下不同输入相关时间 tau 的脉冲相干性 CV (对标 Fig. 7c):")
    duration_cr_ms = 10500.0  # 10 s 有效时长
    for tau_val in (0.2, 1.4, 5.0):
        i_ext = generate_alpha_colored_noise(
            duration_ms=duration_cr_ms,
            tau_ms=tau_val,
            sigma_pa=15.0,
            seed=2026,
        )
        out = simulate_stochastic_hh(
            area_um2=700.0,
            i_ext_pa=i_ext,
            seed=300,
        )
        spk_times, _ = extract_spikes(out["v_mv"])
        mean_isi, std_isi, cv_val = compute_isi_metrics(spk_times)
        print(
            f"  -> tau = {tau_val:3.1f} ms | mean ISI = {mean_isi:5.2f} ms | "
            f"sigma_ISI = {std_isi:5.2f} ms | CV = {cv_val:.4f}"
        )

    # -------------------------------------------------------------------------
    # 测试 3：脉冲时间可靠性 R 快速测试 (对标论文 Section 4, Fig. 10)
    # -------------------------------------------------------------------------
    print("\n[测试 3] A = 700 um^2 下脉冲时间可靠性指标 R (6 次重复实验快速预览, 对标 Fig. 10):")
    n_quick_trials = 6
    duration_rel_ms = 5000.0
    for tau_val in (0.2, 1.5, 5.0):
        # 冻结噪声：所有重复实验使用完全相同的外接刺激电流 (seed=888)
        frozen_i_ext = generate_alpha_colored_noise(
            duration_ms=duration_rel_ms,
            tau_ms=tau_val,
            sigma_pa=15.0,
            seed=888,
        )
        binary_list: list[np.ndarray] = []
        for trial_idx in range(n_quick_trials):
            sim_res = simulate_stochastic_hh(
                area_um2=700.0,
                i_ext_pa=frozen_i_ext,
                seed=1000 + trial_idx * 17,  # 每次实验具有独立的通道噪声
            )
            _, bin_train = extract_spikes(sim_res["v_mv"])
            binary_list.append(bin_train)

        r_val = compute_reliability_r(binary_list)
        print(f"  -> tau = {tau_val:3.1f} ms | 脉冲时间可靠性 R = {r_val:.4f}")

    print("\n" + "=" * 72)
    print("全部核心算子验证完毕！可以正式开始执行 experiments/ 下的图表复现。")
    print("=" * 72)


if __name__ == "__main__":
    run_verification()