[project_readme.md](https://github.com/user-attachments/files/33765802/project_readme.md)
# Optimal Time Scales for Spiking Coherence and Reliability in Stochastic Hodgkin–Huxley Neurons (Reproduction & Modular Toolkit)

**[English](#english-version) | [中文说明](#中文说明)**

---

<a name="english-version"></a>
## English Version

### 1. Overview
This repository provides a complete, Numba-accelerated Python reproduction of the paper:

> **Xinmeng Guo, Jiang Wang, Jing Liu, Haitao Yu, Roberto F. Galán, Yibin Cao, and Bin Deng\***,  
> *"Optimal time scales of input fluctuations for spiking coherence and reliability in stochastic Hodgkin–Huxley neurons"*,  
> **Physica A: Statistical Mechanics and its Applications**, vol. 468, pp. 381–390, 2017.  
> DOI: [10.1016/j.physa.2016.10.087](https://doi.org/10.1016/j.physa.2016.10.087)

We reproduce **all simulation experiments and data figures (Figs. 3–12)** in the paper:
* **Spontaneous Firing Dynamics (Section 3, Figs. 3–5)**: Spontaneous membrane potential traces, log-scale spontaneous firing frequency vs. membrane area $A$, and subthreshold/suprathreshold ion channel opening percentages ($p_{\mathrm{Na}}, p_{\mathrm{K}}$) and ionic currents ($I_{\mathrm{Na}}, I_{\mathrm{K}}$) under zero external input ($I_0 = 0$).
* **Coherence Resonance & Spike-Triggered Average (Section 3, Figs. 6–9)**: Voltage responses to aperiodic fluctuating inputs with varying correlation times $\tau$, interspike interval (ISI) statistics ($\text{mean ISI}$, $\sigma_{\mathrm{ISI}}$, and coefficient of variation $\mathrm{CV}$), Spike-Triggered Average ($\mathrm{STA}$) curves $K(t)$, and $\mathrm{CV}$ across membrane areas $A \in [10, 1500]\ \mu\mathrm{m}^2$ and input amplitudes $\sigma \in \{7, 10, 15, 20\}\ \mathrm{pA}$.
* **Spike-Time Reliability (Section 4, Figs. 10–12)**: Bell-shaped reliability index $R$ vs. correlation time $\tau$, raster-like voltage realizations under frozen fluctuating input, and reliability $R$ across membrane areas $A$ and stimulus amplitudes $\sigma$.

---

### 2. Modular Architecture & Directory Structure
The project is architected with a strict separation between the **reusable core biophysical library (`stochastic_hh/`)** and the **paper reproduction scripts (`experiments/`)**. The `stochastic_hh/` package can be directly copied and imported into future projects (e.g., coupled stochastic neural networks, conductance-based synaptic inputs, or ion-channel blocking studies).

```text
Stochastic_HH_Guo2017/
│
├── stochastic_hh/                  # [Core Reusable Package] Importable standalone toolkit
│   ├── __init__.py                 # Package interface exposing core classes and functions
│   ├── config.py                   # Biophysical constants, unit conversions, and default parameters
│   ├── channels_ssa.py             # 5-state K+ & 8-state Na+ Markov gating with Stochastic-Shielding (SSA)
│   ├── neuron.py                   # Single-step & full-trajectory stochastic HH membrane voltage solver
│   ├── stimulus.py                 # Frozen white noise + alpha-function convolution generator
│   └── metrics.py                  # Spike detection, ISI/CV, STA, square-window convolution & reliability R
│
├── experiments/                    # [Reproduction Scripts] Organized by paper sections & figures
│   ├── exp01_spontaneous.py        # Reproduces Section 3 spontaneous firing: Figs. 3, 4, 5
│   ├── exp02_coherence_sta.py      # Reproduces Section 3 coherence resonance & STA: Figs. 6, 7, 8, 9
│   └── exp03_reliability.py        # Reproduces Section 4 spike-time reliability: Figs. 10, 11, 12
│
├── results/                        # [Outputs] Automatically created at runtime
│   ├── data/                       # Cached parameter-sweep arrays (.npz) to avoid redundant runs
│   └── figures/                    # High-resolution reproduced figures (300 DPI .png)
│
├── main_verify.py                  # [Quick Verification] End-to-end benchmark of all core modules
└── README.md                       # Project documentation (English & Chinese)
```

---

### 3. Model & Numerical Methods Summary
1. **Membrane Potential Equation**:
   $$C \frac{dV}{dt} = -g_{\mathrm{Na}}(V - E_{\mathrm{Na}}) - g_{\mathrm{K}}(V - E_{\mathrm{K}}) - g_L(V - E_L) + I_0(t)$$
   Integrated with a semi-implicit Euler scheme at time step $dt = 0.01\ \mathrm{ms}$ ($100\ \mathrm{kHz}$). Total channel counts scale with membrane area $A\ (\mu\mathrm{m}^2)$ via densities $\rho_{\mathrm{Na}} = 250\ \mu\mathrm{m}^{-2}$ and $\rho_{\mathrm{K}} = 50\ \mu\mathrm{m}^{-2}$, with single-channel conductances $\gamma_{\mathrm{Na}} = \gamma_{\mathrm{K}} = 20\ \mathrm{pS}$.
2. **Stochastic-Shielding Approximation (SSA)**:
   Rather than simulating stochastic transitions across all edges of the 5-state $\mathrm{K}^+$ Markov chain (Fig. 1) and 8-state $\mathrm{Na}^+$ Markov grid (Fig. 2), SSA retains stochasticity **only on transitions directly connected to the open/conducting states** ($S_4 \leftrightarrow S_5$ for $\mathrm{K}^+$; $S_7 \leftrightarrow S_8$ and $S_4 \leftrightarrow S_8$ for $\mathrm{Na}^+$), while internal closed-to-closed transitions evolve deterministically. A hybrid Poisson–Gaussian flux sampler ensures exact non-negativity even at $A = 1\ \mu\mathrm{m}^2$.
3. **External Fluctuating Input $I_0(t)$**:
   Generated by convolving frozen Gaussian white noise $\xi(t)$ with an $\alpha$-function kernel $h(t) = (t/\tau)\exp(-t/\tau)$ and rescaling to target amplitude $\sigma$ (default $\sigma = 15\ \mathrm{pA}$).
4. **Order Parameters**:
   * **Spiking Coherence**: Quantified by the coefficient of variation $\mathrm{CV} = \sigma_{\mathrm{ISI}} / \text{mean ISI}$ after discarding the initial $500\ \mathrm{ms}$ transient. Lower $\mathrm{CV}$ indicates higher coherence.
   * **Spike-Time Reliability ($R$)**: Binary spike trains ($n = 20$ repeated trials of duration $5\ \mathrm{s}$ under identical frozen noise) are convolved with a square window of width $2d = 10\ \mathrm{ms}$ ($d = 5\ \mathrm{ms}$) and unit amplitude. $R$ is the mean normalized inner product (cosine similarity) across all $n(n-1)/2 = 190$ trial pairs.

---

### 4. Key Reproduction Insight: Area-Coupled Input Scaling in `exp02` (Fig. 9)
During the reproduction of **Fig. 9** ($\mathrm{CV}$ vs. membrane area $A \in [10, 1500]\ \mu\mathrm{m}^2$), we identified a crucial implicit physical assumption in the paper's formulation:

* **The Issue**: In Eq. (1), total membrane capacitance $C_{\mathrm{tot}} = C_{\mathrm{unit}} A$ and ionic conductances $G_{\mathrm{Na}}, G_{\mathrm{K}}, G_L$ all scale linearly with membrane area $A$. If the injected current $I_0(t)$ is kept at a fixed absolute amplitude (e.g., $\sigma = 15\ \mathrm{pA}$) regardless of $A$, dividing Eq. (1) by $A$ reveals that the **effective driving current density per unit capacitance is $I_0(t)/A$**, which varies by a factor of $150\times$ across $A \in [10, 1500]\ \mu\mathrm{m}^2$:
  * At very small areas ($A = 10\ \mu\mathrm{m}^2$), a raw $15\ \mathrm{pA}$ current acts as a $70\times$ over-stimulus relative to $A = 700\ \mu\mathrm{m}^2$, driving the neuron into erratic forced bursting and causing $\mathrm{CV}$ to abnormally spike upward ($\mathrm{CV} > 0.5$).
  * At large areas ($A = 1500\ \mu\mathrm{m}^2$), the fixed current is severely diluted by the large membrane capacitance, failing to drive suprathreshold spikes and causing $\mathrm{CV}$ to diverge linearly instead of reaching the horizontal plateau shown in Fig. 9.
* **The Solution**: In Guo et al. (2017), varying the membrane area $A$ is intended purely to modulate the **intrinsic channel noise magnitude ($\propto 1/\sqrt{N} \propto 1/\sqrt{A}$)** while keeping the effective external driving strength per unit membrane area constant, calibrated at the reference area $A_{\mathrm{ref}} = 700\ \mu\mathrm{m}^2$. Scaling the effective external current with $A / A_{\mathrm{ref}}$ in `neuron.py` immediately restores the exact physical behavior in Fig. 9 (all curves converging to $\mathrm{CV} \approx 0.28$ at $A = 10\ \mu\mathrm{m}^2$ and saturating into flat plateaus for $A \ge 500\ \mu\mathrm{m}^2$) as well as the monotonic rise of reliability $R(A)$ in Fig. 12.

---

### 5. Quick Start & Usage

#### Requirements
* Python $\ge 3.10$ (Anaconda recommended)
* `numpy` ($\le 2.3$, compatible with `numba`), `scipy`, `numba`, `matplotlib`

#### Step 1: Run Full-Package Verification
Verify JIT compilation speed and benchmark values for spontaneous firing rate, $\mathrm{CV}$, and reliability $R$:
```bash
python main_verify.py
```

#### Step 2: Reproduce Paper Figures
Run any of the three experiment scripts. Parameter-sweep data are automatically cached in `results/data/*.npz` (set `force_rerun=True` in `run_all()` to recompute from scratch), and 300 DPI figures are saved to `results/figures/`:
```bash
# Section 3 (Part 1): Spontaneous spiking & channel currents (Figs. 3, 4, 5)
python experiments/exp01_spontaneous.py

# Section 3 (Part 2): Coherence resonance & STA (Figs. 6, 7, 8, 9)
python experiments/exp02_coherence_sta.py

# Section 4: Spike-time reliability (Figs. 10, 11, 12)
python experiments/exp03_reliability.py
```

---
---

<a name="中文说明"></a>
## 中文说明

### 1. 项目简介
本项目基于 Python（结合 `Numba` JIT 即时编译加速）完整复现了以下计算神经科学论文中的全部模拟实验：

> **Xinmeng Guo, Jiang Wang, Jing Liu, Haitao Yu, Roberto F. Galán, Yibin Cao, and Bin Deng\***,  
> *"Optimal time scales of input fluctuations for spiking coherence and reliability in stochastic Hodgkin–Huxley neurons"*（随机 Hodgkin–Huxley 神经元中使脉冲相干性和可靠性达到最优的输入波动时间尺度）,  
> **Physica A**, vol. 468, pp. 381–390, 2017.

本项目完成了论文中**全部数据图表（Fig. 3 至 Fig. 12）**的一比一复现：
* **无外部输入下的自发脉冲动力学（Section 3, Figs. 3–5）**：不同膜面积 $A$ 下的自发膜电位时间序列、对数坐标下自发发放频率随 $A$ 的递减关系，以及阈下/阈上阶段开放离子通道百分比（$p_{\mathrm{Na}}, p_{\mathrm{K}}$）与跨膜离子电流（$I_{\mathrm{Na}}, I_{\mathrm{K}}$）的同步演化。
* **非周期波动输入下的相干共振与脉冲触发平均（Section 3, Figs. 6–9）**：不同输入相关时间 $\tau$ 下的电压响应、脉冲间隔统计量（$\text{mean ISI}$、标准差 $\sigma_{\mathrm{ISI}}$、变异系数 $\mathrm{CV}$）的相干共振曲线、脉冲触发平均（$\mathrm{STA}$）曲线 $K(t)$，以及不同 $\tau$ 和输入幅值 $\sigma$ 下 $\mathrm{CV}$ 随膜面积 $A \in [10, 1500]\ \mu\mathrm{m}^2$ 的变化规律。
* **脉冲时间可靠性（Section 4, Figs. 10–12）**：脉冲时间可靠性指标 $R$ 随相关时间 $\tau$ 变化的钟形曲线、同一冻结噪声刺激下多次重复实验的膜电位脉冲对齐图，以及不同 $\tau$ 和 $\sigma$ 下可靠性 $R$ 随膜面积 $A$ 的单调递增曲线。

---

### 2. 模块化架构设计
为了便于在后续课题（如耦合随机 HH 神经网络、突触电导驱动模型、通道阻断效应等）中直接复用底层算子，本项目采用**“核心可复用算法包（`stochastic_hh/`）+ 论文复现实验脚本（`experiments/`）”**的双层解耦架构：

| 模块 / 文件路径 | 功能定位与核心内容 |
| :--- | :--- |
| `stochastic_hh/config.py` | 集中管理生物物理常数、反转电位、通道面密度、量纲换算乘子及默认统计参数。 |
| `stochastic_hh/channels_ssa.py` | 实现 5 状态 $\mathrm{K}^+$ 通道链（Fig. 1）与 8 状态 $\mathrm{Na}^+$ 通道网格（Fig. 2）的马尔可夫门控速率与随机屏蔽近似（SSA）单步更新算子。 |
| `stochastic_hh/neuron.py` | 解耦单步膜电位更新函数 `step_membrane_voltage` 与完整时间序列积分内核 `simulate_stochastic_hh`，支持后续直接在多神经元网络循环中调用。 |
| `stochastic_hh/stimulus.py` | 实现冻结高斯白噪声与 Alpha 函数 $(t/\tau)e^{-t/\tau}$ 的快速卷积、抗混叠高频滚降与目标幅值 $\sigma$ 缩放。 |
| `stochastic_hh/metrics.py` | 实现去瞬态与不应期防抖动的正向过零脉冲提取、ISI/CV 统计、STA 计算、宽度 $2d = 10\ \mathrm{ms}$ 方形窗卷积及成对相关性可靠性指标 $R$。 |
| `experiments/exp01_spontaneous.py` | 复现论文 Section 3 无外部输入自发发放实验（自动生成并保存 **Fig. 3, Fig. 4, Fig. 5**）。 |
| `experiments/exp02_coherence_sta.py` | 复现论文 Section 3 相干共振与 STA 实验（自动生成并保存 **Fig. 6, Fig. 7, Fig. 8, Fig. 9**）。 |
| `experiments/exp03_reliability.py` | 复现论文 Section 4 脉冲时间可靠性实验（自动生成并保存 **Fig. 10, Fig. 11, Fig. 12**）。 |
| `main_verify.py` | 全链路快速自检入口，数秒内验证自发频率、相干共振 $\mathrm{CV}$ 与可靠性 $R$ 的核心基准数值。 |

---

### 3. 核心模型与数值计算方法
1. **随机屏蔽近似（Stochastic-Shielding Approximation, SSA）**：
   全通道马尔可夫链模拟计算量极大。SSA 利用“只有直接连接导通态与关闭态的跃迁边才会直接引起膜电导涨落”的物理特性，仅在 $\mathrm{K}^+$ 通道的 $S_4 \leftrightarrow S_5$ 边以及 $\mathrm{Na}^+$ 通道的 $S_7 \leftrightarrow S_8$、$S_4 \leftrightarrow S_8$ 边上保留随机涨落（采用泊松–高斯混合抽样以兼顾极小膜面积下的精确非负性与大面积下的高速度），其余内部关闭态之间的跃迁采用确定性漂移更新。
2. **半隐式欧拉积分与量纲闭环**：
   时间步长固定为 $dt = 0.01\ \mathrm{ms}$，在 `config.py` 与 `neuron.py` 中将电压（$\mathrm{mV}$）、时间（$\mathrm{ms}$）、电容（$\mathrm{pF}$）、电导（$\mathrm{pS}$）与电流（$\mathrm{pA}$）严格对齐，并采用半隐式欧拉格式防止 $A = 1\ \mu\mathrm{m}^2$ 极小电容下动作电位峰值越过钠离子反转电位（$E_{\mathrm{Na}} = 45\ \mathrm{mV}$）。

---

### 4. 复现过程中的关键发现：`exp02`（Fig. 9）中膜面积与外接电流的量纲耦合问题
在复现 `exp02_coherence_sta.py` 的 **Fig. 9**（变异系数 $\mathrm{CV}$ 随膜面积 $A \in [10, 1500]\ \mu\mathrm{m}^2$ 的变化）时，我们发现并解决了一个极易导致复现曲线完全反转的关键物理细节：

* **问题现象与病因**：在膜电位方程中，总膜电容 $C_{\mathrm{tot}} = C_{\mathrm{unit}} A$ 以及三类通道总电导 $G_{\mathrm{Na}}, G_{\mathrm{K}}, G_L$ 均与膜面积 $A$ 成正比。如果在扫描 $A \in [10, 1500]\ \mu\mathrm{m}^2$ 时直接注入固定绝对幅值（如 $\sigma = 15\ \mathrm{pA}$）的外接电流 $I_0(t)$，将方程两边同除以 $A$ 后可以发现，**单位面积细胞膜承受的有效外接刺激强度实际上变成了 $I_0(t) / A$**（随 $A$ 变化了 $150$ 倍）：
  * 在小膜面积端（$A = 10\ \mu\mathrm{m}^2$），固定 $15\ \mathrm{pA}$ 相当于标准面积（$A = 700\ \mu\mathrm{m}^2$）下 $70$ 倍的超高电流密度，直接将神经元推入剧烈的强迫簇状放电与去极化阻滞交替态，导致 $\mathrm{CV}$ 反向飙升至 $0.5 \sim 0.65$（形成错误的“V 型谷”）；
  * 在大膜面积端（$A = 1500\ \mu\mathrm{m}^2$），固定电流被大电容大幅稀释，无法稳定驱动膜电位跨过阈值，导致大量漏发（skipping），使 $\mathrm{CV}$ 随 $A$ 线性发散，完全无法出现原图中的水平饱和平台。
* **物理修正**：论文改变膜面积 $A$ 的目的，是将其作为纯粹调控**内在通道噪声相对强度（$\propto 1/\sqrt{A}$）**的控制变量，而外接刺激对单位膜电容的有效驱动强度应保持恒定（以论文主实验面积 $A_{\mathrm{ref}} = 700\ \mu\mathrm{m}^2$ 为标定基准）。在 `neuron.py` 中引入基于参考面积 $A_{\mathrm{ref}} = 700\ \mu\mathrm{m}^2$ 的有效电流耦合缩放后，Fig. 9 中“小面积汇聚于 $\mathrm{CV} \approx 0.28$、大面积进入水平饱和平台”以及 Fig. 12 中“可靠性 $R$ 随膜面积单调递增”的规律得到了完美复现。

---

### 5. 快速运行指南

1. **运行全模块基准验证**：
   在 PyCharm 中直接运行根目录下的 `main_verify.py`，或执行命令行：
   ```bash
   python main_verify.py
   ```
2. **运行论文图表复现脚本**：
   依次运行 `experiments/` 目录下的三个脚本，所有耗时的参数扫描结果会自动缓存至 `results/data/*.npz`（如需强制重新跑模拟，可将脚本末尾 `run_all(force_rerun=False)` 改为 `True`），所有 300 DPI 高清复现图自动保存至 `results/figures/`：
   ```bash
   python experiments/exp01_spontaneous.py    # 生成 Fig. 3, Fig. 4, Fig. 5
   python experiments/exp02_coherence_sta.py  # 生成 Fig. 6, Fig. 7, Fig. 8, Fig. 9
   python experiments/exp03_reliability.py    # 生成 Fig. 10, Fig. 11, Fig. 12
   ```
