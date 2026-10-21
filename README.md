[project_readme.md](https://github.com/user-attachments/files/33765802/project_readme.md)
# Optimal Time Scales for Spiking Coherence and Reliability in Stochastic Hodgkin–Huxley Neurons (Reproduction & Modular Toolkit)

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

### 2. Quick Start & Usage

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

### 3. Modular Architecture & Directory Structure
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
└── main_verify.py                  # [Quick Verification] End-to-end benchmark of all core modules
```

---

### 4. Model & Numerical Methods Summary
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
