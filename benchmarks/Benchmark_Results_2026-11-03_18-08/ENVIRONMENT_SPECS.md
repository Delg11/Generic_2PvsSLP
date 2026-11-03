# 🔬 Computational Environment & Reproducibility Specification
> *This document provides the full provenance, hardware specifications, solver toolchains, and algorithmic hyperparameters for reproducing the computational experiments presented in this benchmark.*

---

## 📌 1. Summary for Papers and Thesis (Copy & Paste Text)
```markdown
All computational experiments were conducted on a machine equipped with an tigerlake processor (8 logical threads), 14.86 GB of RAM, running Linux (x86_64-linux-gnu). The algorithms (Two-Phase and UNIF) were implemented in the Julia programming language (v1.12.5). The subproblems were solved using Gurobi Optimizer (v13.0.2) and CUTEst.jl benchmark interface. The optimality tolerance was set to $\epsilon_g = 0.001$, feasibility tolerance to $\epsilon_f = 0.05$, and minimum step length to $\epsilon_s = 0.0001$, with a maximum of 5000 iterations. Source code corresponds to Git commit `bbb0e2a` (main branch).
```

---

## 🛠️ 2. Hardware and Operating System Specifications
| Property | Value |
| :--- | :--- |
| **CPU Model** | tigerlake |
| **Architecture** | Linux (x86_64-linux-gnu) |
| **Available Threads** | 8 |
| **Installed RAM** | 14.86 GB |
| **Host System** | gabriel-Inspiron-15-5510 |

---

## 📦 3. Software Stack and Solvers
| Tool / Library | Version / Details |
| :--- | :--- |
| **Julia Language** | v1.12.5 (commit `bbb0e2a`) |
| **Linear Solver** | Gurobi Optimizer v13.0.2 |
| **BLAS Config** | LBTConfig([ILP64] libopenblas64_.so, [LP64] libopenblas.so) (1 threads) |
| **Benchmark Framework** | CUTEst.jl |

---

## 📜 4. Git Provenance
| Item | Value |
| :--- | :--- |
| **Commit (Short)** | `bbb0e2a` |
| **Commit (Full)** | `bbb0e2accadf97063f944bd3e3cd23d29d86f42e` |
| **Branch** | `main` |
| **Tree Status** | ⚠️ **DIRTY** (uncommitted changes) |
| **Commit Date** | 2026-11-02 20:11:50 -0300 |
| **Commit Message** | 02112026 |
| **Repository URL** | https://github.com/Delg11/Generic_2PvsSLP.git |

---

## ⚙️ 5. Algorithmic Tolerances & Hyperparameters
| Parameter | Symbol | Value | Description |
| :--- | :---: | :---: | :--- |
| **Optimality Tolerance** | $\epsilon_g$ | `0.001` | Projected gradient norm stopping criterion |
| **Feasibility Tolerance** | $\epsilon_f$ | `0.05` | Constraint violation norm stopping criterion |
| **Step Length Tolerance** | $\epsilon_s$ | `0.0001` | Step size stagnation threshold |
| **Initial Trust Region** | $\delta_0$ | `0.1` | Initial trust-region radius |
| **Max Iterations** | $k_{\max}$ | `5000` | Outer iteration cutoff |
| **Consecutive Small Steps** | - | `3` | Consecutive stall trigger limit |
| **Restoration Target** | $\theta_0$ | `0.9` | Infeasibility reduction coefficient |
| **Restoration Step Ratio** | $r_{\text{resto}}$ | `0.9` | Feasibility decrease proportion |
| **Non-monotone Memory** | $M$ | `1` | History length for merit function |

---

## 🚀 6. Execution Details
- **Execution Date:** 2026-11-03T18:09:30
- **Elapsed Wall-Clock Time:** 112.4 seconds
- **Parallel Workers:** 2
- **CLI Invocated:** `src/Generic_main.jl -m test -p ALLINITC,BT8,BT12,CERI651DLS,BA-L1LS,ARGLINB -w 2 --tp-sqp all --unif-sqp all --tp-atr-strat all --unif-atr-strat all -i 500 -t 60.0`
