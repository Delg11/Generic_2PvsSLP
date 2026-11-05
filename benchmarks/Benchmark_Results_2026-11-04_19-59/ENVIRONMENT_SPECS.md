# 🔬 Computational Environment & Reproducibility Specification
> *This document provides the full provenance, hardware specifications, solver toolchains, and algorithmic hyperparameters for reproducing the computational experiments presented in this benchmark.*

---

## 📌 1. Summary for Papers and Thesis (Copy & Paste Text)
```markdown
All computational experiments were conducted on a machine equipped with an znver3 processor (32 logical threads), 124.94 GB of RAM, running Linux (x86_64-linux-gnu). The algorithms (Two-Phase and UNIF) were implemented in the Julia programming language (v1.10.11). The subproblems were solved using Gurobi Optimizer (v13.0.2) and CUTEst.jl benchmark interface. The optimality tolerance was set to $\epsilon_g = 0.001$, feasibility tolerance to $\epsilon_f = 0.05$, and minimum step length to $\epsilon_s = 0.0001$, with a maximum of 5000 iterations. Source code corresponds to Git commit `9019f42` (main branch).
```

---

## 🛠️ 2. Hardware and Operating System Specifications
| Property | Value |
| :--- | :--- |
| **CPU Model** | znver3 |
| **Architecture** | Linux (x86_64-linux-gnu) |
| **Available Threads** | 32 |
| **Installed RAM** | 124.94 GB |
| **Host System** | dantzig |

---

## 📦 3. Software Stack and Solvers
| Tool / Library | Version / Details |
| :--- | :--- |
| **Julia Language** | v1.10.11 (commit `9019f42`) |
| **Linear Solver** | Gurobi Optimizer v13.0.2 |
| **BLAS Config** | LBTConfig([ILP64] libopenblas64_.so, [LP64] libopenblas.so) (1 threads) |
| **Benchmark Framework** | CUTEst.jl |

---

## 📜 4. Git Provenance
| Item | Value |
| :--- | :--- |
| **Commit (Short)** | `9019f42` |
| **Commit (Full)** | `9019f42ec8c2cd6a323dd197719db5a0b2250c3c` |
| **Branch** | `main` |
| **Tree Status** | ⚠️ **DIRTY** (uncommitted changes) |
| **Commit Date** | 2026-11-04 19:37:34 -0300 |
| **Commit Message** | 04112026 |
| **Repository URL** | https://github.com/Delg11/Generic_2PvsSLP |

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
- **Execution Date:** 2026-11-04T20:03:31
- **Elapsed Wall-Clock Time:** 2798.55 seconds
- **Parallel Workers:** 16
- **CLI Invocated:** `src/Generic_main.jl -m filter -w 0 -v 200 -c 100 --tp-sqp all --unif-sqp all --tp-atr-strat all --unif-atr-strat all -i 500 -t 60.0`
