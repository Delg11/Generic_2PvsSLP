# 🔬 Computational Environment & Reproducibility Specification
> *This document provides the full provenance, hardware specifications, solver toolchains, and algorithmic hyperparameters for reproducing the computational experiments presented in this benchmark.*

---

## 📌 1. Summary for Papers and Thesis (Copy & Paste Text)
```markdown
All computational experiments were conducted on a machine equipped with an AMD Ryzen 7 8700G w/ Radeon 780M Graphics processor (16 logical threads), 31.15 GB of RAM, running Windows (x86_64-w64-mingw32). The algorithms (Two-Phase and UNIF) were implemented in the Julia programming language (v1.12.5). The subproblems were solved using Gurobi Optimizer (v13.0.2) and CUTEst.jl benchmark interface. The optimality tolerance was set to $\epsilon_g = 0.001$, feasibility tolerance to $\epsilon_f = 0.05$, and minimum step length to $\epsilon_s = 0.0001$, with a maximum of 5000 iterations. Source code corresponds to Git commit `07d7bd7` (main branch).
```

---

## 🛠️ 2. Hardware and Operating System Specifications
| Property | Value |
| :--- | :--- |
| **CPU Model** | AMD Ryzen 7 8700G w/ Radeon 780M Graphics |
| **Architecture** | Windows (x86_64-w64-mingw32) |
| **Available Threads** | 16 |
| **Installed RAM** | 31.15 GB |
| **Host System** | DESKTOP-F77C9MN |

---

## 📦 3. Software Stack and Solvers
| Tool / Library | Version / Details |
| :--- | :--- |
| **Julia Language** | v1.12.5 (commit `07d7bd7`) |
| **Linear Solver** | Gurobi Optimizer v13.0.2 |
| **BLAS Config** | LBTConfig([ILP64] libopenblas64_.dll, [LP64] libopenblas.dll) (8 threads) |
| **Benchmark Framework** | CUTEst.jl |

---

## 📜 4. Git Provenance
| Item | Value |
| :--- | :--- |
| **Commit (Short)** | `07d7bd7` |
| **Commit (Full)** | `07d7bd70b17d8271099b2702f4401f5d2c6cf081` |
| **Branch** | `main` |
| **Tree Status** | ⚠️ **DIRTY** (uncommitted changes) |
| **Commit Date** | 2026-10-31 09:48:31 -0300 |
| **Commit Message** | Update Generic_module_Twophase.jl |
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
- **Execution Date:** 2026-10-31T19:00:59
- **Elapsed Wall-Clock Time:** 26.81 seconds
- **Parallel Workers:** 1
- **CLI Invocated:** `src/Generic_main.jl -m test -p HS6 -w 2`
