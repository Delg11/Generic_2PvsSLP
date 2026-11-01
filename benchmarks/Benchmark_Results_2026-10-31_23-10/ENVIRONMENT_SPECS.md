# 🔬 Computational Environment & Reproducibility Specification
> *This document provides the full provenance, hardware specifications, solver toolchains, and algorithmic hyperparameters for reproducing the computational experiments presented in this benchmark.*

---

## 📌 1. Summary for Papers and Thesis (Copy & Paste Text)
```markdown
All computational experiments were conducted on a machine equipped with an AMD Ryzen 7 8700G w/ Radeon 780M Graphics processor (16 logical threads), 31.15 GB of RAM, running Windows (x86_64-w64-mingw32). The algorithms (Two-Phase and UNIF) were implemented in the Julia programming language (v1.12.5). The subproblems were solved using Gurobi Optimizer (v13.0.2) and CUTEst.jl benchmark interface. The optimality tolerance was set to $\epsilon_g = 0.001$, feasibility tolerance to $\epsilon_f = 0.05$, and minimum step length to $\epsilon_s = 0.0001$, with a maximum of 5000 iterations. Source code corresponds to Git commit `27f4f57` (main branch).
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
| **Julia Language** | v1.12.5 (commit `27f4f57`) |
| **Linear Solver** | Gurobi Optimizer v13.0.2 |
| **BLAS Config** | LBTConfig([ILP64] libopenblas64_.dll, [LP64] libopenblas.dll) (8 threads) |
| **Benchmark Framework** | CUTEst.jl |

---

## 📜 4. Git Provenance
| Item | Value |
| :--- | :--- |
| **Commit (Short)** | `27f4f57` |
| **Commit (Full)** | `27f4f57c8db884393009f82d6fb96c91065dc701` |
| **Branch** | `main` |
| **Tree Status** | ⚠️ **DIRTY** (uncommitted changes) |
| **Commit Date** | 2026-10-31 21:42:46 -0300 |
| **Commit Message** | 31102026 |
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
- **Execution Date:** 2026-10-31T23:14:11
- **Elapsed Wall-Clock Time:** 992.7 seconds
- **Parallel Workers:** 14 (distributed pmap execution across 16 CPU threads)
- **CLI Invocated:** `src/Generic_main.jl -m filter -v 10 -w 0 --tp-atr-strat all --unif-atr-strat all`
