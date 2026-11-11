# 🔬 Computational Environment & Reproducibility Specification
> *This document provides the full provenance, hardware specifications, solver toolchains, and algorithmic hyperparameters for reproducing the computational experiments presented in this benchmark.*

---

## 📌 1. Summary for Papers and Thesis (Copy & Paste Text)
```markdown
All computational experiments were conducted on a machine equipped with an Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz processor (20 logical threads), 156.86 GB of RAM, running Linux (x86_64-linux-gnu). The algorithms (Two-Phase and UNIF) were implemented in the Julia programming language (v1.12.6). The subproblems were solved using Gurobi Optimizer (v13.0.2) and CUTEst.jl benchmark interface. The optimality tolerance was set to $\epsilon_g = 0.001$, feasibility tolerance to $\epsilon_f = 0.05$, and minimum step length to $\epsilon_s = 0.0001$, with a maximum of 5000 iterations. Source code corresponds to Git commit `0b0cc69` (main branch).
```

---

## 🛠️ 2. Hardware and Operating System Specifications
| Property | Value |
| :--- | :--- |
| **CPU Model** | Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz |
| **Architecture** | Linux (x86_64-linux-gnu) |
| **Available Threads** | 20 |
| **Installed RAM** | 156.86 GB |
| **Host System** | optimizationserver |

---

## 📦 3. Software Stack and Solvers
| Tool / Library | Version / Details |
| :--- | :--- |
| **Julia Language** | v1.12.6 (commit `0b0cc69`) |
| **Linear Solver** | Gurobi Optimizer v13.0.2 |
| **BLAS Config** | LBTConfig([ILP64] libopenblas64_.so, [LP64] libopenblas.so) (1 threads) |
| **Benchmark Framework** | CUTEst.jl |

---

## 📜 4. Git Provenance
| Item | Value |
| :--- | :--- |
| **Commit (Short)** | `0b0cc69` |
| **Commit (Full)** | `0b0cc69bde3ccfd9b7c618a7a1eec3d53aa6afe3` |
| **Branch** | `main` |
| **Tree Status** | ⚠️ **DIRTY** (uncommitted changes) |
| **Commit Date** | 2026-11-10 23:16:57 -0300 |
| **Commit Message** | fix(runner): remove Pkg.precompile from shell runner to prevent lld linker crash |
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
- **Execution Date:** 2026-11-10T23:54:32
- **Elapsed Wall-Clock Time:** 5856.58 seconds
- **Parallel Workers:** 16
- **CLI Invocated:** `/home/gabrieldelgado/Generic_2PvsSLP/src/Generic_main.jl --workers 0 --max-iter 500 --timeout 60.0 --qp-solver gurobi --mode filter --max-var 200 --max-con 100 --tp-sar all --tp-ph all --tp-atr all --tp-atr-strat all --tp-rru all --tp-sqp all --unif-sar all --unif-ph all --unif-atr all --unif-atr-strat all --unif-sqp all`
