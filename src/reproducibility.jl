# ==============================================================================
# REPRODUCIBILITY AND ENVIRONMENT METADATA MODULE
# ==============================================================================
module Reproducibility

using Dates
using DataFrames
using CSV
using LinearAlgebra
using Gurobi

export collect_environment_metadata, save_reproducibility_artifacts, get_git_short_commit

function get_git_info()
    commit = try strip(read(`git rev-parse HEAD`, String)) catch; "unknown" end
    short_commit = try strip(read(`git rev-parse --short HEAD`, String)) catch; "unknown" end
    branch = try strip(read(`git rev-parse --abbrev-ref HEAD`, String)) catch; "unknown" end
    dirty = try !isempty(strip(read(`git status --porcelain`, String))) catch; false end
    date = try strip(read(`git log -1 --format=%cd --date=iso`, String)) catch; "unknown" end
    msg = try strip(read(`git log -1 --format=%s`, String)) catch; "unknown" end
    remote = try strip(read(`git config --get remote.origin.url`, String)) catch; "unknown" end
    return (
        commit = commit,
        short_commit = short_commit,
        branch = branch,
        dirty = dirty,
        date = date,
        message = msg,
        remote = remote
    )
end

function get_git_short_commit()
    return try strip(read(`git rev-parse --short HEAD`, String)) catch; "unknown" end
end

function get_cpu_model()
    if Sys.iswindows()
        try
            return strip(read(`powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"`, String))
        catch
            return string(Sys.CPU_NAME)
        end
    elseif Sys.islinux()
        try
            model = strip(read(`bash -c "lscpu | grep 'Model name:' | cut -d: -f2"`, String))
            return isempty(model) ? string(Sys.CPU_NAME) : model
        catch
            return string(Sys.CPU_NAME)
        end
    else
        return string(Sys.CPU_NAME)
    end
end

function get_gurobi_version()
    try
        maj = Ref{Cint}(0)
        min_ = Ref{Cint}(0)
        tech = Ref{Cint}(0)
        Gurobi.GRBversion(maj, min_, tech)
        return "$(maj[]).$(min_[]).$(tech[])"
    catch
        return "N/A"
    end
end

function collect_environment_metadata(; 
    mode::String="unknown", 
    workers_count::Int=1, 
    args_dict::Dict=Dict(),
    start_time::DateTime=now()
)
    git = get_git_info()
    cpu_model = get_cpu_model()
    ram_gb = round(Sys.total_memory() / (1024^3), digits=2)
    gurobi_ver = get_gurobi_version()
    julia_ver = string(VERSION)
    julia_commit = string(Base.GIT_VERSION_INFO.commit)
    os_info = "$(Sys.iswindows() ? "Windows" : Sys.KERNEL) ($(Sys.MACHINE))"
    hostname = gethostname()

    return Dict{String, Any}(
        "git" => Dict{String, Any}(
            "commit_full" => git.commit,
            "commit_short" => git.short_commit,
            "branch" => git.branch,
            "is_dirty" => git.dirty,
            "commit_date" => git.date,
            "commit_message" => git.message,
            "remote_url" => git.remote
        ),
        "hardware" => Dict{String, Any}(
            "cpu_model" => cpu_model,
            "cpu_threads" => Sys.CPU_THREADS,
            "ram_gb" => ram_gb,
            "hostname" => hostname,
            "os" => os_info
        ),
        "software" => Dict{String, Any}(
            "julia_version" => julia_ver,
            "julia_commit" => julia_commit,
            "gurobi_version" => gurobi_ver,
            "blas_vendor" => string(LinearAlgebra.BLAS.get_config()),
            "blas_threads" => LinearAlgebra.BLAS.get_num_threads()
        ),
        "hyperparameters" => Dict{String, Any}(
            "tolG" => 1e-3,
            "tolF" => 5e-2,
            "tolS" => 1e-4,
            "delta0" => 0.1,
            "max_iter" => 5000,
            "maxcount" => 3,
            "rfeas" => 1e-12,
            "delta_min" => 1e-12,
            "delta_max" => 1e16,
            "r_resto" => 0.9,
            "theta0" => 0.90,
            "tau1" => 0.1,
            "tau2" => 0.25,
            "non_monotone_M" => 1
        ),
        "execution" => Dict{String, Any}(
            "mode" => mode,
            "workers_count" => workers_count,
            "start_time" => Dates.format(start_time, "yyyy-mm-ddTHH:MM:SS"),
            "end_time" => "",
            "elapsed_seconds" => 0.0,
            "command_line" => PROGRAM_FILE * " " * join(ARGS, " ")
        )
    )
end

function metadata_to_dataframe(meta::Dict{String, Any})
    df = DataFrame(Category=String[], Parameter=String[], Value=String[])
    
    # Git
    git = meta["git"]
    push!(df, ("Git Provenance", "Commit SHA (Short)", string(git["commit_short"])))
    push!(df, ("Git Provenance", "Commit SHA (Full)", string(git["commit_full"])))
    push!(df, ("Git Provenance", "Active Branch", string(git["branch"])))
    push!(df, ("Git Provenance", "Working Tree Status", git["is_dirty"] ? "DIRTY (Uncommitted modifications present)" : "CLEAN"))
    push!(df, ("Git Provenance", "Commit Date", string(git["commit_date"])))
    push!(df, ("Git Provenance", "Commit Message", string(git["commit_message"])))
    push!(df, ("Git Provenance", "Remote URL", string(git["remote_url"])))

    # Hardware
    hw = meta["hardware"]
    push!(df, ("Hardware Specs", "CPU Model", string(hw["cpu_model"])))
    push!(df, ("Hardware Specs", "Logical Threads / Cores", string(hw["cpu_threads"])))
    push!(df, ("Hardware Specs", "Installed RAM (GB)", string(hw["ram_gb"])))
    push!(df, ("Hardware Specs", "Host Machine", string(hw["hostname"])))
    push!(df, ("Hardware Specs", "Operating System", string(hw["os"])))

    # Software
    sw = meta["software"]
    push!(df, ("Software Stack", "Julia Version", string(sw["julia_version"])))
    push!(df, ("Software Stack", "Julia Commit", string(sw["julia_commit"])))
    push!(df, ("Software Stack", "LP Solver (Gurobi)", string(sw["gurobi_version"])))
    push!(df, ("Software Stack", "BLAS Config", string(sw["blas_vendor"])))
    push!(df, ("Software Stack", "BLAS Threads", string(sw["blas_threads"])))

    # Hyperparameters
    hp = meta["hyperparameters"]
    push!(df, ("Algorithmic Tolerances", "Optimality Tol (tolG)", string(hp["tolG"])))
    push!(df, ("Algorithmic Tolerances", "Feasibility Tol (tolF)", string(hp["tolF"])))
    push!(df, ("Algorithmic Tolerances", "Min Step Length (tolS)", string(hp["tolS"])))
    push!(df, ("Algorithmic Tolerances", "Initial TR Radius (delta0)", string(hp["delta0"])))
    push!(df, ("Algorithmic Tolerances", "Max Iterations", string(hp["max_iter"])))
    push!(df, ("Algorithmic Tolerances", "Consecutive Small Steps (maxcount)", string(hp["maxcount"])))
    push!(df, ("Algorithmic Tolerances", "Restoration Target (theta0)", string(hp["theta0"])))
    push!(df, ("Algorithmic Tolerances", "Restoration Step Ratio (r_resto)", string(hp["r_resto"])))
    push!(df, ("Algorithmic Tolerances", "Non-monotone Memory (M)", string(hp["non_monotone_M"])))

    # Execution
    ex = meta["execution"]
    push!(df, ("Execution Run", "Benchmark Mode", string(ex["mode"])))
    push!(df, ("Execution Run", "Distributed Workers", string(ex["workers_count"])))
    push!(df, ("Execution Run", "Start Time", string(ex["start_time"])))
    push!(df, ("Execution Run", "End Time", string(ex["end_time"])))
    push!(df, ("Execution Run", "Total Wall Time (s)", string(round(ex["elapsed_seconds"], digits=2))))
    push!(df, ("Execution Run", "CLI Command", string(ex["command_line"])))

    return df
end

function json_escape(s::AbstractString)
    return replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
end

function to_json_string(val::Any, indent_level::Int=0)
    indent = "  "^indent_level
    inner_indent = "  "^(indent_level + 1)
    if isa(val, AbstractDict)
        lines = ["{"]
        keys_list = collect(keys(val))
        for (i, k) in enumerate(keys_list)
            v = val[k]
            comma = i < length(keys_list) ? "," : ""
            push!(lines, "$(inner_indent)\"$(json_escape(string(k)))\": $(to_json_string(v, indent_level + 1))$(comma)")
        end
        push!(lines, "$(indent)}")
        return join(lines, "\n")
    elseif isa(val, AbstractVector)
        lines = ["["]
        for (i, item) in enumerate(val)
            comma = i < length(val) ? "," : ""
            push!(lines, "$(inner_indent)$(to_json_string(item, indent_level + 1))$(comma)")
        end
        push!(lines, "$(indent)]")
        return join(lines, "\n")
    elseif isa(val, AbstractString)
        return "\"$(json_escape(val))\""
    elseif isa(val, Bool)
        return val ? "true" : "false"
    elseif isa(val, Number)
        return string(val)
    else
        return "\"$(json_escape(string(val)))\""
    end
end

function generate_markdown_specs(meta::Dict{String, Any})
    git = meta["git"]
    hw = meta["hardware"]
    sw = meta["software"]
    hp = meta["hyperparameters"]
    ex = meta["execution"]

    dirty_badge = git["is_dirty"] ? "⚠️ **DIRTY** (uncommitted changes)" : "✅ **CLEAN**"

    md = """
# 🔬 Computational Environment & Reproducibility Specification
> *This document provides the full provenance, hardware specifications, solver toolchains, and algorithmic hyperparameters for reproducing the computational experiments presented in this benchmark.*

---

## 📌 1. Summary for Papers and Thesis (Copy & Paste Text)
```markdown
All computational experiments were conducted on a machine equipped with an $(hw["cpu_model"]) processor ($(hw["cpu_threads"]) logical threads), $(hw["ram_gb"]) GB of RAM, running $(hw["os"]). The algorithms (Two-Phase and UNIF) were implemented in the Julia programming language (v$(sw["julia_version"])). The subproblems were solved using Gurobi Optimizer (v$(sw["gurobi_version"])) and CUTEst.jl benchmark interface. The optimality tolerance was set to \$\\epsilon_g = $(hp["tolG"])\$, feasibility tolerance to \$\\epsilon_f = $(hp["tolF"])\$, and minimum step length to \$\\epsilon_s = $(hp["tolS"])\$, with a maximum of $(hp["max_iter"]) iterations. Source code corresponds to Git commit `$(git["commit_short"])` ($(git["branch"]) branch).
```

---

## 🛠️ 2. Hardware and Operating System Specifications
| Property | Value |
| :--- | :--- |
| **CPU Model** | $(hw["cpu_model"]) |
| **Architecture** | $(hw["os"]) |
| **Available Threads** | $(hw["cpu_threads"]) |
| **Installed RAM** | $(hw["ram_gb"]) GB |
| **Host System** | $(hw["hostname"]) |

---

## 📦 3. Software Stack and Solvers
| Tool / Library | Version / Details |
| :--- | :--- |
| **Julia Language** | v$(sw["julia_version"]) (commit `$(git["commit_short"])`) |
| **Linear Solver** | Gurobi Optimizer v$(sw["gurobi_version"]) |
| **BLAS Config** | $(sw["blas_vendor"]) ($(sw["blas_threads"]) threads) |
| **Benchmark Framework** | CUTEst.jl |

---

## 📜 4. Git Provenance
| Item | Value |
| :--- | :--- |
| **Commit (Short)** | `$(git["commit_short"])` |
| **Commit (Full)** | `$(git["commit_full"])` |
| **Branch** | `$(git["branch"])` |
| **Tree Status** | $dirty_badge |
| **Commit Date** | $(git["commit_date"]) |
| **Commit Message** | $(git["commit_message"]) |
| **Repository URL** | $(git["remote_url"]) |

---

## ⚙️ 5. Algorithmic Tolerances & Hyperparameters
| Parameter | Symbol | Value | Description |
| :--- | :---: | :---: | :--- |
| **Optimality Tolerance** | \$\\epsilon_g\$ | `$(hp["tolG"])` | Projected gradient norm stopping criterion |
| **Feasibility Tolerance** | \$\\epsilon_f\$ | `$(hp["tolF"])` | Constraint violation norm stopping criterion |
| **Step Length Tolerance** | \$\\epsilon_s\$ | `$(hp["tolS"])` | Step size stagnation threshold |
| **Initial Trust Region** | \$\\delta_0\$ | `$(hp["delta0"])` | Initial trust-region radius |
| **Max Iterations** | \$k_{\\max}\$ | `$(hp["max_iter"])` | Outer iteration cutoff |
| **Consecutive Small Steps** | - | `$(hp["maxcount"])` | Consecutive stall trigger limit |
| **Restoration Target** | \$\\theta_0\$ | `$(hp["theta0"])` | Infeasibility reduction coefficient |
| **Restoration Step Ratio** | \$r_{\\text{resto}}\$ | `$(hp["r_resto"])` | Feasibility decrease proportion |
| **Non-monotone Memory** | \$M\$ | `$(hp["non_monotone_M"])` | History length for merit function |

---

## 🚀 6. Execution Details
- **Execution Date:** $(ex["start_time"])
- **Elapsed Wall-Clock Time:** $(round(ex["elapsed_seconds"], digits=2)) seconds
- **Parallel Workers:** $(ex["workers_count"])
- **CLI Invocated:** `$(ex["command_line"])`
"""
    return md
end

function save_reproducibility_artifacts(results_dir::String, meta::Dict{String, Any})
    mkpath(results_dir)

    # 1. JSON manifest
    json_path = joinpath(results_dir, "benchmark_manifest.json")
    open(json_path, "w") do f
        write(f, to_json_string(meta))
    end

    # 2. Markdown specs
    md_path = joinpath(results_dir, "ENVIRONMENT_SPECS.md")
    open(md_path, "w") do f
        write(f, generate_markdown_specs(meta))
    end

    # 3. Metadata DataFrame
    df_meta = metadata_to_dataframe(meta)
    CSV.write(joinpath(results_dir, "environment_metadata.csv"), df_meta)

    return df_meta
end

end # module
