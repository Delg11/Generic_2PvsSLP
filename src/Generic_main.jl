# ==============================================================================
# Proprietary Software • All Rights Reserved
# ==============================================================================
# 1. IMPORTS AND INCLUDES
# ==============================================================================
using Pkg

# Activate the environment associated with the directory where this file is saved
Pkg.activate(@__DIR__) 
# Read Manifest.toml and install/precompile all necessary dependencies
Pkg.instantiate()

# Make sure ArgParse is added to your environment
# Pkg.add("ArgParse")
using ArgParse

using CUTEst
using LinearAlgebra
using Printf
using Statistics
using DataFrames
using XLSX
using Dates
using Plots
using CSV

# 1. Load types and export to Main
include("Generic_Sharedtypes.jl")
using .SharedTypes

# 2. Safely load the algorithm modules now that Main knows the types
include("Generic_module_Twophase.jl")
using .Generic_module_Twophase

include("Generic_module_slp.jl")   
using .Generic_module_slp

include("Generic_module_Stats.jl")
using .Generic_module_Stats

# ==============================================================================
# 2. COMMAND LINE ARGUMENTS CONFIGURATION
# ==============================================================================
function parse_commandline()
    s = ArgParseSettings(description="SLP and TwoPhase Optimization Benchmark - Remote Execution")

    @add_arg_table s begin
        "--mode", "-m"
            help = "Execution mode: 'test' or 'filter'."
            arg_type = String
            default = "test"
        "--problems", "-p"
            help = "Comma-separated list of problems."
            arg_type = String
            default = "HS6"
        "--max-var", "-v"
            help = "Maximum number of variables (-1 for no limit)."
            arg_type = Int
            default = -1
        "--max-con", "-c"
            help = "Maximum number of constraints (-1 for no limit)."
            arg_type = Int
            default = -1
        "--run-slp"
            help = "Run SLP variants (true/false)."
            arg_type = Bool
            default = true
        "--run-twophase"
            help = "Run Two-Phase variants (true/false)."
            arg_type = Bool
            default = true
        
        # Two-Phase Parameters
        "--tp-apr"
            help = "Adaptive ratio (Two-Phase). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--tp-bq"
            help = "Parabolic step (Two-Phase). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--tp-atr"
            help = "Anisotropic region (Two-Phase). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--tp-uru"
            help = "Ratio update (Two-Phase). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--tp-sqp"
            help = "SQP strategies for Two-Phase. Comma-separated (e.g., none,identity,spectral,exact) or 'all'."
            arg_type = String
            default = "all"
            
        # SLP Parameters
        "--slp-apr"
            help = "Adaptive ratio (SLP). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--slp-bq"
            help = "Parabolic step (SLP). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--slp-atr"
            help = "Anisotropic region (SLP). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--slp-sqp"
            help = "SQP strategies for SLP. Comma-separated (e.g., none,identity,spectral,exact) or 'all'."
            arg_type = String
            default = "all"
    end

    return parse_args(s)
end

function parse_sqp_args(arg_str::String)
    if lowercase(arg_str) == "all"
        return [:none, :identity, :spectral, :exact]
    else
        return Symbol.(strip.(split(arg_str, ",")))
    end
end

args = parse_commandline()

# ==============================================================================
# 4. GENERAL CONFIGURATION AND ALGORITHM SELECTION
# ==============================================================================
println("\n" * "=" ^ 80)
println("⚙️  CONFIGURING VARIANTS AND EXPORT ENVIRONMENT")
println("=" ^ 80)

println("🖥️  System: $(Sys.CPU_THREADS) threads available")
println("🚀 Julia: Using $(Threads.nthreads()) threads")
println("-" ^ 80)

const RUN_TWOPHASE = args["run-twophase"]
const RUN_SLP      = args["run-slp"]

common_max_iter = 500
common_delta0   = 0.1
common_tolG     = 1e-3
common_tolF     = 5e-2
common_tolS     = 1e-4
common_maxcount = 3
common_verbose  = false 

current_datetime = Dates.format(now(), "yyyy-mm-dd_HH-MM")
results_dir = "Benchmark_Results_$(current_datetime)"
mkpath(results_dir)
println("📁 Results directory created: $results_dir")

# --- BUILDER FUNCTIONS ---
function build_twophase_params(apr, M, bq, atr, uru, use_quad, b_strat)
    return SharedTypes.TwoPhaseParams(
        max_outer_iter = common_max_iter, δ0_opt = common_delta0, δ0_resto = common_delta0,
        tolG = common_tolG, tolF = common_tolF, tolS = common_tolS, maxcount = common_maxcount,
        verbose_out = common_verbose, use_slp_stopping = true, rfeas = 1e-12, δmin = 1e-12, 
        δmax = 1e16, r_resto = 0.9, τ1 = 0.1, τ2 = 0.25, αL = 1e-8, αR = 1e-8,
        θ0 = 0.90, αΦ = 1e-8, max_iter_resto = 500, verbose_in = common_verbose, debugverbose = false,
        aredpred_ratio = apr, non_monotone_M = M, backtracking_quadratic = bq, anisotropic_trust_region = atr, use_ratio_update = uru,
        use_quadratic = use_quad, B_update_strategy = b_strat, quadratic_solver = :ripqp
    )
end

function build_slp_params(apr, bq, atr, use_quad, b_strat)
    return SharedTypes.SLPParams(
        delta0 = common_delta0, tolG = common_tolG, tolF = common_tolF, tolS = common_tolS,
        maxiter = common_max_iter, maxcount = common_maxcount, verbose = common_verbose,
        verbose_out = common_verbose, debugverbose = false, backtracking_quadratic = bq, anisotropic_trust_region = atr,
        aredpred_ratio = apr, use_quadratic = use_quad, B_update_strategy = b_strat, quadratic_solver = :ripqp
    )
end

# --- TWO-PHASE PROCESSING ---
twophase_variants = []
if RUN_TWOPHASE
    tp_apr_opts = args["tp-apr"] == "all" ? [false, true] : (args["tp-apr"] == "true" ? [true] : [false])
    tp_uru_opts = args["tp-uru"] == "all" ? [false, true] : (args["tp-uru"] == "true" ? [true] : [false])
    tp_bq_opts  = args["tp-bq"]  == "all" ? [false, true] : (args["tp-bq"]  == "true" ? [true] : [false])
    tp_atr_opts = args["tp-atr"] == "all" ? [false, true] : (args["tp-atr"] == "true" ? [true] : [false])
    
    sqp_strategies = parse_sqp_args(args["tp-sqp"])
    
    M = 1
    for apr in tp_apr_opts, uru in tp_uru_opts, bq in tp_bq_opts, atr in tp_atr_opts, strat in sqp_strategies
        if !bq && atr continue end # Logical constraint: ATR requires BQ
        
        use_quad = (strat != :none)
        
        # Note: If SQP is active, Isotropic mode only (no ATR) is enforced via SLP block, 
        # but kept active here if intended for testing.
        
        b_strat = use_quad ? strat : :identity
        
        is_base = (apr && M == 1 && !bq && !atr && !uru && !use_quad)
        
        name = is_base ? "BASE_2P" : "2P_APR$(Int(apr))_BQ$(Int(bq))_ATR$(Int(atr))_URU$(Int(uru))"
        if use_quad
            name = name * "_SQP_$(uppercase(string(strat)))"
        end
        
        push!(twophase_variants, (
            name=name, is_base=is_base, params=build_twophase_params(apr, M, bq, atr, uru, use_quad, b_strat), 
            APR=apr, M=M, BQ=bq, ATR=atr, URU=uru
        ))
    end
end

# --- SLP PROCESSING ---
slp_variants = []
if RUN_SLP
    slp_bq_opts  = args["slp-bq"]  == "all" ? [false, true] : (args["slp-bq"]  == "true" ? [true] : [false])
    slp_atr_opts = args["slp-atr"] == "all" ? [false, true] : (args["slp-atr"] == "true" ? [true] : [false])

    sqp_strategies = parse_sqp_args(args["slp-sqp"])

    slp_apr_opts = args["slp-apr"] == "all" ? [false, true] : (args["slp-apr"] == "true" ? [true] : [false])

    for apr in slp_apr_opts, bq in slp_bq_opts, atr in slp_atr_opts, strat in sqp_strategies
        if !bq && atr continue end # Logical constraint: ATR requires BQ
        
        use_quad = (strat != :none)
        
        b_strat = use_quad ? strat : :identity
        
        is_base = (apr && !bq && !atr && !use_quad)
        
        name = is_base ? "BASE_SLP" : "SLP_APR$(Int(apr))_BQ$(Int(bq))_ATR$(Int(atr))"
        if use_quad
            name = name * "_SQP_$(uppercase(string(strat)))"
        end
        
        push!(slp_variants, (
            name=name, is_base=is_base, params=build_slp_params(apr, bq, atr, use_quad, b_strat), 
            APR=apr, BQ=bq, ATR=atr
        ))
    end
end

# Base in-memory DataFrames
df_twophase = DataFrame(Problem=String[], nvar=Int[], ncon=Int[], Variant=String[], 
                        APR=Bool[], M=Int[], BQ=Bool[], ATR=Bool[], URU=Bool[], Status=String[], 
                        Iterations=Int[], Time_ms=Float64[], f_final=Float64[], h_norm=Float64[], Is_Base=Bool[])

df_slp = DataFrame(Problem=String[], nvar=Int[], ncon=Int[], Variant=String[], 
                   APR=Bool[], BQ=Bool[], ATR=Bool[], Status=String[], Iterations=Int[], 
                   Time_ms=Float64[], f_final=Float64[], h_norm=Float64[], Is_Base=Bool[])

# ==============================================================================
# 5. PROBLEM SELECTION AND MAIN LOOP
# ==============================================================================
if args["mode"] == "test"
    problems = String.(split(args["problems"], ","))
    println("\nTEST mode activated. Analyzing fixed problem list.")
elseif args["mode"] == "filter"
    println("\nFILTER mode activated. Building query...")
    
    filters = Dict{Symbol, Int}()
    
    print("Filters -> ")
    if args["max-var"] != -1
        filters[:max_var] = args["max-var"]
        print("Max Var: $(args["max-var"]) | ")
    else
        print("Max Var: [No limit] | ")
    end
    
    if args["max-con"] != -1
        filters[:max_con] = args["max-con"]
        print("Max Con: $(args["max-con"])")
    else
        print("Max Con: [No limit]")
    end
    println()
    
    problems = select_sif_problems(; filters...)
else
    println("\n❌ ERROR: Invalid mode ('$(args["mode"])'). Use 'test' or 'filter'.")
    exit(1)
end

println("Total selected problems: $(length(problems))")

if length(problems) == 0
    println("No problems found matching the selected parameters. Exiting.")
    exit(0)
end

function get_status_string(status_code::Int)
    if status_code == 0 return "KKT_OK"
    elseif status_code == 1 return "SMALL_STEP"
    elseif status_code == 2 return "MAX_IT"
    elseif status_code == 3 return "INFEASIBLE_STATIONARY"
    elseif status_code == 4 return "STALLED_INFEASIBLE"
    else return "UNKNOWN_$(status_code)" end
end

println("\n🔥 EXECUTING WARM-UP (JIT Precompilation of ALL variants)...")
try
    # Using a pure-Julia custom problem to avoid CUTEst DLL locks and Fortran memory overhead
    nlp_w = SharedTypes.create_rosenbrock_problem1()
    prob_w = SharedTypes.build_optimization_problem(nlp_w)
    x0_w = clamp.(nlp_w.meta.x0, prob_w.xl, prob_w.xu)

    if RUN_TWOPHASE && length(twophase_variants) > 0
        print("  Compiling Two-Phase variants... ")
        for var in twophase_variants
            # Run once to force compilation of all code paths (SQP, RipQP, Gurobi, etc.)
            Generic_module_Twophase.two_phase_optimization(prob_w, x0_w, var.params; history=false)
        end
        println("OK")
    end

    if RUN_SLP && length(slp_variants) > 0
        print("  Compiling SLP variants... ")
        for var in slp_variants
            Generic_module_slp.solve_slp_trust_region(prob_w, x0_w, var.params)
        end
        println("OK")
    end
    
    println("🧹 Warm-up complete. Pure Julia problem used (No DLL locks).")
catch e
    println("⚠️  Warning: Warm-up failed ($e). The first recorded times may be inflated.")
end
# ==============================================================================
# 5.2 MAIN LOOP
# ==============================================================================
function main()
    for prob_name in problems
        local nlp = nothing
        try
            nlp = CUTEstModel{Float64}(prob_name)
            
            # # --- MEMORY SAFETY LOCK ---
            # # Reject overly dense Hessians that crash the SparseArrays constructor
            # if nlp.meta.nnzh > 500000
            #     println("\n⏭️  SKIPPING: $prob_name [Hessian too dense: $(nlp.meta.nnzh) nnz]")
            #     finalize(nlp)
            #     continue
            # end
            
            problem = SharedTypes.build_optimization_problem(nlp)
            x0 = clamp.(nlp.meta.x0, problem.xl, problem.xu)
            dim, ncon = nlp.meta.nvar, nlp.meta.ncon

            println("\n🚀 PROCESSING: $prob_name [Var: $dim | Con: $ncon]")

            if RUN_TWOPHASE
                for var in twophase_variants
                    print("  [Two-Phase] > $(var.name)... ")
                    exec_time = @elapsed begin
                        y, λ, θ, iter, status, hist, log = Generic_module_Twophase.two_phase_optimization(problem, x0, var.params; history=false)
                    end
                    push!(df_twophase, (prob_name, dim, ncon, var.name, var.APR, var.M, var.BQ, var.ATR, var.URU, get_status_string(status), iter, exec_time * 1000.0, problem.f(y), norm(problem.h(y)), var.is_base))
                    println("Done. ($(round(exec_time * 1000, digits=1))ms)")
                end
            end

            if RUN_SLP
                for var in slp_variants
                    print("  [SLP] > $(var.name)... ")
                    exec_time = @elapsed begin
                        y, λ, θ, iter, status, hist, log = Generic_module_slp.solve_slp_trust_region(problem, x0, var.params)
                    end
                    push!(df_slp, (prob_name, dim, ncon, var.name, var.APR, var.BQ, var.ATR, get_status_string(status), iter, exec_time * 1000.0, problem.f(y), norm(problem.h(y)), var.is_base))
                    println("Done. ($(round(exec_time * 1000, digits=1))ms)")
                end
            end
        catch e
            println("❌ ERROR in $prob_name: $e")
        finally
            isnothing(nlp) || finalize(nlp)
            
            # Incremental backup to prevent data loss in case of a crash
            CSV.write(joinpath(results_dir, "partial_backup_twophase.csv"), df_twophase)
            CSV.write(joinpath(results_dir, "partial_backup_slp.csv"), df_slp)
        end
    end
end

main()
# Run the statistics module
Generic_module_Stats.run_statistical_analysis(results_dir)