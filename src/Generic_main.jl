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

include("Generic_module_unif.jl")   
using .Generic_module_unif

include("Generic_module_Stats.jl")
using .Generic_module_Stats


# ==============================================================================
# 2. COMMAND LINE ARGUMENTS CONFIGURATION
# ==============================================================================
function parse_commandline()
    s = ArgParseSettings(description="UNIF and TwoPhase Optimization Benchmark - Remote Execution")

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
        "--run-unif"
            help = "Run UNIF variants (true/false)."
            arg_type = Bool
            default = true
        "--run-twophase"
            help = "Run Two-Phase variants (true/false)."
            arg_type = Bool
            default = true
        
        # Two-Phase Parameters
        "--tp-sar"
            help = "Adaptive ratio (Two-Phase). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--tp-ph"
            help = "Parabolic step (Two-Phase). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--tp-atr"
            help = "Anisotropic region (Two-Phase). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--tp-rru"
            help = "Ratio update (Two-Phase). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--tp-sqp"
            help = "SQP strategies for Two-Phase. Comma-separated (e.g., none,identity,spectral,exact) or 'all'."
            arg_type = String
            default = "all"
            
        # UNIF Parameters
        "--unif-sar"
            help = "Adaptive ratio (UNIF). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--unif-ph"
            help = "Parabolic step (UNIF). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--unif-atr"
            help = "Anisotropic region (UNIF). Options: 'all', 'true', 'false'."
            arg_type = String
            default = "all"
        "--unif-sqp"
            help = "SQP strategies for UNIF. Comma-separated (e.g., none,identity,spectral,exact) or 'all'."
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
const RUN_UNIF      = args["run-unif"]

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
function build_twophase_params(sar, M, ph, atr, rru, use_quad, b_strat)
    return SharedTypes.TwoPhaseParams(
        max_outer_iter = common_max_iter, δ0_opt = common_delta0, δ0_resto = common_delta0,
        tolG = common_tolG, tolF = common_tolF, tolS = common_tolS, maxcount = common_maxcount,
        verbose_out = common_verbose, use_unif_stopping = true, rfeas = 1e-12, δmin = 1e-12, 
        δmax = 1e16, r_resto = 0.9, τ1 = 0.1, τ2 = 0.25, αL = 1e-8, αR = 1e-8,
        θ0 = 0.90, αΦ = 1e-8, max_iter_resto = 500, verbose_in = common_verbose, debugverbose = false,
        strong_agreement_rule = sar, non_monotone_M = M, parabolic_heuristic = ph, anisotropic_trust_region = atr, restoration_ratio_update = rru,
        use_quadratic = use_quad, B_update_strategy = b_strat, quadratic_solver = :ripqp
    )
end

function build_unif_params(sar, ph, atr, use_quad, b_strat)
    return SharedTypes.UNIFParams(
        delta0 = common_delta0, tolG = common_tolG, tolF = common_tolF, tolS = common_tolS,
        maxiter = common_max_iter, maxcount = common_maxcount, verbose = common_verbose,
        verbose_out = common_verbose, debugverbose = false, parabolic_heuristic = ph, anisotropic_trust_region = atr,
        strong_agreement_rule = sar, use_quadratic = use_quad, B_update_strategy = b_strat, quadratic_solver = :ripqp
    )
end

# --- TWO-PHASE PROCESSING ---
twophase_variants = []
if RUN_TWOPHASE
    tp_sar_opts = args["tp-sar"] == "all" ? [false, true] : (args["tp-sar"] == "true" ? [true] : [false])
    tp_rru_opts = args["tp-rru"] == "all" ? [false, true] : (args["tp-rru"] == "true" ? [true] : [false])
    tp_ph_opts  = args["tp-ph"]  == "all" ? [false, true] : (args["tp-ph"]  == "true" ? [true] : [false])
    tp_atr_opts = args["tp-atr"] == "all" ? [false, true] : (args["tp-atr"] == "true" ? [true] : [false])
    
    sqp_strategies = parse_sqp_args(args["tp-sqp"])
    
    M = 1
    for sar in tp_sar_opts, rru in tp_rru_opts, ph in tp_ph_opts, atr in tp_atr_opts, strat in sqp_strategies
        if !ph && atr continue end # Logical constraint: ATR requires PH
        
        use_quad = (strat != :none)
        
        # Note: If SQP is active, Isotropic mode only (no ATR) is enforced via UNIF block, 
        # but kept active here if intended for testing.
        
        b_strat = use_quad ? strat : :identity
        
        is_base = (sar && M == 1 && !ph && !atr && !rru && !use_quad)
        
        name = is_base ? "BASE_2P" : "2P_SAR$(Int(sar))_PH$(Int(ph))_ATR$(Int(atr))_RRU$(Int(rru))"
        if use_quad
            name = name * "_SQP_$(uppercase(string(strat)))"
        end
        
        push!(twophase_variants, (
            name=name, is_base=is_base, params=build_twophase_params(sar, M, ph, atr, rru, use_quad, b_strat), 
            SAR=sar, M=M, PH=ph, ATR=atr, RRU=rru
        ))
    end
end

# --- UNIF PROCESSING ---
unif_variants = []
if RUN_UNIF
    unif_ph_opts  = args["unif-ph"]  == "all" ? [false, true] : (args["unif-ph"]  == "true" ? [true] : [false])
    unif_atr_opts = args["unif-atr"] == "all" ? [false, true] : (args["unif-atr"] == "true" ? [true] : [false])

    sqp_strategies = parse_sqp_args(args["unif-sqp"])

    unif_sar_opts = args["unif-sar"] == "all" ? [false, true] : (args["unif-sar"] == "true" ? [true] : [false])

    for sar in unif_sar_opts, ph in unif_ph_opts, atr in unif_atr_opts, strat in sqp_strategies
        if !ph && atr continue end # Logical constraint: ATR requires PH
        
        use_quad = (strat != :none)
        
        b_strat = use_quad ? strat : :identity
        
        is_base = (sar && !ph && !atr && !use_quad)
        
        name = is_base ? "BASE_UNIF" : "UNIF_SAR$(Int(sar))_PH$(Int(ph))_ATR$(Int(atr))"
        if use_quad
            name = name * "_SQP_$(uppercase(string(strat)))"
        end
        
        push!(unif_variants, (
            name=name, is_base=is_base, params=build_unif_params(sar, ph, atr, use_quad, b_strat), 
            SAR=sar, PH=ph, ATR=atr
        ))
    end
end

# Base in-memory DataFrames
df_twophase = DataFrame(Problem=String[], nvar=Int[], ncon=Int[], Variant=String[], 
                        SAR=Bool[], M=Int[], PH=Bool[], ATR=Bool[], RRU=Bool[], Status=String[], 
                        Iterations=Int[], Time_ms=Float64[], f_final=Float64[], h_norm=Float64[], Is_Base=Bool[])

df_unif = DataFrame(Problem=String[], nvar=Int[], ncon=Int[], Variant=String[], 
                   SAR=Bool[], PH=Bool[], ATR=Bool[], Status=String[], Iterations=Int[], 
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

    if RUN_UNIF && length(unif_variants) > 0
        print("  Compiling UNIF variants... ")
        for var in unif_variants
            Generic_module_unif.solve_unif_trust_region(prob_w, x0_w, var.params)
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
                    push!(df_twophase, (prob_name, dim, ncon, var.name, var.SAR, var.M, var.PH, var.ATR, var.RRU, get_status_string(status), iter, exec_time * 1000.0, problem.f(y), norm(problem.h(y)), var.is_base))
                    println("Done. ($(round(exec_time * 1000, digits=1))ms)")
                end
            end

            if RUN_UNIF
                for var in unif_variants
                    print("  [UNIF] > $(var.name)... ")
                    exec_time = @elapsed begin
                        y, λ, θ, iter, status, hist, log = Generic_module_unif.solve_unif_trust_region(problem, x0, var.params)
                    end
                    push!(df_unif, (prob_name, dim, ncon, var.name, var.SAR, var.PH, var.ATR, get_status_string(status), iter, exec_time * 1000.0, problem.f(y), norm(problem.h(y)), var.is_base))
                    println("Done. ($(round(exec_time * 1000, digits=1))ms)")
                end
            end
        catch e
            println("❌ ERROR in $prob_name: $e")
        finally
            isnothing(nlp) || finalize(nlp)
            
            # Incremental backup to prevent data loss in case of a crash
            CSV.write(joinpath(results_dir, "partial_backup_twophase.csv"), df_twophase)
            CSV.write(joinpath(results_dir, "partial_backup_unif.csv"), df_unif)
        end
    end
end

main()
# Run the statistics module
Generic_module_Stats.run_statistical_analysis(results_dir)