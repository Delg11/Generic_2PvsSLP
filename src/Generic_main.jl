# ==============================================================================
# Proprietary Software • All Rights Reserved
# ==============================================================================
# 1. IMPORTS AND INCLUDES
# ==============================================================================
using ArgParse
using Distributed
using Dates
using DataFrames
using CSV
using XLSX
using Plots

# ==============================================================================
# 2. COMMAND LINE ARGUMENTS CONFIGURATION
# ==============================================================================
function parse_commandline()
    s = ArgParseSettings(description="UNIF and TwoPhase Optimization Benchmark - Remote Execution")

    @add_arg_table s begin
        "--mode", "-m"
            help = "Execution mode: 'test', 'small' (fast benchmark on SMALL_PROBLEMS), or 'filter'."
            arg_type = String
            default = "test"
        "--problems", "-p"
            help = "Comma-separated list of problems."
            arg_type = String
            default = "HS6"
        "--workers", "-w"
            help = "Number of parallel workers (1 = sequential, 0 = auto-detect cores, N > 1 = custom)."
            arg_type = Int
            default = 1
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
        return [:exact, :identity, :spectral, :reciprocal, :exponential, :quasi_newton]
    else
        return Symbol.(strip.(split(arg_str, ",")))
    end
end

args = parse_commandline()

# Carrega ambiente unificado no Master
include(normpath(joinpath(@__DIR__, "worker_env.jl")))
include(normpath(joinpath(@__DIR__, "Generic_module_Stats.jl")))
using .Generic_module_Stats
include(normpath(joinpath(@__DIR__, "reproducibility.jl")))
using .Reproducibility

# ==============================================================================
# 4. GENERAL CONFIGURATION AND ALGORITHM SELECTION
# ==============================================================================
println("\n" * "=" ^ 80)
println("⚙️  CONFIGURING VARIANTS AND EXPORT ENVIRONMENT")
println("=" ^ 80)

println("🖥️  System: $(Sys.CPU_THREADS) threads available")
println("🚀 Execution: $(nworkers() > 1 ? "Parallel ($(nworkers()) distributed workers)" : "Sequential (1 worker)")")
println("-" ^ 80)

const RUN_TWOPHASE = args["run-twophase"]
const RUN_UNIF     = args["run-unif"]

common_max_iter = 5000
common_delta0   = 0.1
common_tolG     = 1e-3
common_tolF     = 5e-2
common_tolS     = 1e-4
common_maxcount = 3
common_verbose  = false 

current_datetime = Dates.format(now(), "yyyy-mm-dd_HH-MM")
benchmarks_root = normpath(joinpath(@__DIR__, "..", "benchmarks"))
results_dir = joinpath(benchmarks_root, "Benchmark_Results_$(current_datetime)")
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
    tp_rru_opts = [true] # Forçado para true conforme restrição
    tp_ph_opts  = args["tp-ph"]  == "all" ? [false, true] : (args["tp-ph"]  == "true" ? [true] : [false])
    tp_atr_opts = args["tp-atr"] == "all" ? [false, true] : (args["tp-atr"] == "true" ? [true] : [false])
    
    sqp_strategies = parse_sqp_args(args["tp-sqp"])
    
    M = 1
    for sar in tp_sar_opts, rru in tp_rru_opts, ph in tp_ph_opts, atr in tp_atr_opts, strat in sqp_strategies
        if !ph && atr continue end
        if ph && sar continue end
        
        use_quad = (strat != :none)
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
        if !ph && atr continue end
        if ph && sar continue end
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

# Base in-memory DataFrames with Git Commit Provenance
df_twophase = DataFrame(Problem=String[], Git_Commit=String[], nvar=Int[], ncon=Int[], Variant=String[], 
                        SAR=Bool[], M=Int[], PH=Bool[], ATR=Bool[], RRU=Bool[], Status=String[], 
                        Iterations=Int[], Time_ms=Float64[], f_final=Float64[], h_norm=Float64[], Is_Base=Bool[])

df_unif = DataFrame(Problem=String[], Git_Commit=String[], nvar=Int[], ncon=Int[], Variant=String[], 
                   SAR=Bool[], PH=Bool[], ATR=Bool[], Status=String[], Iterations=Int[], 
                   Time_ms=Float64[], f_final=Float64[], h_norm=Float64[], Is_Base=Bool[])

# ==============================================================================
# 5. PROBLEM SELECTION
# ==============================================================================
if args["mode"] == "test"
    problems = String.(split(args["problems"], ","))
    println("\nTEST mode activated. Analyzing fixed problem list: $(join(problems, ", "))")
elseif args["mode"] == "small"
    problems = copy(SharedTypes.SMALL_PROBLEMS)
    println("\n🎯 SMALL mode activated. Running fast benchmark on $(length(problems)) problems: $(join(problems, ", "))")
elseif args["mode"] == "filter"
    println("\nFILTER mode activated. Building query...")
    filters = Dict{Symbol, Int}()
    if args["max-var"] != -1
        filters[:max_var] = args["max-var"]
    end
    if args["max-con"] != -1
        filters[:max_con] = args["max-con"]
    end
    problems = select_sif_problems(; filters...)
else
    println("\n❌ ERROR: Invalid mode ('$(args["mode"])'). Use 'test', 'small', or 'filter'.")
    exit(1)
end

println("Total selected problems: $(length(problems))")
if length(problems) == 0
    println("No problems found matching the selected parameters. Exiting.")
    exit(0)
end

# ==============================================================================
# 5.2 MAIN FUNCTION
# ==============================================================================
function main()
    benchmark_start_time = now()
    git_commit_short = Reproducibility.get_git_short_commit()
    actual_workers = ensure_parallel_workers(args["workers"])
    is_parallel = actual_workers > 1

    println("\n" * "=" ^ 80)
    println("🚀 RUNNING BENCHMARK ACROSS $(length(problems)) PROBLEMS")
    println("⚙️  Mode: $(is_parallel ? "Parallel ($(nworkers()) distributed workers)" : "Sequential (1 worker)")")
    println("🔖 Git Commit: $git_commit_short")
    println("=" ^ 80)

    if is_parallel
        # 1. Precompile CUTEst problems sequentially on master to avoid Windows file locks
        print("🔧 Pre-compiling $(length(problems)) CUTEst problems on master... ")
        for p in problems
            try
                nlp = CUTEstModel{Float64}(p; decode=true)
                finalize(nlp)
            catch
            end
        end
        println("OK")

        # 2. Worker warmup
        print("🔥 Warming up workers (JIT)... ")
        run_worker_warmup()
        @sync for w in workers()
            @async remotecall_fetch(run_worker_warmup, w)
        end
        println("OK")

        # 3. Parallel dispatch with pmap
        total_time = @elapsed begin
            if length(problems) < nworkers()
                # Fine-grained parallelization: distribute (problem, variant) tasks to saturate all workers
                println("⚡ Distributing variant tasks across $(nworkers()) workers (fine-grained load balancing)...")
                tasks = []
                for p in problems
                    if RUN_TWOPHASE
                        for v in twophase_variants
                            push!(tasks, (p, :twophase, v))
                        end
                    end
                    if RUN_UNIF
                        for v in unif_variants
                            push!(tasks, (p, :unif, v))
                        end
                    end
                end

                results = pmap(t -> solve_main_variant_job(t[1], t[2], t[3], false), tasks)

                for (kind, r) in results
                    isnothing(r) && continue
                    if kind == :twophase
                        push!(df_twophase, (r[1], git_commit_short, r[2:end]...))
                        println("  [Two-Phase] $(r[1]) > $(r[4])... $(r[10]) in $(round(r[12], digits=1))ms")
                    elseif kind == :unif
                        push!(df_unif, (r[1], git_commit_short, r[2:end]...))
                        println("  [UNIF] $(r[1]) > $(r[4])... $(r[8]) in $(round(r[10], digits=1))ms")
                    end
                end
                RUN_TWOPHASE && CSV.write(joinpath(results_dir, "partial_backup_twophase.csv"), df_twophase)
                RUN_UNIF && CSV.write(joinpath(results_dir, "partial_backup_unif.csv"), df_unif)
            else
                # Problem-level parallelization: 1 problem per job (minimizes CUTEst model load overhead)
                println("⚡ Distributing $(length(problems)) problem batches across $(nworkers()) workers...")
                results = pmap(p -> solve_main_problem_job(p, twophase_variants, unif_variants, RUN_TWOPHASE, RUN_UNIF, false), problems)

                for (prob_name, (tp_rows, unif_rows)) in zip(problems, results)
                    println("\n📦 Finished problem: $prob_name")
                    for r in tp_rows
                        push!(df_twophase, (r[1], git_commit_short, r[2:end]...))
                        println("  [Two-Phase] > $(r[4])... $(r[10]) in $(round(r[12], digits=1))ms")
                    end
                    for r in unif_rows
                        push!(df_unif, (r[1], git_commit_short, r[2:end]...))
                        println("  [UNIF] > $(r[4])... $(r[8]) in $(round(r[10], digits=1))ms")
                    end
                    # Incremental backups
                    RUN_TWOPHASE && CSV.write(joinpath(results_dir, "partial_backup_twophase.csv"), df_twophase)
                    RUN_UNIF && CSV.write(joinpath(results_dir, "partial_backup_unif.csv"), df_unif)
                end
            end
        end

        if !isinteractive()
            rmprocs(workers())
        end
    else
        # Sequential Execution
        total_time = @elapsed begin
            for prob_name in problems
                local nlp = nothing
                try
                    nlp = CUTEstModel{Float64}(prob_name; decode=true)
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
                            push!(df_twophase, (prob_name, git_commit_short, dim, ncon, var.name, var.SAR, var.M, var.PH, var.ATR, var.RRU, get_status_string(status), iter, exec_time * 1000.0, problem.f(y), norm(problem.h(y)), var.is_base))
                            println("Done. ($(round(exec_time * 1000, digits=1))ms)")
                        end
                    end

                    if RUN_UNIF
                        for var in unif_variants
                            print("  [UNIF] > $(var.name)... ")
                            exec_time = @elapsed begin
                                y, λ, θ, iter, status, hist, log = Generic_module_unif.solve_unif_trust_region(problem, x0, var.params)
                            end
                            push!(df_unif, (prob_name, git_commit_short, dim, ncon, var.name, var.SAR, var.PH, var.ATR, get_status_string(status), iter, exec_time * 1000.0, problem.f(y), norm(problem.h(y)), var.is_base))
                            println("Done. ($(round(exec_time * 1000, digits=1))ms)")
                        end
                    end
                catch e
                    println("❌ ERROR in $prob_name: $e")
                finally
                    isnothing(nlp) || finalize(nlp)
                    RUN_TWOPHASE && CSV.write(joinpath(results_dir, "partial_backup_twophase.csv"), df_twophase)
                    RUN_UNIF && CSV.write(joinpath(results_dir, "partial_backup_unif.csv"), df_unif)
                end
            end
        end
    end

    # Save final consolidated files
    println("\n" * "=" ^ 80)
    @printf("🏁 Benchmark completed in %.2f seconds\n", total_time)
    println("💾 Saving final consolidated results...")
    if RUN_TWOPHASE
        CSV.write(joinpath(results_dir, "Consolidated_Twophase.csv"), df_twophase)
    end
    if RUN_UNIF
        CSV.write(joinpath(results_dir, "Consolidated_UNIF.csv"), df_unif)
    end
    println("Done! Results saved in: $results_dir")

    # Reproducibility & Environment Metadata
    println("\n🔬 Generating reproducibility and provenance metadata...")
    meta = Reproducibility.collect_environment_metadata(
        mode = args["mode"],
        workers_count = is_parallel ? nworkers() : 1,
        start_time = benchmark_start_time
    )
    meta["execution"]["end_time"] = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
    meta["execution"]["elapsed_seconds"] = total_time
    df_meta = Reproducibility.save_reproducibility_artifacts(results_dir, meta)
    println("   • benchmark_manifest.json (machine-readable)")
    println("   • ENVIRONMENT_SPECS.md (paper/thesis copy-paste)")
    println("   • environment_metadata.csv")

    # Run statistical analysis
    try
        println("\n📊 Generating statistical performance profiles...")
        Generic_module_Stats.run_statistical_analysis(results_dir, df_meta)
    catch e
        println("⚠️ Note: Statistical analysis finished with: $e")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
