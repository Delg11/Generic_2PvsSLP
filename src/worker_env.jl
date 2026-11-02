# ==============================================================================
# WORKER ENVIRONMENT AND PROBLEM SOLVER FUNCTIONS
# ==============================================================================
# Este arquivo é incluído tanto no processo Master quanto em todos os Workers.
# Garante consistência de ambiente independente de como Julia foi iniciado
# (Linha de comando, terminal SSH, ou interativamente no VS Code REPL).
# ==============================================================================

using LinearAlgebra
using Printf
using Statistics
using Dates
using CUTEst
using DataFrames
using CSV
using JuMP
using Gurobi

# Carrega módulos essenciais no processo local (Master ou Worker)
include(normpath(joinpath(@__DIR__, "Generic_Sharedtypes.jl")))
using .SharedTypes

include(normpath(joinpath(@__DIR__, "Generic_module_ComputeB.jl")))
using .Generic_module_ComputeB

include(normpath(joinpath(@__DIR__, "Generic_module_Twophase.jl")))
using .Generic_module_Twophase

include(normpath(joinpath(@__DIR__, "Generic_module_unif.jl")))
using .Generic_module_unif

function get_status_name(status_code::Int)
    if status_code == 0 return "KKT_OK"
    elseif status_code == 1 return "SMALL_STEP"
    elseif status_code == 2 return "MAX_IT"
    elseif status_code == 3 return "INFEASIBLE_STATIONARY"
    elseif status_code == 4 return "STALLED_INFEASIBLE"
    elseif status_code == 5 return "TIMEOUT"
    else return "UNKNOWN_$(status_code)" end
end
const get_status_string = get_status_name

"""
    solve_fast_benchmark_problem(prob_name::String, max_iter::Int; decode::Bool=false)

Resolve as 4 variantes de teste para um problema CUTEst.
"""
function solve_fast_benchmark_problem(prob_name::String, max_iter::Int; decode::Bool=false)
    delta0   = 0.1
    tolG     = 1e-3
    tolF     = 5e-2
    tolS     = 1e-4
    maxcount = 3

    tp_base = SharedTypes.TwoPhaseParams(
        max_outer_iter = max_iter, δ0_opt = delta0, δ0_resto = delta0,
        tolG = tolG, tolF = tolF, tolS = tolS, maxcount = maxcount,
        verbose_out = false, verbose_in = false, debugverbose = false,
        use_unif_stopping = true, rfeas = 1e-12, δmin = 1e-12, δmax = 1e16,
        r_resto = 0.9, τ1 = 0.1, τ2 = 0.25, τ3 = 2.0, αL = 1e-8, αR = 1e-8,
        θ0 = 0.90, strong_agreement_rule = true, parabolic_heuristic = false,
        anisotropic_trust_region = false, restoration_ratio_update = false,
        use_quadratic = false, non_monotone_M = 1
    )

    tp_iso = SharedTypes.TwoPhaseParams(
        max_outer_iter = max_iter, δ0_opt = delta0, δ0_resto = delta0,
        tolG = tolG, tolF = tolF, tolS = tolS, maxcount = maxcount,
        verbose_out = false, verbose_in = false, debugverbose = false,
        use_unif_stopping = true, rfeas = 1e-12, δmin = 1e-12, δmax = 1e16,
        r_resto = 0.9, τ1 = 0.1, τ2 = 0.25, τ3 = 2.0, αL = 1e-8, αR = 1e-8,
        θ0 = 0.90, strong_agreement_rule = false, parabolic_heuristic = true,
        anisotropic_trust_region = false, restoration_ratio_update = false,
        use_quadratic = false, non_monotone_M = 1
    )

    unif_base = SharedTypes.UNIFParams(
        maxiter = max_iter, delta0 = delta0, tolG = tolG, tolF = tolF, tolS = tolS,
        maxcount = maxcount, verbose = false,
        strong_agreement_rule = true, parabolic_heuristic = false,
        anisotropic_trust_region = false, use_quadratic = false
    )

    unif_iso = SharedTypes.UNIFParams(
        maxiter = max_iter, delta0 = delta0, tolG = tolG, tolF = tolF, tolS = tolS,
        maxcount = maxcount, verbose = false,
        strong_agreement_rule = false, parabolic_heuristic = true,
        anisotropic_trust_region = false, use_quadratic = false
    )

    configs = [
        ("2P_BASE",   :twophase, tp_base),
        ("2P_ISO",    :twophase, tp_iso),
        ("UNIF_BASE", :unif,     unif_base),
        ("UNIF_ISO",  :unif,     unif_iso)
    ]

    results = []
    local nlp = nothing
    try
        nlp = CUTEstModel{Float64}(prob_name; decode=decode)
        prob = SharedTypes.build_optimization_problem(nlp)
        x0 = clamp.(nlp.meta.x0, prob.xl, prob.xu)

        for (name, kind, params) in configs
            t_ms = 0.0
            f_val = NaN
            h_val = NaN
            iter_count = 0
            status_str = "ERROR"

            if kind == :twophase
                t_sec = @elapsed begin
                    y, λ, θ, iters, status_code, _, _ = Generic_module_Twophase.two_phase_optimization(prob, x0, params; history=false)
                end
                t_ms = t_sec * 1000.0
                iter_count = iters
                f_val = prob.f(y)
                h_val = norm(prob.h(y))
                status_str = status_code == 0 ? "KKT_OK" : (status_code == 1 ? "SMALL_STEP" : (status_code == 2 ? "MAX_IT" : "FAIL_$status_code"))

            elseif kind == :unif
                t_sec = @elapsed begin
                    y, λ, θ, iters, status_code, _, _ = Generic_module_unif.solve_unif_trust_region(prob, x0, params)
                end
                t_ms = t_sec * 1000.0
                iter_count = iters
                f_val = prob.f(y)
                h_val = norm(prob.h(y))
                status_str = status_code == 0 ? "KKT_OK" : (status_code == 1 ? "SMALL_STEP" : (status_code == 2 ? "MAX_IT" : "FAIL_$status_code"))
            end

            push!(results, (prob_name=prob_name, name=name, status_str=status_str, iter_count=iter_count, t_ms=t_ms, f_val=f_val, h_val=h_val, nvar=prob.nvar, ncon=prob.ncon))
        end
    catch err
        println("❌ Erro ao processar $prob_name: $err")
    finally
        isnothing(nlp) || finalize(nlp)
    end

    return results
end

"""
    solve_main_problem_job(prob_name::String, tp_vars, un_vars, run_tp::Bool, run_un::Bool, decode_flag::Bool)

Resolve um lote de variantes completas para o Generic_main.jl.
"""
function solve_main_problem_job(prob_name::String, tp_vars, un_vars, run_tp::Bool, run_un::Bool, decode_flag::Bool)
    tp_rows = []
    unif_rows = []
    local nlp = nothing
    try
        nlp = CUTEstModel{Float64}(prob_name; decode=decode_flag)
        problem = SharedTypes.build_optimization_problem(nlp)
        x0 = clamp.(nlp.meta.x0, problem.xl, problem.xu)
        dim, ncon = nlp.meta.nvar, nlp.meta.ncon

        if run_tp
            for (v_idx, var) in enumerate(tp_vars)
                try
                    exec_time = @elapsed begin
                        y, λ, θ, iter, status, hist, log = Generic_module_Twophase.two_phase_optimization(problem, x0, var.params; history=false)
                    end
                    fy = try problem.f(y) catch; NaN end
                    hy = try norm(problem.h(y)) catch; NaN end
                    st_name = get_status_name(status)
                    push!(tp_rows, (prob_name, dim, ncon, var.name, var.SAR, var.M, var.PH, var.ATR, var.RRU, st_name, iter, exec_time * 1000.0, isfinite(fy) ? fy : NaN, isfinite(hy) ? hy : NaN, var.is_base))
                    @printf("  [%s] [W%d] %-10s > [TP %2d/%2d] %-42s... %-8s (iter: %3d, %6.1fms)\n",
                            Dates.format(now(), "HH:MM:SS"), myid(), prob_name, v_idx, length(tp_vars), var.name, st_name, iter, exec_time * 1000.0)
                    flush(stdout)
                catch e_var
                    println("⚠️ Worker $(myid()) exception in TwoPhase $prob_name ($(var.name)): $e_var")
                    push!(tp_rows, (prob_name, dim, ncon, var.name, var.SAR, var.M, var.PH, var.ATR, var.RRU, "ERROR", 0, 0.0, NaN, NaN, var.is_base))
                end
            end
        end

        if run_un
            for (v_idx, var) in enumerate(un_vars)
                try
                    exec_time = @elapsed begin
                        y, λ, θ, iter, status, hist, log = Generic_module_unif.solve_unif_trust_region(problem, x0, var.params)
                    end
                    fy = try problem.f(y) catch; NaN end
                    hy = try norm(problem.h(y)) catch; NaN end
                    st_name = get_status_name(status)
                    push!(unif_rows, (prob_name, dim, ncon, var.name, var.SAR, var.PH, var.ATR, st_name, iter, exec_time * 1000.0, isfinite(fy) ? fy : NaN, isfinite(hy) ? hy : NaN, var.is_base))
                    @printf("  [%s] [W%d] %-10s > [UNIF %2d/%2d] %-40s... %-8s (iter: %3d, %6.1fms)\n",
                            Dates.format(now(), "HH:MM:SS"), myid(), prob_name, v_idx, length(un_vars), var.name, st_name, iter, exec_time * 1000.0)
                    flush(stdout)
                catch e_var
                    println("⚠️ Worker $(myid()) exception in UNIF $prob_name ($(var.name)): $e_var")
                    push!(unif_rows, (prob_name, dim, ncon, var.name, var.SAR, var.PH, var.ATR, "ERROR", 0, 0.0, NaN, NaN, var.is_base))
                end
            end
        end
    catch e
        println("❌ ERROR on worker $(myid()) in $prob_name: $e")
    finally
        isnothing(nlp) || finalize(nlp)
    end
    return (tp_rows, unif_rows)
end

"""
    solve_main_variant_job(prob_name::String, kind::Symbol, var, decode_flag::Bool)

Resolve uma única variante de forma independente em um worker paralelo (para paralelismo refinado).
"""
function solve_main_variant_job(prob_name::String, kind::Symbol, var, decode_flag::Bool)
    local nlp = nothing
    try
        nlp = CUTEstModel{Float64}(prob_name; decode=decode_flag)
        problem = SharedTypes.build_optimization_problem(nlp)
        x0 = clamp.(nlp.meta.x0, problem.xl, problem.xu)
        dim, ncon = nlp.meta.nvar, nlp.meta.ncon

        if kind == :twophase
            exec_time = @elapsed begin
                y, λ, θ, iter, status, hist, log = Generic_module_Twophase.two_phase_optimization(problem, x0, var.params; history=false)
            end
            row = (prob_name, dim, ncon, var.name, var.SAR, var.M, var.PH, var.ATR, var.RRU, get_status_name(status), iter, exec_time * 1000.0, problem.f(y), norm(problem.h(y)), var.is_base)
            return (:twophase, row)
        elseif kind == :unif
            exec_time = @elapsed begin
                y, λ, θ, iter, status, hist, log = Generic_module_unif.solve_unif_trust_region(problem, x0, var.params)
            end
            row = (prob_name, dim, ncon, var.name, var.SAR, var.PH, var.ATR, get_status_name(status), iter, exec_time * 1000.0, problem.f(y), norm(problem.h(y)), var.is_base)
            return (:unif, row)
        end
    catch e
        println("❌ ERROR on worker $(myid()) in $prob_name ($(var.name)): $e")
        return (kind, nothing)
    finally
        isnothing(nlp) || finalize(nlp)
    end
end

function run_worker_warmup()
    local nlp_cutest = nothing
    try
        # 1. Warmup completo com CUTEst para compilar antecipadamente os bindings C e especializações de tipos
        nlp_cutest = try
            CUTEstModel{Float64}("HS6"; decode=false)
        catch
            try
                CUTEstModel{Float64}("HS6"; decode=true)
            catch
                nothing
            end
        end
        if !isnothing(nlp_cutest)
            prob_cutest = SharedTypes.build_optimization_problem(nlp_cutest)
            x0_cutest = clamp.(nlp_cutest.meta.x0, prob_cutest.xl, prob_cutest.xu)

            # 2. Aquece TwoPhase (Base sem heurística e variante com heurística)
            tp_warm_base = SharedTypes.TwoPhaseParams(max_outer_iter=2, verbose_out=false, verbose_in=false, parabolic_heuristic=false, anisotropic_trust_region=false)
            tp_warm_ph   = SharedTypes.TwoPhaseParams(max_outer_iter=2, verbose_out=false, verbose_in=false, parabolic_heuristic=true, anisotropic_trust_region=false)
            Generic_module_Twophase.two_phase_optimization(prob_cutest, x0_cutest, tp_warm_base; history=false)
            Generic_module_Twophase.two_phase_optimization(prob_cutest, x0_cutest, tp_warm_ph; history=false)

            # 3. Aquece UNIF (Base sem heurística e variante com heurística)
            unif_warm_base = SharedTypes.UNIFParams(maxiter=2, verbose=false, parabolic_heuristic=false, anisotropic_trust_region=false)
            unif_warm_ph   = SharedTypes.UNIFParams(maxiter=2, verbose=false, parabolic_heuristic=true, anisotropic_trust_region=false)
            Generic_module_unif.solve_unif_trust_region(prob_cutest, x0_cutest, unif_warm_base)
            Generic_module_unif.solve_unif_trust_region(prob_cutest, x0_cutest, unif_warm_ph)
        else
            error("CUTEst HS6 warmup failed, using Rosenbrock fallback")
        end
    catch
        try
            nlp_w = SharedTypes.create_rosenbrock_problem1()
            prob_w = SharedTypes.build_optimization_problem(nlp_w)
            x0_w = clamp.(nlp_w.meta.x0, prob_w.xl, prob_w.xu)
            tp_test = SharedTypes.TwoPhaseParams(max_outer_iter=2, verbose_out=false, verbose_in=false)
            unif_test = SharedTypes.UNIFParams(maxiter=2, verbose=false)
            Generic_module_Twophase.two_phase_optimization(prob_w, x0_w, tp_test; history=false)
            Generic_module_unif.solve_unif_trust_region(prob_w, x0_w, unif_test)
        catch
        end
    finally
        isnothing(nlp_cutest) || finalize(nlp_cutest)
    end
end

"""
    ensure_parallel_workers(num_workers::Int)

Inicializa workers sob demanda se ainda não existirem, e sincroniza o ambiente neles.
Se os workers já estiverem ativos na sessão (ex: no REPL do VS Code), reutiliza-os instantaneamente.
"""
function ensure_parallel_workers(num_workers::Int)
    actual_workers = num_workers == 0 ? clamp(Sys.CPU_THREADS - 2, 1, 16) : max(1, num_workers)
    if actual_workers > 1
        needed = actual_workers - (nprocs() - 1)
        if needed > 0
            new_w = addprocs(needed)
            worker_file = normpath(joinpath(@__DIR__, "worker_env.jl"))
            for w in new_w
                remotecall_fetch(include, w, worker_file)
            end
        end
    end
    return actual_workers
end

"""
    stop_workers()

Encerra todos os workers distribuídos ativos e libera memória.
"""
function stop_workers()
    if nworkers() > 1
        rmprocs(workers())
        println("🛑 Workers encerrados. Sessão retornou a 1 processo.")
    else
        println("ℹ️ Nenhum worker ativo para encerrar.")
    end
end
