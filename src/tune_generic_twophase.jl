# ==============================================================================
# SCRIPT DE TUNING E COMPARAÇÃO: TwoPhase vs UNIF nos Problemas Genéricos (CUTEst)
# ==============================================================================

using Printf
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using CUTEst

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "worker_env.jl"))

const PROBLEMS_LIST = SharedTypes.SMALL_PROBLEMS

struct TuningCandidate
    name::String
    δ0_opt::Float64
    δmin::Float64
    tolS::Float64
    τ1::Float64
    τ2::Float64
    τ3::Float64
    αL::Float64
    sar::Bool
    ph::Bool
end

const CANDIDATES = [
    # Baseline original do TwoPhase (com bug τ3=2.0 corrigido para 0.5)
    TuningCandidate("2P_Default_Corr",   0.10, 1e-12, 1e-4, 0.10, 0.25, 0.50, 1e-8, true,  false),
    
    # C1: Piso delta_floor = 1e-4 + tolS = 1e-6 + expansão 2.0x (τ3=0.5) + τ2=0.5
    TuningCandidate("2P_C1_Floor1e4_2x", 0.10, 1e-4,  1e-6, 0.25, 0.50, 0.50, 1e-8, true,  false),
    
    # C2: Piso 1e-4 + tolS = 1e-6 + expansão 2.5x (τ3=0.40) + δ0 = 0.15
    TuningCandidate("2P_C2_Exp25_D015",  0.15, 1e-4,  1e-6, 0.25, 0.50, 0.40, 1e-8, true,  false),
    
    # C3: Expansão livre no primeiro aceite (SAR = false) + Piso 1e-4 + τ3=0.40
    TuningCandidate("2P_C3_NoSAR_Exp25", 0.15, 1e-4,  1e-6, 0.25, 0.50, 0.40, 1e-8, false, false),
    
    # C4: Heurística Parabólica (PH = true) + Piso 1e-4 + tolS = 1e-6 + δ0 = 0.15
    TuningCandidate("2P_C4_Parab_Floor", 0.15, 1e-4,  1e-6, 0.25, 0.50, 0.50, 1e-8, false, true),
]

function run_generic_comparison()
    println("="^110)
    println("🔬 COMPARANDO UNIF vs VARIANTES TWOPHASE EM $(length(PROBLEMS_LIST)) PROBLEMAS CUTEst")
    println("="^110)

    # 1. Executa UNIF Base e UNIF ISO como referências
    unif_base_params = SharedTypes.UNIFParams(delta0=0.1, tolG=1e-3, tolF=5e-2, tolS=1e-4, maxiter=500, verbose=false, strong_agreement_rule=true, parabolic_heuristic=false)
    unif_iso_params  = SharedTypes.UNIFParams(delta0=0.1, tolG=1e-3, tolF=5e-2, tolS=1e-4, maxiter=500, verbose=false, strong_agreement_rule=false, parabolic_heuristic=true)

    rows = NamedTuple[]

    for prob_name in PROBLEMS_LIST
        local nlp = nothing
        try
            nlp = CUTEstModel{Float64}(prob_name)
            prob = SharedTypes.build_optimization_problem(nlp)
            x0 = clamp.(nlp.meta.x0, prob.xl, prob.xu)

            # Executa UNIF Base
            t_u = @elapsed begin
                y_u, _, _, it_u, st_u, _, _ = Generic_module_unif.solve_unif_trust_region(prob, copy(x0), unif_base_params)
            end
            f_u = prob.f(y_u)
            h_u = norm(prob.h(y_u))
            kkt_u = (st_u == 0) && (h_u <= 1e-3)
            push!(rows, (Problem=prob_name, Solver="UNIF_BASE", Status=st_u, KKT=kkt_u, Iters=it_u, Time_ms=t_u*1000, f_val=f_u, h_val=h_u))

            # Executa UNIF ISO (com PH)
            t_ui = @elapsed begin
                y_ui, _, _, it_ui, st_ui, _, _ = Generic_module_unif.solve_unif_trust_region(prob, copy(x0), unif_iso_params)
            end
            f_ui = prob.f(y_ui)
            h_ui = norm(prob.h(y_ui))
            kkt_ui = (st_ui == 0) && (h_ui <= 1e-3)
            push!(rows, (Problem=prob_name, Solver="UNIF_ISO", Status=st_ui, KKT=kkt_ui, Iters=it_ui, Time_ms=t_ui*1000, f_val=f_ui, h_val=h_ui))

            # Executa Candidatos TwoPhase
            for c in CANDIDATES
                tp_params = SharedTypes.TwoPhaseParams(
                    δ0_opt=c.δ0_opt, δ0_resto=c.δ0_opt, δmin=c.δmin, tolS=c.tolS, tolG=1e-3, tolF=5e-2,
                    τ1=c.τ1, τ2=c.τ2, τ3=c.τ3, αL=c.αL,
                    strong_agreement_rule=c.sar, parabolic_heuristic=c.ph,
                    verbose_out=false, verbose_in=false
                )
                t_tp = @elapsed begin
                    y_tp, _, _, it_tp, st_tp, _, _ = Generic_module_Twophase.two_phase_optimization(prob, copy(x0), tp_params)
                end
                f_tp = prob.f(y_tp)
                h_tp = norm(prob.h(y_tp))
                kkt_tp = (st_tp == 0) && (h_tp <= 1e-3)
                push!(rows, (Problem=prob_name, Solver=c.name, Status=st_tp, KKT=kkt_tp, Iters=it_tp, Time_ms=t_tp*1000, f_val=f_tp, h_val=h_tp))
            end
        catch err
            println("❌ Erro em $prob_name: $err")
        finally
            isnothing(nlp) || finalize(nlp)
        end
    end

    df = DataFrame(rows)
    out_csv = joinpath(ROOT_DIR, "benchmarks", "tuning_generic_results.csv")
    CSV.write(out_csv, df)
    println("💾 Resultados salvos em: $out_csv")

    # Resumo agregado por Solver
    println("\n" * "="^110)
    println("📊 RESUMO AGREGADO POR SOLVER EM $(length(PROBLEMS_LIST)) PROBLEMAS")
    println("="^110)
    gdf = combine(groupby(df, :Solver),
        :KKT => sum => :Total_KKT,
        :Time_ms => mean => :Tempo_Medio_ms,
        :Time_ms => sum => :Tempo_Total_ms,
        :Iters => mean => :Iters_Medias,
        :f_val => mean => :F_Medio
    )
    sort!(gdf, [order(:Total_KKT, rev=true), order(:Tempo_Total_ms, rev=false)])

    println(rpad("Solver", 20), " | ", rpad("Total KKT", 10), " | ", rpad("Tempo Total (ms)", 18), " | ", rpad("Tempo Médio (ms)", 18), " | ", rpad("Iters Média", 12))
    println("-"^110)
    for r in eachrow(gdf)
        @printf("%-20s | %-10s | %-18.2f | %-18.2f | %-12.1f\n",
            r.Solver, "$(r.Total_KKT)/$(length(PROBLEMS_LIST))", r.Tempo_Total_ms, r.Tempo_Medio_ms, r.Iters_Medias
        )
    end
    println("="^110)

    return df, gdf
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_generic_comparison()
end
