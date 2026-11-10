# ==============================================================================
# BENCHMARK EXTENSIVO CUTEst: TwoPhase vs UNIF (n <= 10, Igualdade ou Irrestritos)
# ==============================================================================

using Distributed
using Printf
using DataFrames
using CSV
using LinearAlgebra
using Statistics

const SRC_DIR = normpath(joinpath(@__DIR__, "..", "src"))
include(joinpath(SRC_DIR, "worker_env.jl"))

# Conjunto estendido e rigoroso de problemas CUTEst (n <= 10)
const EXTENDED_PROBLEMS = [
    # Problemas clássicos SMALL_PROBLEMS
    "HS6", "TRY-B", "HS28", "HS26", "HS35MOD", "EXTRASIM", "TAME",
    "ROSENBR", "HS3", "HS4", "HS45", "SISSER",
    # Igualdade com curvatura e não-linearidade acentuada (Biegler-Tjoa, Hock-Schittkowski)
    "BT1", "BT2", "BT3", "BT6", "BT7", "BT8", "BT9", "BT11",
    "HS7", "HS8", "HS9", "HS39", "HS40", "HS46", "HS47", "HS48",
    "BOOTH", "CUBENE", "MARATOS", "ZANGWIL3",
    # Irrestritos e com cotas desafiadores (Powell, Rosenbrock, Beale, etc.)
    "BEALE", "CUBE", "HELIX", "HS1", "HS2", "HS5",
    "ZANGWIL2", "MEXHAT", "HUMPS"
]

function run_extended_cutest_benchmark(; num_workers=4)
    actual_workers = ensure_parallel_workers(num_workers)
    is_parallel = actual_workers > 1

    println("\n" * "="^120)
    println("🚀 BENCHMARK EXTENSIVO CUTEst: TwoPhase vs UNIF ($(length(EXTENDED_PROBLEMS)) problemas, n <= 10)")
    println("⚙️  Configuração: M=1 estrito | Workers: $actual_workers | Critérios: Tempo e Compliance f(x*)")
    println("="^120)

    print("🔥 Aquecendo JIT... ")
    run_worker_warmup()
    if is_parallel
        @sync for w in workers()
            @async remotecall_fetch(run_worker_warmup, w)
        end
    end
    println("OK")

    print("🔧 Pré-compilando modelos CUTEst... ")
    valid_problems = String[]
    for p in EXTENDED_PROBLEMS
        try
            nlp = CUTEstModel{Float64}(p; decode=true)
            push!(valid_problems, p)
            finalize(nlp)
        catch err
            println("\n⚠️ Pulando $p (erro de decodificação: $err)")
        end
    end
    println("$(length(valid_problems)) problemas validados e prontos.")

    println("⚡ Executando $(length(valid_problems)) problemas em paralelo...")
    all_res = pmap(p -> solve_fast_benchmark_problem(p, 500; decode=false), valid_problems)

    df_rows = []
    for res_list in all_res
        for r in res_list
            push!(df_rows, (
                Problem = r.prob_name,
                Variant = r.name,
                Status = r.status_str,
                Iterations = r.iter_count,
                Time_ms = r.t_ms,
                f_final = r.f_val,
                h_norm = r.h_val,
                nvar = r.nvar,
                ncon = r.ncon
            ))
        end
    end

    df = DataFrame(df_rows)
    out_csv = normpath(joinpath(@__DIR__, "..", "benchmarks", "cutest_extended_results.csv"))
    CSV.write(out_csv, df)
    println("\n💾 Resultados detalhados salvos em: $out_csv")

    # ==============================================================================
    # ANÁLISE COMPARATIVA AGREGADA: TwoPhase vs UNIF
    # ==============================================================================
    println("\n" * "="^120)
    println("📊 RESUMO COMPARATIVO AGREGADO POR VARIANTE ($(length(valid_problems)) PROBLEMAS)")
    println("="^120)

    gdf = combine(groupby(df, :Variant),
        :Status => (s -> count(==( "KKT_OK"), s)) => :KKT_Solves,
        :Status => (s -> count(in(["KKT_OK", "SMALL_STEP"]), s)) => :Total_Success,
        :Iterations => mean => :Iters_Media,
        :Time_ms => mean => :Tempo_Medio_ms,
        :Time_ms => sum => :Tempo_Total_ms,
        :f_final => (f -> count(!isnan, f)) => :Evaluated
    )
    sort!(gdf, [order(:KKT_Solves, rev=true), order(:Tempo_Total_ms, rev=false)])

    println(
        rpad("Variante", 14), " | ",
        rpad("KKT Ok", 10), " | ",
        rpad("Taxa KKT", 10), " | ",
        rpad("Iters Média", 12), " | ",
        rpad("Tempo Médio (ms)", 18), " | ",
        rpad("Tempo Total (ms)", 18)
    )
    println("-"^120)
    n_probs = length(valid_problems)
    for r in eachrow(gdf)
        taxa = @sprintf("%.1f%%", (r.KKT_Solves / n_probs) * 100)
        @printf("%-14s | %-10s | %-10s | %-12.1f | %-18.2f | %-18.2f\n",
            r.Variant, "$(r.KKT_Solves)/$n_probs", taxa, r.Iters_Media, r.Tempo_Medio_ms, r.Tempo_Total_ms
        )
    end
    println("="^120)

    # ==============================================================================
    # HEAD-TO-HEAD: 2P_BASE vs UNIF_BASE e 2P_ISO vs UNIF_ISO
    # ==============================================================================
    println("\n" * "="^120)
    println("⚔️  COMPARAÇÃO DIRETA HEAD-TO-HEAD (CONFRONTO DIRETO)")
    println("="^120)

    for (tp_name, unif_name) in [("2P_BASE", "UNIF_BASE"), ("2P_ISO", "UNIF_ISO")]
        df_tp = filter(r -> r.Variant == tp_name, df)
        df_unif = filter(r -> r.Variant == unif_name, df)
        joined = innerjoin(df_tp, df_unif, on=:Problem, makeunique=true)

        tp_better_f = 0
        unif_better_f = 0
        tie_f = 0

        tp_faster = 0
        unif_faster = 0

        tp_fewer_iter = 0
        unif_fewer_iter = 0
        tie_iter = 0

        for row in eachrow(joined)
            # Comparação de compliance / f(x*)
            f_tp = row.f_final
            f_u = row.f_final_1
            if !isnan(f_tp) && !isnan(f_u)
                if abs(f_tp - f_u) <= 1e-5 * max(1.0, abs(f_u))
                    tie_f += 1
                elseif f_tp < f_u
                    tp_better_f += 1
                else
                    unif_better_f += 1
                end
            end

            # Comparação de tempo
            if row.Time_ms < row.Time_ms_1
                tp_faster += 1
            else
                unif_faster += 1
            end

            # Comparação de iterações
            if row.Iterations < row.Iterations_1
                tp_fewer_iter += 1
            elseif row.Iterations > row.Iterations_1
                unif_fewer_iter += 1
            else
                tie_iter += 1
            end
        end

        println("📌 CONFRONTO: $tp_name vs $unif_name")
        println("   • Menor/Melhor Compliance f(x*): $tp_name venceu $tp_better_f | $unif_name venceu $unif_better_f | Empates: $tie_f")
        println("   • Menor Tempo de Execução:       $tp_name venceu $tp_faster | $unif_name venceu $unif_faster")
        println("   • Menor Número de Iterações:     $tp_name venceu $tp_fewer_iter | $unif_name venceu $unif_fewer_iter | Empates: $tie_iter")
        println("-"^120)
    end

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_extended_cutest_benchmark(num_workers=4)
end
