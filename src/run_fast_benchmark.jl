# ==============================================================================
# RUNNER RÁPIDO DE BENCHMARK (FAST TEST BENCHMARK)
# ==============================================================================
# Executa rapidamente um conjunto selecionado de problemas pequenos (SMALL_PROBLEMS)
# comparando variantes do Two-Phase e do UNIF.
# Suporta execução sequencial (padrão) e paralela (-w / --workers N, onde 0 = auto).
# Funciona perfeitamente tanto via CLI (terminal) quanto no REPL do VS Code.
# Suporta gravação e comparação automática contra baseline numérico.
# ==============================================================================

using Distributed

# Carrega o ambiente no processo principal (Master)
include(normpath(joinpath(@__DIR__, "worker_env.jl")))

const DEFAULT_BASELINE_FILE = normpath(joinpath(@__DIR__, "..", "benchmarks", "fast_baseline.csv"))

"""
    run_fast_benchmark(; problems=SharedTypes.SMALL_PROBLEMS, max_iter=500, verbose=false, 
                         save_baseline::Bool=false, compare_baseline::Bool=false,
                         baseline_file=DEFAULT_BASELINE_FILE, num_workers::Int=1,
                         cleanup_workers::Bool=false)

Executa benchmark rápido na lista de problemas especificados.
Por padrão, `cleanup_workers=false` no REPL para manter os workers vivos e aquecidos
para próximas execuções ultrarrápidas sem overhead de inicialização.
"""
function run_fast_benchmark(; problems=SharedTypes.SMALL_PROBLEMS, max_iter::Int=500, verbose::Bool=false,
                             save_baseline::Bool=false, compare_baseline::Bool=false,
                             baseline_file::String=DEFAULT_BASELINE_FILE,
                             num_workers::Int=1,
                             cleanup_workers::Bool=false)
    
    actual_workers = ensure_parallel_workers(num_workers)
    is_parallel = actual_workers > 1

    println("\n" * "="^120)
    println("🚀 INICIANDO BENCHMARK RÁPIDO ($(length(problems)) problemas selecionados)")
    println("⚙️  Execução: $(is_parallel ? "Paralela ($actual_workers workers distribuídos)" : "Sequencial (1 processo)")")
    if save_baseline
        println("💾 MODO: Gravando baseline em $(basename(baseline_file))")
    elseif compare_baseline
        println("🔍 MODO: Comparando resultados contra baseline $(basename(baseline_file))")
    end
    println("="^120)

    # Aquecimento JIT no master e em todos os workers ativos
    print("🔥 Aquecendo JIT (Two-Phase e UNIF)... ")
    run_worker_warmup()
    if is_parallel
        @sync for w in workers()
            @async remotecall_fetch(run_worker_warmup, w)
        end
    end
    println("OK")

    df_results = DataFrame(
        Problem = String[],
        Variant = String[],
        Status = String[],
        Iterations = Int[],
        Time_ms = Float64[],
        f_final = Float64[],
        h_norm = Float64[]
    )

    if is_parallel
        # 1. Pré-compilação sequencial dos problemas CUTEst no master (evita conflitos de DLL no Windows)
        print("🔧 Pré-compilando $(length(problems)) problemas CUTEst no master... ")
        for p in problems
            try
                nlp = CUTEstModel{Float64}(p; decode=true)
                finalize(nlp)
            catch
            end
        end
        println("OK")

        # 2. Despacho paralelo balanceado via pmap
        println("⚡ Distribuindo problemas entre $(nworkers()) workers...")
        total_time_benchmark = @elapsed begin
            all_res = pmap(p -> solve_fast_benchmark_problem(p, max_iter; decode=false), problems)

            for res_list in all_res
                isempty(res_list) && continue
                first_r = res_list[1]
                println("\n" * "-"^120)
                @printf("📌 PROBLEMA: %-12s | Dim: n=%-2d, m=%-2d\n", first_r.prob_name, first_r.nvar, first_r.ncon)
                println("-"^120)
                println(
                    rpad("Variante", 14), " | ",
                    rpad("Status", 16), " | ",
                    rpad("Iter", 6), " | ",
                    rpad("Tempo (ms)", 12), " | ",
                    rpad("f(x*)", 14), " | ",
                    rpad("||h(x*)||", 12)
                )
                println("-"^120)

                for r in res_list
                    @printf("%-14s | %-16s | %-6d | %-12.2f | %-14.6e | %-12.2e\n",
                        r.name, r.status_str, r.iter_count, r.t_ms, r.f_val, r.h_val
                    )
                    push!(df_results, (r.prob_name, r.name, r.status_str, r.iter_count, r.t_ms, r.f_val, r.h_val))
                end
            end
        end

        if cleanup_workers
            rmprocs(workers())
        end
    else
        # Execução Sequencial
        total_time_benchmark = @elapsed begin
            for prob_name in problems
                res_list = solve_fast_benchmark_problem(prob_name, max_iter; decode=true)
                isempty(res_list) && continue
                first_r = res_list[1]
                println("\n" * "-"^120)
                @printf("📌 PROBLEMA: %-12s | Dim: n=%-2d, m=%-2d\n", first_r.prob_name, first_r.nvar, first_r.ncon)
                println("-"^120)
                println(
                    rpad("Variante", 14), " | ",
                    rpad("Status", 16), " | ",
                    rpad("Iter", 6), " | ",
                    rpad("Tempo (ms)", 12), " | ",
                    rpad("f(x*)", 14), " | ",
                    rpad("||h(x*)||", 12)
                )
                println("-"^120)

                for r in res_list
                    @printf("%-14s | %-16s | %-6d | %-12.2f | %-14.6e | %-12.2e\n",
                        r.name, r.status_str, r.iter_count, r.t_ms, r.f_val, r.h_val
                    )
                    push!(df_results, (r.prob_name, r.name, r.status_str, r.iter_count, r.t_ms, r.f_val, r.h_val))
                end
            end
        end
    end

    println("\n" * "="^120)
    @printf("🏁 BENCHMARK RÁPIDO CONCLUÍDO EM: %.2f segundos\n", total_time_benchmark)
    println("="^120)

    # Gravação do Baseline
    if save_baseline
        CSV.write(baseline_file, df_results)
        println("💾 Baseline gravado com sucesso em: $baseline_file ($(nrow(df_results)) execuções registradas)")
    end

    # Comparação rigorosa contra Baseline
    if compare_baseline
        if !isfile(baseline_file)
            println("⚠️ Arquivo de baseline não encontrado para comparação: $baseline_file")
            return df_results
        end

        df_base = CSV.read(baseline_file, DataFrame)
        df_cmp = innerjoin(df_results, df_base, on=[:Problem, :Variant], renamecols="" => "_base")

        total_checks = nrow(df_cmp)
        status_mismatches = 0
        iter_mismatches = 0
        f_mismatches = 0
        h_mismatches = 0

        time_curr_total = sum(df_cmp.Time_ms)
        time_base_total = sum(df_cmp.Time_ms_base)
        speedup_pct = (time_base_total - time_curr_total) / time_base_total * 100.0

        println("\n" * "="^120)
        println("🔬 RELATÓRIO DE VERIFICAÇÃO CONTRA O BASELINE ($(total_checks) casos)")
        println("="^120)

        for row in eachrow(df_cmp)
            # Status
            s_ok = (row.Status == row.Status_base)
            s_ok || (status_mismatches += 1)

            # Iterations
            it_ok = (row.Iterations == row.Iterations_base)
            it_ok || (iter_mismatches += 1)

            # f_final
            f_ok = (isnan(row.f_final) && isnan(row.f_final_base)) || 
                   isapprox(row.f_final, row.f_final_base, atol=1e-5, rtol=1e-4)
            f_ok || (f_mismatches += 1)

            # h_norm
            h_ok = (isnan(row.h_norm) && isnan(row.h_norm_base)) ||
                   isapprox(row.h_norm, row.h_norm_base, atol=1e-5, rtol=1e-4)
            h_ok || (h_mismatches += 1)

            if !s_ok || !it_ok || !f_ok || !h_ok
                println("⚠️ DIVERGÊNCIA DETECTADA em $(row.Problem) - $(row.Variant):")
                s_ok || println("   Status: $(row.Status_base) -> $(row.Status)")
                it_ok || println("   Iterações: $(row.Iterations_base) -> $(row.Iterations)")
                f_ok || println("   f(x*): $(row.f_final_base) -> $(row.f_final)")
                h_ok || println("   ||h(x*)||: $(row.h_norm_base) -> $(row.h_norm)")
            end
        end

        println("-"^120)
        if status_mismatches == 0 && iter_mismatches == 0 && f_mismatches == 0 && h_mismatches == 0
            println("✅ SUCESSO TOTAL: 100% DE CONFORMIDADE MATEMÁTICA!")
            println("   • Status: 0 divergências")
            println("   • Iterações: 0 divergências")
            println("   • f(x*): 0 divergências")
            println("   • ||h(x*)||: 0 divergências")
            @printf("   • Tempo somado das soluções: %.2f ms (novo) vs %.2f ms (base) => Diferença de %.2f%%\n",
                time_curr_total, time_base_total, speedup_pct)
        else
            println("❌ FALHA DE CONFORMIDADE:")
            println("   • Status divergentes: $status_mismatches")
            println("   • Iterações divergentes: $iter_mismatches")
            println("   • f(x*) divergentes: $f_mismatches")
            println("   • ||h(x*)|| divergentes: $h_mismatches")
        end
        println("="^120)
    end

    return df_results
end

# Execução automática se invocado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    save_flag = in("--save-baseline", ARGS)
    cmp_flag = in("--compare-baseline", ARGS) || (!save_flag && isfile(DEFAULT_BASELINE_FILE))
    
    workers_idx = findfirst(x -> x == "-w" || x == "--workers", ARGS)
    num_workers = 1
    if workers_idx !== nothing && workers_idx < length(ARGS)
        w_val = parse(Int, ARGS[workers_idx + 1])
        num_workers = w_val == 0 ? clamp(Sys.CPU_THREADS - 2, 1, 16) : max(1, w_val)
    end

    run_fast_benchmark(; save_baseline=save_flag, compare_baseline=cmp_flag, num_workers=num_workers)
end

# Execução interativa para o REPL do VS Code (descomente para executar direto com Shift+Enter / Ctrl+Enter)
run_fast_benchmark(num_workers=1, compare_baseline=true)
# stop_workers()