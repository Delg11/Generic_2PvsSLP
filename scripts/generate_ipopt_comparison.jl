# ==============================================================================
# SCRIPT DE COMPARAÇÃO DE PERFORMANCE: SLP (2P & UNIF) vs IPOPT
# ==============================================================================
# Gera perfis de desempenho de Dolan-Moré na interseção de problemas CUTEst
# entre a bateria noturna e o benchmark do IPOPT (ipopt_benchmark_fstar.csv).
# ==============================================================================

using CSV
using DataFrames
using Dates
ENV["GKSwstype"] = "100"
using Plots
using Statistics
using Printf

include(normpath(joinpath(@__DIR__, "..", "src", "Generic_module_Stats.jl")))
using .Generic_module_Stats

bench_dir = normpath(joinpath(@__DIR__, "..", "benchmarks", "Benchmark_Results_2026-10-31_23-10"))
tp_file = joinpath(bench_dir, "Consolidated_Twophase.csv")
un_file = joinpath(bench_dir, "Consolidated_UNIF.csv")
ipopt_file = normpath(joinpath(@__DIR__, "..", "ipopt_benchmark_fstar.csv"))

out_dir = joinpath(bench_dir, "Performance Profile Plots", "IPOPT_Comparison")
mkpath(out_dir)

println("\n" * "="^90)
println("📊 COMPARATIVO DE PERFORMANCE: SLP (2P / UNIF) vs IPOPT")
println("📁 Destino: $out_dir")
println("="^90)

df_tp = CSV.read(tp_file, DataFrame)
df_un = CSV.read(un_file, DataFrame)
df_ipopt_raw = CSV.read(ipopt_file, DataFrame)

probs_common = sort(intersect(unique(df_tp.Problem), unique(df_un.Problem), unique(df_ipopt_raw.Problem)))
n_probs = length(probs_common)
println("Total de problemas na interseção: $n_probs")

# Padronizar dataset do IPOPT
df_ipopt = DataFrame(
    Problem = df_ipopt_raw.Problem,
    Variant = fill("IPOPT", nrow(df_ipopt_raw)),
    Status = [s == "first_order" ? "KKT_OK" : (s == "max_iter" ? "MAX_IT" : (s == "small_step" ? "SMALL_STEP" : string(s))) for s in df_ipopt_raw.Status],
    Iterations = df_ipopt_raw.Iter,
    Time_ms = df_ipopt_raw.Time_s .* 1000.0,
    f_final = df_ipopt_raw.f_star,
    h_norm = df_ipopt_raw.h_norm
)

# Filtrar para a interseção exata
df_tp_sub = filter(r -> r.Problem in probs_common, df_tp)
df_un_sub = filter(r -> r.Problem in probs_common, df_un)
df_ipopt_sub = filter(r -> r.Problem in probs_common, df_ipopt)

df_all = vcat(
    select(df_tp_sub, :Problem, :Variant, :Status, :Iterations, :Time_ms, :f_final, :h_norm),
    select(df_un_sub, :Problem, :Variant, :Status, :Iterations, :Time_ms, :f_final, :h_norm),
    select(df_ipopt_sub, :Problem, :Variant, :Status, :Iterations, :Time_ms, :f_final, :h_norm)
)

# ------------------------------------------------------------------------------
# 1. Sumário Estatístico Comparativo
# ------------------------------------------------------------------------------
variants = unique(df_all.Variant)
summary_df = DataFrame(
    Variant = String[],
    Total = Int[],
    Solved_KKT = Int[],
    Success_Rate_KKT = Float64[],
    Best_f_Count = Int[],
    Best_f_Rate = Float64[],
    Mean_Iters_Solved = Float64[],
    Mean_Time_Solved_ms = Float64[]
)

f_tol = 1e-5
h_tol = 1e-5
infeasible_statuses = ["INFEASIBLE_STATIONARY", "STALLED_INFEASIBLE", "FAIL", "ERROR", "infeasible"]

best_f_map = Dict{String, Float64}()
for p in probs_common
    df_p = filter(r -> r.Problem == p, df_all)
    best_f = Inf
    for row in eachrow(df_p)
        st = row.Status
        h = row.h_norm
        f = row.f_final
        if !(st in infeasible_statuses) && (isnan(h) || h <= h_tol) && isfinite(f)
            best_f = min(best_f, f)
        end
    end
    best_f_map[p] = best_f
end

for v in variants
    df_v = filter(r -> r.Variant == v, df_all)
    tot = nrow(df_v)
    solved_kkt_df = filter(r -> r.Status == "KKT_OK" && (isnan(r.h_norm) || r.h_norm <= h_tol), df_v)
    solved_kkt = nrow(solved_kkt_df)
    rate_kkt = tot > 0 ? (solved_kkt / tot) * 100 : 0.0

    best_f_cnt = 0
    for row in eachrow(df_v)
        p = row.Problem
        target_f = get(best_f_map, p, Inf)
        if isfinite(target_f)
            st = row.Status
            h = row.h_norm
            f = row.f_final
            if !(st in infeasible_statuses) && (isnan(h) || h <= h_tol) && isfinite(f)
                if f <= target_f + f_tol * (1.0 + abs(target_f))
                    best_f_cnt += 1
                end
            end
        end
    end
    rate_best_f = tot > 0 ? (best_f_cnt / tot) * 100 : 0.0

    m_iters = solved_kkt > 0 ? mean(solved_kkt_df.Iterations) : Inf
    m_time = solved_kkt > 0 ? mean(solved_kkt_df.Time_ms) : Inf

    push!(summary_df, (v, tot, solved_kkt, rate_kkt, best_f_cnt, rate_best_f, m_iters, m_time))
end

sort!(summary_df, :Best_f_Count, rev=true)
summary_file = joinpath(out_dir, "IPOPT_vs_SLP_Summary.csv")
CSV.write(summary_file, summary_df)
println("✅ Sumário comparativo gravado em: $(basename(summary_file))")

# ------------------------------------------------------------------------------
# 2. Geração dos Perfis Dolan-Moré
# ------------------------------------------------------------------------------
# Paleta de cores consistente
cores = Dict{String, Any}(
    "IPOPT"                               => :black,
    "2P_SAR0_PH1_ATR1_RRU1_SECANT_BOX"   => :darkgreen,
    "UNIF_SAR0_PH1_ATR1_SECANT_BOX"      => :blue,
    "2P_SAR0_PH1_ATR0_RRU1"              => :orange,
    "UNIF_SAR0_PH1_ATR0"                 => :cyan,
    "2P_SAR0_PH1_ATR1_RRU1"              => :purple,
    "UNIF_SAR0_PH1_ATR1"                 => :magenta,
    "BASE_UNIF"                          => :gray,
    "2P_SAR1_PH0_ATR0_RRU1"              => :brown
)

# GRUPO A: Grand Finale (IPOPT vs Melhores Variantes de Cada Família)
g_main = [
    "IPOPT",
    "2P_SAR0_PH1_ATR1_RRU1_SECANT_BOX",
    "UNIF_SAR0_PH1_ATR1_SECANT_BOX",
    "2P_SAR0_PH1_ATR0_RRU1",
    "UNIF_SAR0_PH1_ATR0",
    "BASE_UNIF"
]

println("\n📈 Gerando Perfil Principal: IPOPT vs SLP (Melhor f)...")
Generic_module_Stats.plot_profile(
    df_all, g_main, :Time_ms,
    "IPOPT vs SLP: Melhores Variantes (Tempo) [Melhor f(x*)]",
    joinpath(out_dir, "Profile_IPOPT_vs_SLP_Best_f_Time.png"),
    cores;
    criterion = :best_f
)

Generic_module_Stats.plot_profile(
    df_all, g_main, :Iterations,
    "IPOPT vs SLP: Melhores Variantes (Iterações) [Melhor f(x*)]",
    joinpath(out_dir, "Profile_IPOPT_vs_SLP_Best_f_Iters.png"),
    cores;
    criterion = :best_f
)

println("📈 Gerando Perfil Principal: IPOPT vs SLP (Convergência KKT_OK)...")
Generic_module_Stats.plot_profile(
    df_all, g_main, :Time_ms,
    "IPOPT vs SLP: Melhores Variantes (Tempo) [KKT_OK]",
    joinpath(out_dir, "Profile_IPOPT_vs_SLP_KKT_Time.png"),
    cores;
    criterion = :kkt_ok
)

Generic_module_Stats.plot_profile(
    df_all, g_main, :Iterations,
    "IPOPT vs SLP: Melhores Variantes (Iterações) [KKT_OK]",
    joinpath(out_dir, "Profile_IPOPT_vs_SLP_KKT_Iters.png"),
    cores;
    criterion = :kkt_ok
)

# GRUPO B: Two-Phase vs IPOPT (Todas as Estratégias ATR)
g_2p = [
    "IPOPT",
    "2P_SAR0_PH1_ATR1_RRU1_SECANT_BOX",
    "2P_SAR0_PH1_ATR1_RRU1_DECOMPOSED_CUT",
    "2P_SAR0_PH1_ATR1_RRU1_GRAD_SIGN",
    "2P_SAR0_PH1_ATR1_RRU1",
    "2P_SAR0_PH1_ATR0_RRU1",
    "2P_SAR1_PH0_ATR0_RRU1"
]

println("📈 Gerando Perfil: Two-Phase ATR vs IPOPT...")
Generic_module_Stats.plot_profile(
    df_all, g_2p, :Time_ms,
    "Two-Phase ATR vs IPOPT (Tempo) [Melhor f(x*)]",
    joinpath(out_dir, "Profile_2P_ATR_vs_IPOPT_Best_f_Time.png"),
    cores;
    criterion = :best_f
)

Generic_module_Stats.plot_profile(
    df_all, g_2p, :Iterations,
    "Two-Phase ATR vs IPOPT (Iterações) [Melhor f(x*)]",
    joinpath(out_dir, "Profile_2P_ATR_vs_IPOPT_Best_f_Iters.png"),
    cores;
    criterion = :best_f
)

# GRUPO C: UNIF vs IPOPT (Todas as Estratégias ATR)
g_unif = [
    "IPOPT",
    "UNIF_SAR0_PH1_ATR1_SECANT_BOX",
    "UNIF_SAR0_PH1_ATR1_DECOMPOSED_CUT",
    "UNIF_SAR0_PH1_ATR1_GRAD_SIGN",
    "UNIF_SAR0_PH1_ATR1",
    "UNIF_SAR0_PH1_ATR0",
    "BASE_UNIF"
]

println("📈 Gerando Perfil: UNIF ATR vs IPOPT...")
Generic_module_Stats.plot_profile(
    df_all, g_unif, :Time_ms,
    "UNIF ATR vs IPOPT (Tempo) [Melhor f(x*)]",
    joinpath(out_dir, "Profile_UNIF_ATR_vs_IPOPT_Best_f_Time.png"),
    cores;
    criterion = :best_f
)

Generic_module_Stats.plot_profile(
    df_all, g_unif, :Iterations,
    "UNIF ATR vs IPOPT (Iterações) [Melhor f(x*)]",
    joinpath(out_dir, "Profile_UNIF_ATR_vs_IPOPT_Best_f_Iters.png"),
    cores;
    criterion = :best_f
)

println("\n" * "="^90)
println("🎉 GERAÇÃO DE PERFIS COMPARATIVOS COM IPOPT CONCLUÍDA!")
println("📁 Gráficos salvos em: $out_dir")
println("="^90)

