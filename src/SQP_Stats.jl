# ==============================================================================
# Proprietary Software • All Rights Reserved
# ==============================================================================
module Generic_module_Stats

export run_statistical_analysis

using CSV
using Colors
using DataFrames
using Plots
using Statistics
using XLSX

# ==============================================================================
# 1. AUXILIARY METRIC FUNCTIONS (For XLSX Comparisons)
# ==============================================================================
function evaluate_m1(it_v, it_b, f_v, f_b, h_v, h_b, eps)
    if h_v > h_b + eps || f_v > f_b + eps return :Loss end
    if it_v < it_b return :Win elseif it_v > it_b return :Loss else return :Tie end
end

function evaluate_m2(f_v, f_b, h_v, h_b, eps)
    if h_v > h_b + eps return :Loss end
    if f_v < f_b - eps return :Win elseif f_v > f_b + eps return :Loss else return :Tie end
end

function evaluate_m3(t_v, t_b, f_v, f_b, h_v, h_b, eps)
    if h_v > h_b + eps || f_v > f_b + eps return :Loss end
    if t_v < t_b return :Win elseif t_v > t_b return :Loss else return :Tie end
end

# ==============================================================================
# 1b. SAR/PH CONFLICT RESOLUTION
# ==============================================================================
"""
Quando PH=1, a variação de SAR (0 ou 1) não produz diferença relevante nos
resultados (a diferença observada é de milésimos de segundo, dentro do ruído
de medição). Manter as duas variantes (PH1_SAR0 e PH1_SAR1) duplicava
observações essencialmente idênticas e distorcia as estatísticas agregadas.

Esta função descarta a variante PH1_SAR0 sempre que existir, com todos os
demais parâmetros idênticos, a variante irmã PH1_SAR1 — tanto para a família
UNIF quanto para a família 2P (2-Fases).
"""
function remove_redundant_sar_ph_variants(df::DataFrame)
    variants = unique(df.Variant)
    to_drop = String[]

    for v in variants
        if occursin("_SAR0_PH1", v)
            sibling = replace(v, "_SAR0_PH1" => "_SAR1_PH1")
            if sibling in variants
                push!(to_drop, v)
            end
        end
    end

    if !isempty(to_drop)
        println("⚠️  SAR e PH são conflitantes quando PH=1 (diferença de milésimos de segundo).")
        println("    Descartando variantes redundantes PH1_SAR0 (mantendo PH1_SAR1): ", join(to_drop, ", "))
        return filter(r -> !(r.Variant in to_drop), df)
    end
    return df
end

# ==============================================================================
# 2. MAIN STATISTICAL ANALYSIS ROUTINE
# ==============================================================================
"""
Main entry point for generating statistics, performance profiles, and XLSX reports.
"""
function run_statistical_analysis(results_dir::String)
    println("\n" * "=" ^ 80)
    println("📊 STARTING CONSOLIDATED ANALYSIS AND STATISTICS")
    println("=" ^ 80)

    unif_file = joinpath(results_dir, "partial_backup_unif.csv")
    tp_file   = joinpath(results_dir, "partial_backup_twophase.csv")

    println("Loading databases...")
    df_unif = isfile(unif_file) ? CSV.read(unif_file, DataFrame) : DataFrame()
    df_tp   = isfile(tp_file)   ? CSV.read(tp_file, DataFrame)   : DataFrame()

    if nrow(df_unif) == 0 && nrow(df_tp) == 0
        println("⚠️ No data found to process. Exiting analysis.")
        return
    end

    df_all = vcat(df_unif, df_tp, cols=:union)
    # filter!(r -> !occursin("SQP", r.Variant), df_all)
    df_all = remove_redundant_sar_ph_variants(df_all)

    # --------------------------------------------------------------------------
    # PART A: TOURNAMENT PROFILES
    # --------------------------------------------------------------------------
    summary_df = generate_summary_table(df_all)
    CSV.write(joinpath(results_dir, "Summary_Statistics_chap3.csv"), summary_df)
    println("✅ Summary statistics (CSV) saved.")

    println("📈 Generating Dolan-Moré Performance Profiles...")
    generate_tournament_profiles(df_all, summary_df, results_dir)
    plot_outcome_distribution(df_all, summary_df, results_dir)
    # --------------------------------------------------------------------------
    # PART B: METRICS AND XLSX EXPORT
    # --------------------------------------------------------------------------
    println("📊 Calculating mathematical metrics for XLSX export...")
    
    # Filter invalid results (NaN or Inf) for the XLSX metrics
    invalid_mask = isnan.(df_all.f_final) .| isinf.(df_all.f_final) .| isnan.(df_all.h_norm) .| isinf.(df_all.h_norm)
    df_valid = df_all[.!invalid_mask, :]
    plot_solution_quality_histogram(df_valid, summary_df, results_dir)
    all_variants  = unique(df_valid.Variant)
    unif_variants = filter(v -> startswith(v, "UNIF") || v == "BASE_UNIF", all_variants)
    tp_variants   = filter(v -> startswith(v, "2P")   || v == "BASE_2P", all_variants)

    # SCENARIO TABLES
    df_Scenario_A = df_valid
    df_Scenario_B = filter(r -> !ismissing(r.Iterations) && r.Iterations > 0 && r.Status != "MAX_IT", df_valid)
    df_Scenario_C = filter(r -> !ismissing(r.Status) && r.Status == "KKT_OK", df_valid)

    function calculate_statistics(df_scenario)
        if nrow(df_scenario) == 0 return DataFrame() end
        combine(groupby(df_scenario, :Variant),
            :Iterations => (x -> mean(skipmissing(x))) => :Mean_Iterations,
            :Iterations => (x -> median(skipmissing(x))) => :Median_Iterations,
            :Time_ms    => (x -> mean(skipmissing(x))) => :Mean_Time_ms,
            :f_final    => (x -> mean(skipmissing(x))) => :Mean_f_final,
            :f_final    => (x -> median(skipmissing(x))) => :Median_f_final,
            :h_norm     => (x -> mean(skipmissing(x))) => :Mean_h_norm,
            :h_norm     => (x -> median(skipmissing(x))) => :Median_h_norm,
            nrow => :Num_Problems
        )
    end

    stats_A = calculate_statistics(df_Scenario_A)
    stats_B = calculate_statistics(df_Scenario_B)
    stats_C = calculate_statistics(df_Scenario_C)

    eps = 1e-3

    # ==========================================================================
    # INTRA-METHOD COMPARISON (SQP IDENTITY vs Outras Hessianas da mesma geometria)
    # ==========================================================================
    intra_pairs = []
    
    # Identifica todas as variantes que usam IDENTITY e as define como base
    for v_base in all_variants
        if endswith(v_base, "_SQP_IDENTITY")
            prefix = replace(v_base, "_SQP_IDENTITY" => "")
            
            # Procura outras variantes que tenham a mesma geometria (mesmo prefixo)
            for v_other in all_variants
                if startswith(v_other, prefix) && v_other != v_base && occursin("SQP", v_other)
                    push!(intra_pairs, (v_base, v_other))
                end
            end
        end
    end

    intra_results = []
    for (base_name, var_name) in intra_pairs
        df_base = filter(r -> r.Variant == base_name, df_valid)
        df_var  = filter(r -> r.Variant == var_name, df_valid)
        
        if nrow(df_base) == 0 || nrow(df_var) == 0 continue end
        
        df_join = innerjoin(df_var, df_base, on=:Problem, makeunique=true)
        if nrow(df_join) == 0 continue end

        m1_w, m1_t, m1_l, m2_w, m2_t, m2_l, m3_w, m3_t, m3_l = zeros(Int, 9)
        for r in eachrow(df_join)
            res_m1 = evaluate_m1(r.Iterations, r.Iterations_1, r.f_final, r.f_final_1, r.h_norm, r.h_norm_1, eps)
            res_m1 == :Win ? m1_w += 1 : (res_m1 == :Tie ? m1_t += 1 : m1_l += 1)
            res_m2 = evaluate_m2(r.f_final, r.f_final_1, r.h_norm, r.h_norm_1, eps)
            res_m2 == :Win ? m2_w += 1 : (res_m2 == :Tie ? m2_t += 1 : m2_l += 1)
            res_m3 = evaluate_m3(r.Time_ms, r.Time_ms_1, r.f_final, r.f_final_1, r.h_norm, r.h_norm_1, eps)
            res_m3 == :Win ? m3_w += 1 : (res_m3 == :Tie ? m3_t += 1 : m3_l += 1)
        end
        push!(intra_results, (
            Base=base_name, 
            Variant=var_name, 
            Total=nrow(df_join), 
            M1_W=m1_w, M1_T=m1_t, M1_L=m1_l, 
            M2_W=m2_w, M2_T=m2_t, M2_L=m2_l, 
            M3_W=m3_w, M3_T=m3_t, M3_L=m3_l
        ))
    end
    df_res_intra = DataFrame(intra_results)

    # INTER-METHOD COMPARISON (Best 2P vs Best UNIF)
    m1_w, m1_t, m1_l, m2_w, m2_t, m2_l, m3_w, m3_t, m3_l, total_inter = zeros(Int, 10)
    inter_details = []

    for prob in unique(df_valid.Problem)
        df_prob = filter(r -> r.Problem == prob, df_valid)
        unifs_valid = filter(r -> r.Variant in unif_variants && r.h_norm <= eps, df_prob)
        tps_valid   = filter(r -> r.Variant in tp_variants && r.h_norm <= eps, df_prob)
        
        if nrow(unifs_valid) > 0 && nrow(tps_valid) > 0
            best_unif = sort(unifs_valid, [:f_final, :Iterations])[1, :]
            best_2p   = sort(tps_valid,   [:f_final, :Iterations])[1, :]
            total_inter += 1
            
            res_m1 = evaluate_m1(best_2p.Iterations, best_unif.Iterations, best_2p.f_final, best_unif.f_final, best_2p.h_norm, best_unif.h_norm, eps)
            res_m1 == :Win ? m1_w += 1 : (res_m1 == :Tie ? m1_t += 1 : m1_l += 1)
            res_m2 = evaluate_m2(best_2p.f_final, best_unif.f_final, best_2p.h_norm, best_unif.h_norm, eps)
            res_m2 == :Win ? m2_w += 1 : (res_m2 == :Tie ? m2_t += 1 : m2_l += 1)
            res_m3 = evaluate_m3(best_2p.Time_ms, best_unif.Time_ms, best_2p.f_final, best_unif.f_final, best_2p.h_norm, best_unif.h_norm, eps)
            res_m3 == :Win ? m3_w += 1 : (res_m3 == :Tie ? m3_t += 1 : m3_l += 1)

            push!(inter_details, (Problem = prob, Best_UNIF = best_unif.Variant, UNIF_f_final = best_unif.f_final, UNIF_Iterations = best_unif.Iterations, Best_2P = best_2p.Variant, f_final_2P = best_2p.f_final, Iterations_2P = best_2p.Iterations, Result_M1_Iters = string(res_m1), Result_M2_F_Final = string(res_m2), Result_M3_Time = string(res_m3)))
        end
    end

    df_res_inter = DataFrame(Comparison=["Best 2P vs Best UNIF"], Total=[total_inter], M1_W=[m1_w], M1_T=[m1_t], M1_L=[m1_l], M2_W=[m2_w], M2_T=[m2_t], M2_L=[m2_l], M3_W=[m3_w], M3_T=[m3_t], M3_L=[m3_l])
    df_inter_details = DataFrame(inter_details)

    # XLSX EXPORT
    output_file = joinpath(results_dir, "Consolidated_Statistics.xlsx")
    XLSX.openxlsx(output_file, mode="w") do xf
        XLSX.rename!(xf[1], "Scenario_A")
        if nrow(stats_A) > 0 XLSX.writetable!(xf[1], collect(eachcol(stats_A)), names(stats_A)) end
        
        XLSX.addsheet!(xf, "Scenario_B")
        if nrow(stats_B) > 0 XLSX.writetable!(xf[2], collect(eachcol(stats_B)), names(stats_B)) end
        
        XLSX.addsheet!(xf, "Scenario_C")
        if nrow(stats_C) > 0 XLSX.writetable!(xf[3], collect(eachcol(stats_C)), names(stats_C)) end
        
        XLSX.addsheet!(xf, "Intra_Comparison")
        if nrow(df_res_intra) > 0 XLSX.writetable!(xf[4], collect(eachcol(df_res_intra)), names(df_res_intra)) end
        
        XLSX.addsheet!(xf, "Inter_Comparison")
        if nrow(df_res_inter) > 0 XLSX.writetable!(xf[5], collect(eachcol(df_res_inter)), names(df_res_inter)) end
        
        XLSX.addsheet!(xf, "Inter_Details")
        if nrow(df_inter_details) > 0 XLSX.writetable!(xf[6], collect(eachcol(df_inter_details)), names(df_inter_details)) end
    end

    println("✅ Execution and Analysis complete. Results, Plots, and XLSX saved to: $results_dir")
end

# ==============================================================================
# 3. PERFORMANCE PROFILES LOGIC (DOLAN-MORÉ TOURNAMENTS)
# ==============================================================================
function generate_summary_table(df::DataFrame)
    variants = unique(df.Variant)
    
    results = DataFrame(
        Variant = String[],
        Total_Problems = Int[],
        Solved_KKT = Int[],
        Success_Rate = Float64[],
        Mean_Time_ms = Float64[],
        Mean_Iterations = Float64[]
    )

    for v in variants
        sub_df = df[df.Variant .== v, :]
        total = nrow(sub_df)
        
        solved_df = sub_df[sub_df.Status .== "KKT_OK", :]
        solved = nrow(solved_df)
        rate = total > 0 ? (solved / total) * 100 : 0.0
        
        m_time = solved > 0 ? mean(solved_df.Time_ms) : Inf
        m_iters = solved > 0 ? mean(solved_df.Iterations) : Inf
        
        push!(results, (v, total, solved, rate, m_time, m_iters))
    end

    sort!(results, [:Solved_KKT, :Mean_Time_ms], rev=[true, false])
    return results
end

function generate_tournament_profiles(df::DataFrame, summary::DataFrame, out_dir::String)
    all_variants = unique(df.Variant)
    get_existing(wanted::Vector{String}) = intersect(wanted, all_variants)

    color_map = build_variant_color_map(all_variants)

    plots_dir = joinpath(out_dir, "Plots")
    mkpath(plots_dir)

    # ---------------------------------------------------------
    # GROUP 1: UNIF - Hessian Approximations
    # ---------------------------------------------------------
    g1_base_unif = "UNIF_SAR1_PH0_ATR0_SQP_"
    g1 = get_existing([
        g1_base_unif * "IDENTITY",
        g1_base_unif * "SPECTRAL",
        g1_base_unif * "RECIPROCAL",
        g1_base_unif * "EXPONENTIAL",
        g1_base_unif * "QUASI_NEWTON"
    ])
    plot_profile(df, g1, :Time_ms, "Group 1: UNIF - Hessian Approximations", joinpath(plots_dir, "Profile_G1_UNIF_Hessians.png"))

    # ---------------------------------------------------------
    # GROUP 2: 2P - Hessian Approximations
    # ---------------------------------------------------------
    g2_base_2p = "2P_SAR0_PH1_ATR0_RRU1_SQP_"
    g2 = get_existing([
        g2_base_2p * "IDENTITY",
        g2_base_2p * "SPECTRAL",
        g2_base_2p * "RECIPROCAL",
        g2_base_2p * "EXPONENTIAL",
        g2_base_2p * "QUASI_NEWTON"
    ])
    plot_profile(df, g2, :Time_ms, "Group 2: 2P - Hessian Approximations", joinpath(plots_dir, "Profile_G2_2P_Hessians.png"))

    # ---------------------------------------------------------
    # GROUP 3: UNIF - Geometry Impact (Fixing best Hessian)
    # ---------------------------------------------------------
    best_unif_hessian_full = get_best_in_class(g1, summary)
    if !isnothing(best_unif_hessian_full)
        best_hessian_unif = replace(best_unif_hessian_full, g1_base_unif => "")
        g3 = get_existing([
            "UNIF_SAR0_PH0_ATR0_SQP_" * best_hessian_unif,
            "UNIF_SAR0_PH1_ATR0_SQP_" * best_hessian_unif,
            "UNIF_SAR0_PH1_ATR1_SQP_" * best_hessian_unif,
            "UNIF_SAR1_PH0_ATR0_SQP_" * best_hessian_unif 
        ])
        plot_profile(df, g3, :Time_ms, "Group 3: UNIF - Geometry Impact ($best_hessian_unif)", joinpath(plots_dir, "Profile_G3_UNIF_Geometry.png"))
    end

    # ---------------------------------------------------------
    # GROUP 4: 2P - Geometry Impact (Fixing best Hessian)
    # ---------------------------------------------------------
    best_2p_hessian_full = get_best_in_class(g2, summary)
    if !isnothing(best_2p_hessian_full)
        best_hessian_2p = replace(best_2p_hessian_full, g2_base_2p => "")
        g4 = get_existing([
            "2P_SAR0_PH0_ATR0_RRU1_SQP_" * best_hessian_2p,
            "2P_SAR0_PH1_ATR0_RRU1_SQP_" * best_hessian_2p,
            "2P_SAR0_PH1_ATR1_RRU1_SQP_" * best_hessian_2p,
            "2P_SAR1_PH0_ATR0_RRU1_SQP_" * best_hessian_2p 
        ])
        plot_profile(df, g4, :Time_ms, "Group 4: 2P - Geometry Impact ($best_hessian_2p)", joinpath(plots_dir, "Profile_G4_2P_Geometry.png"))
    end

    # ---------------------------------------------------------
    # GROUP 5: Overall Robustness (Best UNIF vs Best 2P)
    # ---------------------------------------------------------
    unif_sqp = filter(v -> occursin("UNIF", v) && occursin("SQP", v), all_variants)
    tp_sqp   = filter(v -> occursin("2P", v)   && occursin("SQP", v), all_variants)

    best_unif_sqp = get_best_in_class(unif_sqp, summary)
    best_tp_sqp   = get_best_in_class(tp_sqp, summary)

    g5 = String[]
    isnothing(best_unif_sqp) || push!(g5, best_unif_sqp)
    isnothing(best_tp_sqp)   || push!(g5, best_tp_sqp)

    plot_profile(df, g5, :Time_ms, "Group 5: Overall Robustness (Best UNIF vs Best 2P)", joinpath(plots_dir, "Profile_G5_Overall_Robustness.png"))
end
"""
Paleta de cores vívidas e saturadas, escolhida à mão para manter o mesmo
estilo usado no capítulo (ex.: azul #009CE8, terracota #D8724C, verde
#4EA553), com bom contraste sobre fundo branco — sem tons pastéis.

Se houver mais variantes do que cores na paleta, cores extras são geradas
automaticamente seguindo a mesma faixa de luminosidade/saturação, evitando
repetir tons parecidos com os já usados na paleta fixa. O mapa é calculado
uma única vez a partir de TODAS as variantes presentes nos dados e repassado
a todas as chamadas de `plot_profile`, então cada variante mantém sempre a
mesma cor entre os diferentes gráficos. A ordenação alfabética garante que o
resultado seja determinístico entre execuções.
"""
const VIVID_PALETTE = [
    colorant"#009CE8",  # 1. Light Blue
    colorant"#D8724C",  # 2. Terracotta / Light Orange
    colorant"#4EA553",  # 3. Green
    colorant"#8E5FBF",  # 4. Purple
    colorant"#D64550",  # 5. Red
    colorant"#D9A441",  # 6. Mustard Yellow
    colorant"#0F8C8A",  # 7. Dark Turquoise
    colorant"#C72C73",  # 8. Strong Magenta
    colorant"#295CA3",  # 9. Cobalt Blue
    colorant"#965C19",  # 10. Coppery Brown
    colorant"#6B8224",  # 11. Vibrant Moss Green
    colorant"#91204D",  # 12. Burgundy / Dark Wine
    colorant"#D16215",  # 13. Burnt Orange
    colorant"#57398C",  # 14. Indigo
    colorant"#1A7A54",  # 15. Dark Emerald Green
]
function build_variant_color_map(variants::Vector{String})
    sorted_variants = sort(unique(variants))
    n = length(sorted_variants)
    n == 0 && return Dict{String,Any}()

    if n <= length(VIVID_PALETTE)
        colors = VIVID_PALETTE[1:n]
    else
        extra_needed = n - length(VIVID_PALETTE)
        extra_colors = distinguishable_colors(
            extra_needed,
            vcat([RGB(1, 1, 1), RGB(0, 0, 0)], VIVID_PALETTE);
            dropseed  = true,
            lchoices  = range(15, stop=65, length=12),
            cchoices  = range(60, stop=100, length=8),
            hchoices  = range(0, stop=340, length=20)
        )
        colors = vcat(VIVID_PALETTE, extra_colors)
    end

    return Dict(sorted_variants[i] => colors[i] for i in 1:n)
end

function get_best_in_class(class_variants::Vector{String}, summary::DataFrame)
    if isempty(class_variants) return nothing end
    sub = summary[in.(summary.Variant, Ref(class_variants)), :]
    if nrow(sub) == 0 return nothing end
    return sub[1, :Variant]
end

"""
Builds a Dolan-Moré performance profile (step plot) for a specific metric and subset of solvers.

`color_map` (opcional) associa cada nome de variante a uma cor fixa, para que a
mesma variante mantenha sempre a mesma cor em todos os gráficos gerados. Se
omitido, cai de volta no ciclo automático de cores do Plots.jl.
"""
# function plot_profile(df::DataFrame, solvers::Vector{String}, metric_col::Symbol, title_str::String, filename::String, color_map::AbstractDict=Dict{String,Any}())
#     if length(solvers) < 2
#         println("  > Skipping '$title_str': Not enough valid solvers in this group.")
#         return
#     end

#     problems = unique(df.Problem)
#     n_probs = length(problems)
#     n_solvers = length(solvers)

#     perf_matrix = fill(Inf, n_probs, n_solvers)

#     for (j, s) in enumerate(solvers)
#         for (i, p) in enumerate(problems)
#             row = df[(df.Variant .== s) .& (df.Problem .== p), :]
#             if nrow(row) > 0 && row[1, :Status] == "KKT_OK"
#                 val = Float64(row[1, metric_col])
#                 perf_matrix[i, j] = val <= 0.0 ? 1e-8 : val
#             end
#         end
#     end

#     min_vals = minimum(perf_matrix, dims=2)
#     ratios = perf_matrix ./ min_vals
#     ratios[isnan.(ratios)] .= Inf

#     valid_ratios = filter(x -> !isinf(x), ratios)
#     max_tau = isempty(valid_ratios) ? 10.0 : maximum(valid_ratios)
#     plot_max = max(10.0, max_tau * 1.1)

#     p = plot(title=title_str, 
#              xlabel="Performance Ratio (τ)", 
#              ylabel="Fraction of Solved Problems", 
#              legend=:bottomright, 
#              xscale=:log10, 
#              framestyle=:box,
#              dpi=300,
#              size=(800, 600))

#     for (j, s) in enumerate(solvers)
#         sorted_r = sort(ratios[:, j])
#         filter!(x -> !isinf(x), sorted_r)
        
#         n_solved = length(sorted_r)
        
#         x_vals = [1.0]
#         y_vals = [0.0]
        
#         for (k, r) in enumerate(sorted_r)
#             push!(x_vals, r)
#             push!(y_vals, k / n_probs)
#         end
        
#         push!(x_vals, plot_max)
#         push!(y_vals, n_solved / n_probs)

#         clean_label = replace(s, "UNIF_" => "UNIF ", "2P_" => "2P ", "BASE_UNIF" => "UNIF SAR1_PH0_ATR0", "BASE_2P" => "2P SAR1_PH0_ATR0_RRU0")

#         line_color = get(color_map, s, nothing)
#         if line_color === nothing
#             plot!(p, x_vals, y_vals, linetype=:steppost, label=clean_label, linewidth=2.5)
#         else
#             plot!(p, x_vals, y_vals, linetype=:steppost, label=clean_label, linewidth=2.5, color=line_color)
#         end
#     end

#     savefig(p, filename)
#     println("  > Saved: $(basename(filename))")
# end
function plot_profile(df::DataFrame, solvers::Vector{String}, metric_col::Symbol, title_str::String, filename::String, color_map::AbstractDict=Dict{String,Any}())
    if length(solvers) < 2
        println("  > Skipping '$title_str': Not enough valid solvers in this group.")
        return
    end

    problems = unique(df.Problem)
    n_probs = length(problems)
    n_solvers = length(solvers)

    perf_matrix = fill(Inf, n_probs, n_solvers)

    infeasible_statuses = ["INFEASIBLE_STATIONARY", "STALLED_INFEASIBLE"]
    f_tol = 1e-5 # Tolerância numérica para considerar empate na função objetivo
    h_tol = 1e-5 # Tolerância máxima de violação para atestar viabilidade

    for (i, p) in enumerate(problems)
        df_p = df[df.Problem .== p, :]
        
        # Etapa 1: Encontrar a melhor função objetivo (F_min) entre os solvers viáveis
        best_f = Inf
        for s in solvers
            row = df_p[df_p.Variant .== s, :]
            if nrow(row) > 0
                status = row[1, :Status]
                h_val = Float64(row[1, :h_norm])
                
                if !(status in infeasible_statuses) && h_val <= h_tol
                    f_val = Float64(row[1, :f_final])
                    best_f = min(best_f, f_val)
                end
            end
        end

        isinf(best_f) && continue

        # Etapa 2: Preencher a matriz de performance para os que atingiram F_min
        for (j, s) in enumerate(solvers)
            row = df_p[df_p.Variant .== s, :]
            if nrow(row) > 0
                status = row[1, :Status]
                h_val = Float64(row[1, :h_norm])
                
                if !(status in infeasible_statuses) && h_val <= h_tol
                    f_val = Float64(row[1, :f_final])
                    
                    if f_val <= best_f + f_tol + f_tol * abs(best_f)
                        val = Float64(row[1, metric_col])
                        perf_matrix[i, j] = val <= 0.0 ? 1e-8 : val
                    end
                end
            end
        end
    end

    min_vals = minimum(perf_matrix, dims=2)
    ratios = perf_matrix ./ min_vals
    ratios[isnan.(ratios)] .= Inf

    valid_ratios = filter(x -> !isinf(x), ratios)
    max_tau = isempty(valid_ratios) ? 10.0 : maximum(valid_ratios)
    plot_max = max(10.0, max_tau * 1.1)

    p = plot(title=title_str, 
             xlabel="Performance Ratio (τ)", 
             ylabel="Fraction of Solved Problems", 
             legend=:bottomright, 
             xscale=:log10, 
             framestyle=:box,
             dpi=300,
             size=(800, 600))

    for (j, s) in enumerate(solvers)
        sorted_r = sort(ratios[:, j])
        filter!(x -> !isinf(x), sorted_r)
        
        n_solved = length(sorted_r)
        
        x_vals = [1.0]
        y_vals = [0.0]
        
        for (k, r) in enumerate(sorted_r)
            push!(x_vals, r)
            push!(y_vals, k / n_probs)
        end
        
        push!(x_vals, plot_max)
        push!(y_vals, n_solved / n_probs)

        clean_label = replace(s, "UNIF_" => "UNIF ", "2P_" => "2P ", "BASE_UNIF" => "UNIF SAR1_PH0_ATR0", "BASE_2P" => "2P SAR1_PH0_ATR0_RRU0")

        line_color = get(color_map, s, nothing)
        if line_color === nothing
            plot!(p, x_vals, y_vals, linetype=:steppost, label=clean_label, linewidth=2.5)
        else
            plot!(p, x_vals, y_vals, linetype=:steppost, label=clean_label, linewidth=2.5, color=line_color)
        end
    end

    savefig(p, filename)
    println("  > Saved: $(basename(filename))")
end

function plot_outcome_distribution(df::DataFrame, summary::DataFrame, out_dir::String)
    println("📈 Generating Outcome Distribution (2P-QN vs UNIF-QN)...")
    
    all_variants = unique(df.Variant)
    # Filtra exclusivamente as variantes Quasi-Newton
    unif_qn = filter(v -> startswith(v, "UNIF") && occursin("QUASI_NEWTON", v), all_variants)
    tp_qn   = filter(v -> startswith(v, "2P")   && occursin("QUASI_NEWTON", v), all_variants)
    
    best_unif_var = get_best_in_class(unif_qn, summary)
    best_2p_var   = get_best_in_class(tp_qn, summary)

    if isnothing(best_unif_var) || isnothing(best_2p_var)
        println("⚠️ Not enough data to compare 2P-QN vs UNIF-QN.")
        return
    end

    v_2p_absoluto = 0
    v_2p_gap      = 0
    empate        = 0
    v_unif_gap    = 0
    v_unif_absoluto = 0

    function get_result(v_name, p_name)
        rows = df[(df.Variant .== v_name) .& (df.Problem .== p_name), :]
        if nrow(rows) > 0
            st = rows[1, :Status]
            f = rows[1, :f_final]
            hn = rows[1, :h_norm]
            is_ok = st == "KKT_OK" && hn <= 1e-3
            return is_ok, f
        end
        return false, nothing
    end

    for p in unique(df.Problem)
        ok_2p, f_2p     = get_result(best_2p_var, p)
        ok_unif, f_unif = get_result(best_unif_var, p)

        if ok_2p && !ok_unif
            v_2p_absoluto += 1
        elseif !ok_2p && ok_unif
            v_unif_absoluto += 1
        elseif ok_2p && ok_unif
            gap = (f_2p - f_unif) / max(1e-8, abs(f_unif))
            
            if gap < -1e-3
                v_2p_gap += 1
            elseif gap > 1e-3
                v_unif_gap += 1
            else
                empate += 1
            end
        end
    end

    # Atualizado para refletir QN
    categories = [
        "2P-QN Wins\n(UNIF Failed)", 
        "2P-QN Wins\n(Better f)", 
        "Tie\n(|Gap| <= 1e-3)", 
        "UNIF-QN Wins\n(Better f)", 
        "UNIF-QN Wins\n(2P Failed)"
    ]
    counts = [v_2p_absoluto, v_2p_gap, empate, v_unif_gap, v_unif_absoluto]
    max_c = maximum(counts)
    
    colors = [colorant"#1A7A54", colorant"#4EA553", colorant"#999999", colorant"#009CE8", colorant"#295CA3"]

    p = bar(categories, counts, 
            color=colors, 
            legend=false, 
            title="Outcome Distribution: 2P-QN vs UNIF-QN",
            ylabel="Number of Problems",
            framestyle=:box,
            ylims=(0, max_c * 1.15), # 15% de margem no topo para caber o texto
            dpi=300,
            size=(800, 500))
    
    for (i, c) in enumerate(counts)
        annotate!(p, categories[i], c + (max_c * 0.015), text(string(c), 11, :center, :bottom))
    end

    savefig(p, joinpath(out_dir, "Plots", "Outcome_Distribution_2P_vs_UNIF_QN.png"))
end

function plot_solution_quality_histogram(df::DataFrame, summary::DataFrame, out_dir::String)
    println("📈 Generating Solution Quality Histograms (Relative Gap - QN only)...")
    
    df_ok = filter(r -> r.Status == "KKT_OK" && r.h_norm <= 1e-3, df)
    
    all_variants = unique(df.Variant)
    # Filtra exclusivamente as variantes Quasi-Newton
    unif_qn = filter(v -> startswith(v, "UNIF") && occursin("QUASI_NEWTON", v), all_variants)
    tp_qn   = filter(v -> startswith(v, "2P")   && occursin("QUASI_NEWTON", v), all_variants)
    
    best_unif_var = get_best_in_class(unif_qn, summary)
    best_2p_var   = get_best_in_class(tp_qn, summary)
    ipopt_var     = "IPOPT"

    gaps_2p_ipopt = Float64[]
    gaps_unif_ipopt = Float64[]
    gaps_2p_unif = Float64[]

    function get_f(v_name, p_name)
        if isnothing(v_name) return nothing end
        rows = df_ok[(df_ok.Variant .== v_name) .& (df_ok.Problem .== p_name), :]
        return nrow(rows) > 0 ? rows[1, :f_final] : nothing
    end

    for p in unique(df_ok.Problem)
        f_ipopt = get_f(ipopt_var, p)
        f_2p    = get_f(best_2p_var, p)
        f_unif  = get_f(best_unif_var, p)

        if !isnothing(f_2p) && !isnothing(f_ipopt)
            push!(gaps_2p_ipopt, (f_2p - f_ipopt) / max(1e-8, abs(f_ipopt)))
        end

        if !isnothing(f_unif) && !isnothing(f_ipopt)
            push!(gaps_unif_ipopt, (f_unif - f_ipopt) / max(1e-8, abs(f_ipopt)))
        end

        if !isnothing(f_2p) && !isnothing(f_unif)
            push!(gaps_2p_unif, (f_2p - f_unif) / max(1e-8, abs(f_unif)))
        end
    end

    filter_gap(g) = filter(x -> abs(x) <= 0.1, g)

    # Força os títulos para QN
    n_2p = "2P-QN"
    n_unif = "UNIF-QN"

    if !isempty(gaps_2p_ipopt)
        p1 = histogram(filter_gap(gaps_2p_ipopt), bins=50, normalize=:probability, 
                       title="$n_2p vs IPOPT",
                       xlabel="Gap < 0 ($n_2p wins) | Gap > 0 (IPOPT wins)", 
                       ylabel="Frequency", color=colorant"#D8724C", legend=false)
        savefig(p1, joinpath(out_dir, "Plots", "Hist_Gap_2P_vs_IPOPT_QN.png"))
    end

    if !isempty(gaps_unif_ipopt)
        p2 = histogram(filter_gap(gaps_unif_ipopt), bins=50, normalize=:probability, 
                       title="$n_unif vs IPOPT",
                       xlabel="Gap < 0 ($n_unif wins) | Gap > 0 (IPOPT wins)", 
                       ylabel="Frequency", color=colorant"#009CE8", legend=false)
        savefig(p2, joinpath(out_dir, "Plots", "Hist_Gap_UNIF_vs_IPOPT_QN.png"))
    end

    if !isempty(gaps_2p_unif)
        p3 = histogram(filter_gap(gaps_2p_unif), bins=50, normalize=:probability, 
                       title="$n_2p vs $n_unif",
                       xlabel="Gap < 0 ($n_2p wins) | Gap > 0 ($n_unif wins)", 
                       ylabel="Frequency", color=colorant"#4EA553", legend=false)
        savefig(p3, joinpath(out_dir, "Plots", "Hist_Gap_2P_vs_UNIF_QN.png"))
    end
end

end # End of module