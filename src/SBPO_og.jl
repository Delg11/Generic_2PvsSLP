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

    unif_file = joinpath(results_dir, "benchmark_slp.csv")
    tp_file   = joinpath(results_dir, "benchmark_twophase.csv")

    println("Loading databases...")
    df_unif = isfile(unif_file) ? CSV.read(unif_file, DataFrame) : DataFrame()
    df_tp   = isfile(tp_file)   ? CSV.read(tp_file, DataFrame)   : DataFrame()

    if nrow(df_unif) == 0 && nrow(df_tp) == 0
        println("⚠️ No data found to process. Exiting analysis.")
        return
    end

    df_all = vcat(df_unif, df_tp, cols=:union)
    filter!(r -> !occursin("SQP", r.Variant), df_all)
    df_all = remove_redundant_sar_ph_variants(df_all)

    # 1. Filtro de dimensão: até 10 variáveis e até 1 restrição
    filter!(r -> r.n <= 10 && r.m <= 1, df_all)

    # 2. UNIF sempre com SAR = true (mantém BASE_UNIF e SAR1, descarta SAR0)
    filter!(r -> !(occursin("UNIF", r.Variant) && occursin("SAR0", r.Variant)), df_all)

    # 3. 2P sempre com SAR = false (mantém SAR0, descarta BASE_2P e SAR1)
    filter!(r -> !(occursin("2P", r.Variant) && (occursin("SAR1", r.Variant) || r.Variant == "BASE_2P")), df_all)

    df_all = remove_redundant_sar_ph_variants(df_all)

    # --------------------------------------------------------------------------
    # PART A: TOURNAMENT PROFILES
    # --------------------------------------------------------------------------
    summary_df = generate_summary_table(df_all)
    CSV.write(joinpath(results_dir, "Summary_Statistics_chap3.csv"), summary_df)
    println("✅ Summary statistics (CSV) saved.")

    println("📈 Generating Dolan-Moré Performance Profiles...")
    generate_tournament_profiles(df_all, summary_df, results_dir)

    # --------------------------------------------------------------------------
    # PART B: METRICS AND XLSX EXPORT
    # --------------------------------------------------------------------------
    println("📊 Calculating mathematical metrics for XLSX export...")
    
    # Filter invalid results (NaN or Inf) for the XLSX metrics
    invalid_mask = isnan.(df_all.f_final) .| isinf.(df_all.f_final) .| isnan.(df_all.h_norm) .| isinf.(df_all.h_norm)
    df_valid = df_all[.!invalid_mask, :]

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

    # INTRA-METHOD COMPARISON (Base vs Variants)
    intra_pairs = []
    if "BASE_UNIF" in unif_variants
        for v in unif_variants
            if v != "BASE_UNIF" push!(intra_pairs, ("BASE_UNIF", v)) end
        end
    end
    if "BASE_2P" in tp_variants
        for v in tp_variants
            if v != "BASE_2P" push!(intra_pairs, ("BASE_2P", v)) end
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
        push!(intra_results, (Base=base_name, Variant=var_name, Total=nrow(df_join), M1_W=m1_w, M1_T=m1_t, M1_L=m1_l, M2_W=m2_w, M2_T=m2_t, M2_L=m2_l, M3_W=m3_w, M3_T=m3_t, M3_L=m3_l))
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

    # Cor fixa por variante, igual em todos os grupos/gráficos deste chapter
    color_map = build_variant_color_map(all_variants)

    plots_dir = joinpath(out_dir, "Plots")
    mkpath(plots_dir)

    # ---------------------------------------------------------
    # GROUP 1: UNIF - The Impact of SAR and PH
    # ---------------------------------------------------------
    g1 = get_existing([
        "BASE_UNIF",            # SAR1, PH0
        "UNIF_SAR0_PH0_ATR0",   # SAR0, PH0
        "UNIF_SAR1_PH1_ATR0",   # SAR1, PH1
        "UNIF_SAR0_PH1_ATR0"    # SAR0, PH1
    ])
    plot_profile(df, g1, :Time_ms, "Group 1: UNIF - SAR vs PH", joinpath(plots_dir, "Profile_G1_UNIF_SAR_PH.png"), color_map)

    # ---------------------------------------------------------
    # GROUP 2: UNIF - The Impact of Anisotropy (ATR)
    # ---------------------------------------------------------
    g2 = get_existing([
        "UNIF_SAR1_PH1_ATR0",
        "UNIF_SAR1_PH1_ATR1",
        "UNIF_SAR0_PH1_ATR0",
        "UNIF_SAR0_PH1_ATR1"
    ])
    plot_profile(df, g2, :Time_ms, "Group 2: UNIF - Anisotropy", joinpath(plots_dir, "Profile_G2_UNIF_ATR.png"), color_map)
    
    # ---------------------------------------------------------
    # GROUP 3: Two-Phase - Ratio Strategies (SAR & RRU)
    # ---------------------------------------------------------
    g3 = get_existing([
        "BASE_2P",               # SAR1, RRU0 (Base teórica: usa Ared/Pred, sem RRU)
        "2P_SAR0_PH0_ATR0_RRU0", # SAR0, RRU0
        "2P_SAR1_PH0_ATR0_RRU1", # SAR1, RRU1
        "2P_SAR0_PH0_ATR0_RRU1"  # SAR0, RRU1
    ])
    plot_profile(df, g3, :Time_ms, "Group 3: 2P - Ratio Strategies", joinpath(plots_dir, "Profile_G3_2P_Ratio.png"), color_map)
    
    # ---------------------------------------------------------
    # GROUP 4: Two-Phase - Step Shapes (PH & ATR)
    # ---------------------------------------------------------
    g4 = get_existing([
        "2P_SAR1_PH0_ATR0_RRU1",
        "2P_SAR1_PH1_ATR0_RRU1",
        "2P_SAR1_PH1_ATR1_RRU1"
    ])
    plot_profile(df, g4, :Time_ms, "Group 4: 2P - Step Shapes", joinpath(plots_dir, "Profile_G4_2P_Shapes.png"), color_map)

    # ---------------------------------------------------------
    # PRE-CALCULUS: Best pure linear of each family
    # ---------------------------------------------------------
    unif_pure = filter(v -> occursin("UNIF", v) && !occursin("SQP", v), all_variants)
    tp_pure   = filter(v -> occursin("2P", v)   && !occursin("SQP", v), all_variants)

    best_unif = get_best_in_class(unif_pure, summary)
    best_tp   = get_best_in_class(tp_pure, summary)

    # ---------------------------------------------------------
    # GROUP 5: Best UNIF vs Best 2P (Linear Only)
    # ---------------------------------------------------------
    g5_variants = ["BASE_UNIF", "BASE_2P"]
    isnothing(best_unif) || push!(g5_variants, best_unif)
    isnothing(best_tp)   || push!(g5_variants, best_tp)
    
    g5 = get_existing(unique(g5_variants))

    plot_profile(df, g5, :Time_ms, "Group 5: Best UNIF vs Best 2P", joinpath(plots_dir, "Profile_G5_Best_Linear.png"), color_map)

    # ---------------------------------------------------------
    # GROUP 6: Overall Comparison
    # ---------------------------------------------------------
    unif_sqp = filter(v -> occursin("UNIF", v) && occursin("SQP", v), all_variants)
    tp_sqp   = filter(v -> occursin("2P", v)   && occursin("SQP", v), all_variants)

    best_unif_sqp = get_best_in_class(unif_sqp, summary)
    best_tp_sqp   = get_best_in_class(tp_sqp, summary)

    g6 = String[]
    isnothing(best_unif)     || push!(g6, best_unif)
    isnothing(best_unif_sqp) || push!(g6, best_unif_sqp)
    isnothing(best_tp)       || push!(g6, best_tp)
    isnothing(best_tp_sqp)   || push!(g6, best_tp_sqp)

    plot_profile(df, g6, :Time_ms, "Group 6: Overall Comparison", joinpath(plots_dir, "Profile_G6_Overall.png"), color_map)
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
    colorant"#009CE8",  # Azul
    colorant"#D8724C",  # Terracota / Laranja
    colorant"#4EA553",  # Verde
    colorant"#8E5FBF",  # Roxo
    colorant"#D64550",  # Vermelho
    colorant"#D9A441",  # Mostarda
    colorant"#0F8C8A",  # Turquesa escuro (maior contraste que o ciano original)
    colorant"#C72C73",  # Magenta forte (mais saturado e escuro que o rosa original)
    colorant"#295CA3",  # Azul cobalto (distingue-se do primeiro azul mais claro)
    colorant"#965C19",  # Marrom acobreado (mais vivo que o marrom tradicional)
    colorant"#6B8224",  # Verde musgo vibrante (evita o aspecto opaco do verde oliva)
    colorant"#91204D",  # Bordô / Vinho escuro (distingue-se do vermelho e do magenta)
    colorant"#D16215",  # Laranja queimado (legível em branco, não confunde com o mostarda)
    colorant"#57398C",  # Índigo (mais escuro que o roxo da quarta posição)
    colorant"#1A7A54",  # Verde esmeralda escuro (distingue-se do primeiro verde)
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
function plot_profile(df::DataFrame, solvers::Vector{String}, metric_col::Symbol, title_str::String, filename::String, color_map::AbstractDict=Dict{String,Any}())
    if length(solvers) < 2
        println("  > Skipping '$title_str': Not enough valid solvers in this group.")
        return
    end

    problems = unique(df.Problem)
    n_probs = length(problems)
    n_solvers = length(solvers)

    perf_matrix = fill(Inf, n_probs, n_solvers)

    for (j, s) in enumerate(solvers)
        for (i, p) in enumerate(problems)
            row = df[(df.Variant .== s) .& (df.Problem .== p), :]
            if nrow(row) > 0 && row[1, :Status] == "KKT_OK"
                val = Float64(row[1, metric_col])
                perf_matrix[i, j] = val <= 0.0 ? 1e-8 : val
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

        # Ajuste da legenda: refatorado para remover menções ao SAR
        clean_label = replace(s, 
            "UNIF_" => "UNIF ", 
            "2P_" => "2P ", 
            "BASE_UNIF" => "UNIF PH0_ATR0", # SAR1 implícito removido da base
            "BASE_2P" => "2P PH0_ATR0_RRU0"
        )
        # Remove sufixos para que o SAR não apareça na legenda final
        clean_label = replace(clean_label, "SAR1_" => "", "SAR0_" => "")

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

end # End of module