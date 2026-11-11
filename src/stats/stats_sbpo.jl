# ==============================================================================
# Estatísticas e Perfis de Desempenho para o SBPO (Resumo Expandido e Artigo)
# ==============================================================================
module StatsSBPO

export run_statistical_analysis

using CSV
using Colors
using DataFrames
using Plots
using Statistics
using XLSX

# ==============================================================================
# 1. FUNÇÕES AUXILIARES DE MÉTRICAS (Comparações XLSX)
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
# 1b. RESOLUÇÃO DE CONFLITO SAR / PH
# ==============================================================================
"""
Quando PH=1, a variação de SAR (0 ou 1) não produz diferença relevante nos
resultados. Manter as duas variantes duplicava observações quase idênticas.
Esta função descarta a variante PH1_SAR0 sempre que existir a variante irmã PH1_SAR1.
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
        println("⚠️  SAR e PH são conflitantes quando PH=1 (diferença desprezível de tempo).")
        println("    Descartando variantes redundantes PH1_SAR0 (mantendo PH1_SAR1): ", join(to_drop, ", "))
        return filter(r -> !(r.Variant in to_drop), df)
    end
    return df
end

# ==============================================================================
# 2. ROTINA PRINCIPAL DE ANÁLISE ESTATÍSTICA
# ==============================================================================
"""
    run_statistical_analysis(results_dir::String; target::Symbol=:artigo)

Gera estatísticas, perfis de desempenho e relatórios em XLSX.
Parâmetro `target`:
  - `:artigo` (padrão): Benchmark completo para o artigo (SBPO 2027), gerando os 6 grupos (G1 a G6).
  - `:resumo`: Filtro para problemas pequenos (n <= 10, m <= 1) com SAR fixo, gerando os 4 perfis do Resumo Expandido.
"""
function run_statistical_analysis(results_dir::String; target::Symbol=:artigo)
    println("\n" * "=" ^ 80)
    println("📊 INICIANDO ANÁLISE ESTATÍSTICA SBPO - MODO: $(uppercase(string(target)))")
    println("=" ^ 80)

    unif_candidates = ["Consolidated_UNIF.csv", "benchmark_slp.csv", "partial_backup_unif.csv"]
    tp_candidates   = ["Consolidated_Twophase.csv", "benchmark_twophase.csv", "partial_backup_twophase.csv"]

    unif_file = ""
    for c in unif_candidates
        if isfile(joinpath(results_dir, c))
            unif_file = joinpath(results_dir, c)
            break
        end
    end

    tp_file = ""
    for c in tp_candidates
        if isfile(joinpath(results_dir, c))
            tp_file = joinpath(results_dir, c)
            break
        end
    end

    println("Carregando bases de dados de: $results_dir ...")
    println("  UNIF/SLP:  $(isempty(unif_file) ? "não encontrado" : basename(unif_file))")
    println("  Two-Phase: $(isempty(tp_file) ? "não encontrado" : basename(tp_file))")
    df_unif = !isempty(unif_file) ? CSV.read(unif_file, DataFrame) : DataFrame()
    df_tp   = !isempty(tp_file)   ? CSV.read(tp_file, DataFrame)   : DataFrame()

    if nrow(df_unif) == 0 && nrow(df_tp) == 0
        println("⚠️ Nenhum dado encontrado em $results_dir. Abortando análise.")
        return
    end

    df_all = vcat(df_unif, df_tp, cols=:union)
    filter!(r -> !occursin("SQP", r.Variant), df_all)
    df_all = remove_redundant_sar_ph_variants(df_all)

    # Filtros específicos do Resumo Expandido
    if target == :resumo
        println("🔍 Aplicando filtros do Resumo Expandido (n <= 10, m <= 1, UNIF com SAR1, 2P com SAR0)...")
        filter!(r -> r.n <= 10 && r.m <= 1, df_all)
        filter!(r -> !(occursin("UNIF", r.Variant) && occursin("SAR0", r.Variant)), df_all)
        filter!(r -> !(occursin("2P", r.Variant) && (occursin("SAR1", r.Variant) || r.Variant == "BASE_2P")), df_all)
        df_all = remove_redundant_sar_ph_variants(df_all)
    end

    # --------------------------------------------------------------------------
    # PARTE A: PERFIS DE DESEMPENHO (DOLAN-MORÉ)
    # --------------------------------------------------------------------------
    summary_df = generate_summary_table(df_all)
    summary_csv = joinpath(results_dir, target == :resumo ? "Summary_Statistics_resumo.csv" : "Summary_Statistics_chap3.csv")
    CSV.write(summary_csv, summary_df)
    println("✅ Estatísticas sumárias salvas em: $(basename(summary_csv))")

    println("📈 Gerando Perfis de Desempenho Dolan-Moré...")
    generate_tournament_profiles(df_all, summary_df, results_dir; target=target)

    # --------------------------------------------------------------------------
    # PARTE B: MÉTRICAS E EXPORTAÇÃO XLSX
    # --------------------------------------------------------------------------
    println("📊 Calculando métricas para exportação XLSX...")
    
    invalid_mask = isnan.(df_all.f_final) .| isinf.(df_all.f_final) .| isnan.(df_all.h_norm) .| isinf.(df_all.h_norm)
    df_valid = df_all[.!invalid_mask, :]

    all_variants  = unique(df_valid.Variant)
    unif_variants = filter(v -> startswith(v, "UNIF") || v == "BASE_UNIF", all_variants)
    tp_variants   = filter(v -> startswith(v, "2P")   || v == "BASE_2P", all_variants)

    # TABELAS DE CENÁRIO
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

    # COMPARAÇÃO INTRA-MÉTODO (Base vs Variantes)
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
        push!(intra_results, (Comparison="$var_name vs $base_name",
            M1_Win=m1_w, M1_Tie=m1_t, M1_Loss=m1_l,
            M2_Win=m2_w, M2_Tie=m2_t, M2_Loss=m2_l,
            M3_Win=m3_w, M3_Tie=m3_t, M3_Loss=m3_l
        ))
    end
    df_intra = DataFrame(intra_results)

    # COMPARAÇÃO INTER-MÉTODO (Best UNIF vs Best 2P)
    inter_results = []
    unif_kkt_probs = unique(filter(r -> r.Variant in unif_variants && r.Status == "KKT_OK", df_valid).Problem)
    tp_kkt_probs   = unique(filter(r -> r.Variant in tp_variants   && r.Status == "KKT_OK", df_valid).Problem)
    common_probs   = intersect(unif_kkt_probs, tp_kkt_probs)

    if !isempty(common_probs)
        m1_w, m1_t, m1_l, m2_w, m2_t, m2_l, m3_w, m3_t, m3_l = zeros(Int, 9)
        for p in common_probs
            sub_unif = filter(r -> r.Problem == p && r.Variant in unif_variants && r.Status == "KKT_OK", df_valid)
            sub_tp   = filter(r -> r.Problem == p && r.Variant in tp_variants   && r.Status == "KKT_OK", df_valid)
            
            sort!(sub_unif, [:f_final, :Iterations])
            sort!(sub_tp,   [:f_final, :Iterations])
            
            best_u = sub_unif[1, :]
            best_t = sub_tp[1, :]
            
            res_m1 = evaluate_m1(best_t.Iterations, best_u.Iterations, best_t.f_final, best_u.f_final, best_t.h_norm, best_u.h_norm, eps)
            res_m1 == :Win ? m1_w += 1 : (res_m1 == :Tie ? m1_t += 1 : m1_l += 1)
            res_m2 = evaluate_m2(best_t.f_final, best_u.f_final, best_t.h_norm, best_u.h_norm, eps)
            res_m2 == :Win ? m2_w += 1 : (res_m2 == :Tie ? m2_t += 1 : m2_l += 1)
            res_m3 = evaluate_m3(best_t.Time_ms, best_u.Time_ms, best_t.f_final, best_u.f_final, best_t.h_norm, best_u.h_norm, eps)
            res_m3 == :Win ? m3_w += 1 : (res_m3 == :Tie ? m3_t += 1 : m3_l += 1)
        end
        push!(inter_results, (Comparison="Best 2P vs Best UNIF (Problem-by-Problem)",
            M1_Win=m1_w, M1_Tie=m1_t, M1_Loss=m1_l,
            M2_Win=m2_w, M2_Tie=m2_t, M2_Loss=m2_l,
            M3_Win=m3_w, M3_Tie=m3_t, M3_Loss=m3_l
        ))
    end
    df_inter = DataFrame(inter_results)

    # GRAVAÇÃO XLSX
    xlsx_name = target == :resumo ? "Consolidated_Statistics_resumo.xlsx" : "Consolidated_Statistics.xlsx"
    xlsx_file = joinpath(results_dir, xlsx_name)
    XLSX.openxlsx(xlsx_file, mode="w") do xf
        XLSX.rename!(xf[1], "Scenario_A")
        XLSX.writetable!(xf["Scenario_A"], stats_A)
        XLSX.addsheet!(xf, "Scenario_B")
        XLSX.writetable!(xf["Scenario_B"], stats_B)
        XLSX.addsheet!(xf, "Scenario_C")
        XLSX.writetable!(xf["Scenario_C"], stats_C)
        if nrow(df_intra) > 0
            XLSX.addsheet!(xf, "Intra_Method")
            XLSX.writetable!(xf["Intra_Method"], df_intra)
        end
        if nrow(df_inter) > 0
            XLSX.addsheet!(xf, "Inter_Method")
            XLSX.writetable!(xf["Inter_Method"], df_inter)
        end
    end
    println("✅ Relatório XLSX salvo com sucesso em: $(basename(xlsx_file))")
    println("=" ^ 80)
end

# ==============================================================================
# 3. LÓGICA DOS PERFIS DE DESEMPENHO (DOLAN-MORÉ)
# ==============================================================================
function generate_summary_table(df::DataFrame)
    variants = unique(df.Variant)
    results = DataFrame(
        Variant = String[],
        Total_Runs = Int[],
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

function generate_tournament_profiles(df::DataFrame, summary::DataFrame, out_dir::String; target::Symbol=:artigo)
    all_variants = unique(df.Variant)
    get_existing(wanted::Vector{String}) = intersect(wanted, all_variants)

    color_map = build_variant_color_map(all_variants)
    plots_dir = joinpath(out_dir, "Plots")
    mkpath(plots_dir)

    unif_pure = filter(v -> occursin("UNIF", v) && !occursin("SQP", v), all_variants)
    tp_pure   = filter(v -> occursin("2P", v)   && !occursin("SQP", v), all_variants)

    best_unif = get_best_in_class(unif_pure, summary)
    best_tp   = get_best_in_class(tp_pure, summary)

    if target == :resumo
        # --- PERFIS DO RESUMO EXPANDIDO (4 gráficos) ---
        # 1. UNIF Shapes
        g1 = get_existing([
            "BASE_UNIF",
            "UNIF_SAR1_PH1_ATR0",
            "UNIF_SAR1_PH1_ATR1"
        ])
        plot_profile(df, g1, :Time_ms, "UNIF - Shapes", joinpath(plots_dir, "Profile_1_UNIF_Shapes.png"), color_map; clean_sar=true)

        # 2. 2P Ratio
        g2 = get_existing([
            "2P_SAR0_PH0_ATR0_RRU0",
            "2P_SAR0_PH0_ATR0_RRU1"
        ])
        plot_profile(df, g2, :Time_ms, "2P - Ratio Strategies", joinpath(plots_dir, "Profile_2_2P_Ratio.png"), color_map; clean_sar=true)

        # 3. 2P Shapes
        g3 = get_existing([
            "2P_SAR0_PH0_ATR0_RRU0",
            "2P_SAR0_PH1_ATR0_RRU0",
            "2P_SAR0_PH1_ATR1_RRU0"
        ])
        plot_profile(df, g3, :Time_ms, "2P - Step Shapes", joinpath(plots_dir, "Profile_3_2P_Shapes.png"), color_map; clean_sar=true)

        # 4. Best Comparison
        g4_variants = String[]
        isnothing(best_unif) || push!(g4_variants, best_unif)
        isnothing(best_tp)   || push!(g4_variants, best_tp)
        g4 = get_existing(unique(g4_variants))
        plot_profile(df, g4, :Time_ms, "Best UNIF vs Best 2P", joinpath(plots_dir, "Profile_4_Best_Comparison.png"), color_map; clean_sar=true)

    else
        # --- PERFIS DO ARTIGO COMPLETO (6 grupos) ---
        # Grupo 1: UNIF - SAR vs PH
        g1 = get_existing([
            "BASE_UNIF",
            "UNIF_SAR0_PH0_ATR0",
            "UNIF_SAR1_PH1_ATR0",
            "UNIF_SAR0_PH1_ATR0"
        ])
        plot_profile(df, g1, :Time_ms, "Group 1: UNIF - SAR vs PH", joinpath(plots_dir, "Profile_G1_UNIF_SAR_PH.png"), color_map)

        # Grupo 2: UNIF - Anisotropia (ATR)
        g2 = get_existing([
            "UNIF_SAR1_PH1_ATR0",
            "UNIF_SAR1_PH1_ATR1",
            "UNIF_SAR0_PH1_ATR0",
            "UNIF_SAR0_PH1_ATR1"
        ])
        plot_profile(df, g2, :Time_ms, "Group 2: UNIF - Anisotropy", joinpath(plots_dir, "Profile_G2_UNIF_ATR.png"), color_map)
        
        # Grupo 3: Two-Phase - Estratégias de Razão (SAR & RRU)
        g3 = get_existing([
            "BASE_2P",
            "2P_SAR0_PH0_ATR0_RRU0",
            "2P_SAR1_PH0_ATR0_RRU1",
            "2P_SAR0_PH0_ATR0_RRU1"
        ])
        plot_profile(df, g3, :Time_ms, "Group 3: 2P - Ratio Strategies", joinpath(plots_dir, "Profile_G3_2P_Ratio.png"), color_map)
        
        # Grupo 4: Two-Phase - Step Shapes (PH & ATR)
        g4 = get_existing([
            "2P_SAR0_PH0_ATR0_RRU1",
            "2P_SAR1_PH0_ATR0_RRU1",
            "2P_SAR0_PH1_ATR0_RRU1",
            "2P_SAR0_PH1_ATR1_RRU1"
        ])
        plot_profile(df, g4, :Time_ms, "Group 4: 2P - Step Shapes", joinpath(plots_dir, "Profile_G4_2P_Shapes.png"), color_map)

        # Grupo 5: Best UNIF vs Best 2P (Linear)
        g5_variants = ["BASE_UNIF", "BASE_2P"]
        isnothing(best_unif) || push!(g5_variants, best_unif)
        isnothing(best_tp)   || push!(g5_variants, best_tp)
        g5 = get_existing(unique(g5_variants))
        plot_profile(df, g5, :Time_ms, "Group 5: Best UNIF vs Best 2P", joinpath(plots_dir, "Profile_G5_Best_Linear.png"), color_map)

        # Grupo 6: Overall Comparison
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
end

const VIVID_PALETTE = [
    colorant"#009CE8", colorant"#D8724C", colorant"#4EA553", colorant"#8E5FBF",
    colorant"#D64550", colorant"#D9A441", colorant"#0F8C8A", colorant"#C72C73",
    colorant"#295CA3", colorant"#965C19", colorant"#6B8224", colorant"#91204D",
    colorant"#D16215", colorant"#57398C", colorant"#1A7A54"
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

function plot_profile(df::DataFrame, solvers::Vector{String}, metric_col::Symbol, title_str::String, filename::String, color_map::AbstractDict=Dict{String,Any}(); clean_sar::Bool=false)
    if length(solvers) < 2
        println("  > Pulando '$title_str': Não há solvers suficientes no grupo.")
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
                floor_val = metric_col == :Iterations ? 1.0 : 1e-3
                perf_matrix[i, j] = val <= 0.0 ? floor_val : val
            end
        end
    end

    min_vals = minimum(perf_matrix, dims=2)
    ratios = perf_matrix ./ min_vals
    ratios[isnan.(ratios)] .= Inf

    valid_ratios = filter(x -> !isinf(x), ratios)
    max_tau = isempty(valid_ratios) ? 10.0 : maximum(valid_ratios)
    plot_max = max(10.0, min(max_tau * 1.05, 1000.0))

    p = plot(title=title_str, 
             xlabel="Performance Ratio (τ)", 
             ylabel="Fraction of Solved Problems", 
             legend=:bottomright, 
             xscale=:log10, 
             framestyle=:box,
             dpi=300,
             size=(800, 600),
             xlims=(1.0, plot_max))

    for (j, s) in enumerate(solvers)
        sorted_r = sort(ratios[:, j])
        filter!(x -> !isinf(x), sorted_r)
        
        n_solved = length(sorted_r)
        
        x_vals = [1.0]
        y_vals = [0.0]
        
        for (k, r) in enumerate(sorted_r)
            if r <= plot_max
                push!(x_vals, r)
                push!(y_vals, k / n_probs)
            else
                push!(x_vals, plot_max)
                push!(y_vals, k / n_probs)
                break
            end
        end
        
        if isempty(x_vals) || x_vals[end] < plot_max
            push!(x_vals, plot_max)
            push!(y_vals, n_solved / n_probs)
        end

        if clean_sar
            clean_label = replace(s, 
                "UNIF_" => "UNIF ", 
                "2P_" => "2P ", 
                "BASE_UNIF" => "UNIF PH0_ATR0",
                "BASE_2P" => "2P PH0_ATR0_RRU0"
            )
            clean_label = replace(clean_label, "SAR1_" => "", "SAR0_" => "")
        else
            clean_label = replace(s, "UNIF_" => "UNIF ", "2P_" => "2P ", "BASE_UNIF" => "UNIF SAR1_PH0_ATR0", "BASE_2P" => "2P SAR1_PH0_ATR0_RRU0")
        end

        line_color = get(color_map, s, nothing)
        if line_color === nothing
            plot!(p, x_vals, y_vals, linetype=:steppost, label=clean_label, linewidth=2.5)
        else
            plot!(p, x_vals, y_vals, linetype=:steppost, label=clean_label, linewidth=2.5, color=line_color)
        end
    end

    savefig(p, filename)
    println("  > Gráfico salvo: $(basename(filename))")
end

end # module

