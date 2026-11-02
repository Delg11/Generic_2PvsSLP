# ==============================================================================
# Proprietary Software • All Rights Reserved
# ==============================================================================
module Generic_module_Stats

export run_statistical_analysis

using CSV
using DataFrames
using Dates
ENV["GKSwstype"] = "100"
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
# 2. MAIN STATISTICAL ANALYSIS ROUTINE
# ==============================================================================
"""
Main entry point for generating statistics, performance profiles, and XLSX reports.
"""
function run_statistical_analysis(results_dir::String, df_meta::Union{DataFrame, Nothing}=nothing)
    println("\n" * "=" ^ 80)
    println("📊 STARTING CONSOLIDATED ANALYSIS AND STATISTICS")
    println("=" ^ 80)

    unif_candidates = ["Consolidated_UNIF.csv", "benchmark_slp.csv", "partial_backup_unif.csv"]
    tp_candidates = ["Consolidated_Twophase.csv", "benchmark_twophase.csv", "partial_backup_twophase.csv"]

    unif_file = ""
    for c in unif_candidates
        p = joinpath(results_dir, c)
        if isfile(p)
            unif_file = p
            break
        end
    end

    tp_file = ""
    for c in tp_candidates
        p = joinpath(results_dir, c)
        if isfile(p)
            tp_file = p
            break
        end
    end

    println("Loading databases...")
    df_unif = isfile(unif_file) ? CSV.read(unif_file, DataFrame) : DataFrame()
    df_tp = isfile(tp_file) ? CSV.read(tp_file, DataFrame) : DataFrame()

    if nrow(df_unif) == 0 && nrow(df_tp) == 0
        println("⚠️ No data found to process. Exiting analysis.")
        return
    end

    df_all = vcat(df_unif, df_tp, cols=:union)

    # --------------------------------------------------------------------------
    # PART A: TOURNAMENT PROFILES (From Version 1)
    # --------------------------------------------------------------------------
    summary_df = generate_summary_table(df_all)
    CSV.write(joinpath(results_dir, "Summary_Statistics.csv"), summary_df)
    println("✅ Summary statistics (CSV) saved.")

    println("📈 Generating Dolan-Moré Performance Profiles...")
    generate_tournament_profiles(df_all, summary_df, results_dir)

    # --------------------------------------------------------------------------
    # PART B: METRICS AND XLSX EXPORT (From Version 2)
    # --------------------------------------------------------------------------
    println("📊 Calculating mathematical metrics for XLSX export...")
    
    # Filter invalid results (NaN or Inf) for the XLSX metrics
    invalid_mask = isnan.(df_all.f_final) .| isinf.(df_all.f_final) .| isnan.(df_all.h_norm) .| isinf.(df_all.h_norm)
    df_valid = df_all[.!invalid_mask, :]

    all_variants = unique(df_valid.Variant)
    unif_variants = filter(v -> startswith(v, "UNIF") || v == "BASE_UNIF" || startswith(v, "SLP") || v == "BASE_SLP", all_variants)
    tp_variants  = filter(v -> startswith(v, "2P") || v == "BASE_2P", all_variants)

    # SCENARIO TABLES
    df_Scenario_A = df_valid
    df_Scenario_B = filter(r -> !ismissing(r.Iterations) && r.Iterations > 0 && r.Status != "MAX_IT" && r.Status != "TIMEOUT", df_valid)
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
    unif_base = ("BASE_UNIF" in unif_variants) ? "BASE_UNIF" :
                (("UNIF_SAR0_PH1_ATR0" in unif_variants) ? "UNIF_SAR0_PH1_ATR0" :
                (("BASE_SLP" in unif_variants) ? "BASE_SLP" : nothing))
    if !isnothing(unif_base)
        for v in unif_variants
            if v != unif_base push!(intra_pairs, (unif_base, v)) end
        end
    end
    tp_base = ("BASE_2P" in tp_variants) ? "BASE_2P" :
              (("2P_SAR1_PH0_ATR0_RRU1" in tp_variants) ? "2P_SAR1_PH0_ATR0_RRU1" :
              (("2P_SAR0_PH1_ATR0_RRU1" in tp_variants) ? "2P_SAR0_PH1_ATR0_RRU1" : nothing))
    if !isnothing(tp_base)
        for v in tp_variants
            if v != tp_base push!(intra_pairs, (tp_base, v)) end
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

    # INTER-METHOD COMPARISON (Best 2P vs Best SLP)
    m1_w, m1_t, m1_l, m2_w, m2_t, m2_l, m3_w, m3_t, m3_l, total_inter = zeros(Int, 10)
    inter_details = []

    for prob in unique(df_valid.Problem)
        df_prob = filter(r -> r.Problem == prob, df_valid)
        unifs_valid = filter(r -> r.Variant in unif_variants && r.h_norm <= eps, df_prob)
        tps_valid  = filter(r -> r.Variant in tp_variants && r.h_norm <= eps, df_prob)
        
        if nrow(unifs_valid) > 0 && nrow(tps_valid) > 0
            best_unif = sort(unifs_valid, [:f_final, :Iterations])[1, :]
            best_2p = sort(tps_valid, [:f_final, :Iterations])[1, :]
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

    # Se não passou df_meta, tenta ler do arquivo gerado
    if df_meta === nothing
        meta_csv = joinpath(results_dir, "environment_metadata.csv")
        if isfile(meta_csv)
            df_meta = try CSV.read(meta_csv, DataFrame) catch; nothing end
        end
    end

    function _write_sheets(target_path)
        XLSX.openxlsx(target_path, mode="w") do xf
            sheet_idx = 1
            if df_meta !== nothing && nrow(df_meta) > 0
                XLSX.rename!(xf[sheet_idx], "Metadata_Environment")
                XLSX.writetable!(xf[sheet_idx], collect(eachcol(df_meta)), names(df_meta))
                sheet_idx += 1
                XLSX.addsheet!(xf, "Scenario_A")
            else
                XLSX.rename!(xf[sheet_idx], "Scenario_A")
            end

            if nrow(stats_A) > 0 XLSX.writetable!(xf[sheet_idx], collect(eachcol(stats_A)), names(stats_A)) end
            sheet_idx += 1
            
            XLSX.addsheet!(xf, "Scenario_B")
            if nrow(stats_B) > 0 XLSX.writetable!(xf[sheet_idx], collect(eachcol(stats_B)), names(stats_B)) end
            sheet_idx += 1
            
            XLSX.addsheet!(xf, "Scenario_C")
            if nrow(stats_C) > 0 XLSX.writetable!(xf[sheet_idx], collect(eachcol(stats_C)), names(stats_C)) end
            sheet_idx += 1
            
            XLSX.addsheet!(xf, "Intra_Comparison")
            if nrow(df_res_intra) > 0 XLSX.writetable!(xf[sheet_idx], collect(eachcol(df_res_intra)), names(df_res_intra)) end
            sheet_idx += 1
            
            XLSX.addsheet!(xf, "Inter_Comparison")
            if nrow(df_res_inter) > 0 XLSX.writetable!(xf[sheet_idx], collect(eachcol(df_res_inter)), names(df_res_inter)) end
            sheet_idx += 1
            
            XLSX.addsheet!(xf, "Inter_Details")
            if nrow(df_inter_details) > 0 XLSX.writetable!(xf[sheet_idx], collect(eachcol(df_inter_details)), names(df_inter_details)) end
        end
    end

    try
        _write_sheets(output_file)
        println("✅ Relatório XLSX gravado em: $(basename(output_file))")
    catch err
        if occursin("Permission denied", string(err)) || occursin("SystemError", string(err))
            fallback_file = joinpath(results_dir, "Consolidated_Statistics_$(Dates.format(now(), "HH-MM-SS")).xlsx")
            println("⚠️ Não foi possível sobrescrever $(basename(output_file)) (arquivo aberto no Excel).")
            println("   Salvando como: $(basename(fallback_file))")
            try
                _write_sheets(fallback_file)
                println("✅ Relatório XLSX gravado com sucesso no arquivo alternativo!")
            catch e2
                println("⚠️ Falha ao gravar XLSX alternativo: $e2")
            end
        else
            rethrow(err)
        end
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
    # --------------------------------------------------------------------------
    # 1. PERFIS COMPLETOS DE DIAGNÓSTICO (TODAS AS VARIANTES / GRUPOS G1 A G8)
    # --------------------------------------------------------------------------
    plots_dir_all = joinpath(out_dir, "Performance Profile Plots ALL")
    mkpath(plots_dir_all)
    dir_kkt_all = joinpath(plots_dir_all, "1_Status_KKT_OK")
    dir_best_f_all = joinpath(plots_dir_all, "2_Melhor_Objetivo_f")
    mkpath(dir_kkt_all)
    mkpath(dir_best_f_all)

    readme_all = joinpath(plots_dir_all, "README_CRITERIOS.txt")
    write(readme_all, """
================================================================================
METODOLOGIAS DE AVALIAÇÃO DOS PERFIS DE DESEMPENHO (DOLAN-MORÉ) - DIAGNÓSTICO
================================================================================

1. Pasta '1_Status_KKT_OK':
   - Critério: Convergência Formal KKT (Status == "KKT_OK" com viabilidade satisfeita).
   - O patamar final (τ -> ∞) reflete rigorosamente a taxa de sucesso no atendimento
     das condições de otimalidade de primeira ordem KKT.
   - Corresponde 1:1 à métrica 'Solved_KKT' do sumário estatístico.

2. Pasta '2_Melhor_Objetivo_f':
   - Critério: Qualidade Global da Solução (Melhor f(x*) viável encontrada).
   - Um solver pontua no problema se for viável (||h(x*)|| <= 1e-5) e encontrar
     o menor valor da função objetivo f(x*) entre todos os concorrentes viáveis daquele grupo
     (dentro de tolerância relativa/absoluta de 1e-5).
   - O patamar final (τ -> ∞) indica em quantos problemas o algoritmo encontrou a
     melhor solução viável do torneio.
================================================================================
""")

    # --------------------------------------------------------------------------
    # 2. PERFIS PRONTOS PARA O ARTIGO (6 ATOS ESTRUTURADOS E DIDÁTICOS)
    # --------------------------------------------------------------------------
    plots_dir_paper = joinpath(out_dir, "Paper Performance Profile")
    mkpath(plots_dir_paper)
    dir_kkt_paper = joinpath(plots_dir_paper, "1_Status_KKT_OK")
    dir_best_f_paper = joinpath(plots_dir_paper, "2_Melhor_Objetivo_f")
    mkpath(dir_kkt_paper)
    mkpath(dir_best_f_paper)

    readme_paper = joinpath(plots_dir_paper, "README_PAPER_STRUCTURE.txt")
    write(readme_paper, """
================================================================================
DOLAN-MORÉ PERFORMANCE PROFILES FOR PAPER PRESENTATION (6 SCIENTIFIC ACTS)
================================================================================

Act 1: Baseline Ablation Study (SAR, PH, RRU)
       - 1A: UNIF - Impact of SAR & Parabolic Step (PH).
       - 1B: Two-Phase - Impact of Restoration (RRU) & Parabolic Step (PH).
       Objective: Establish the foundational first-order structure prior to anisotropy.

Act 2: The Power of First-Order Anisotropy (Isotropic vs Novel ATR Strategies)
       - 2A: UNIF Anisotropic (Isotropic vs Legacy, Grad Sign, Decomposed Cut, Secant Box).
       - 2B: Two-Phase Anisotropic (Isotropic vs Legacy, Grad Sign, Decomposed Cut, Secant Box).
       Objective: Demonstrate performance gains resulting from coordinate-wise trust-region deformation.

Act 3: First-Order Architectural Duel (UNIF vs Two-Phase)
       - Compares isotropic baselines against elite anisotropic variants in pure SLP.
       Objective: Determine which architecture is more robust and efficient without second order.

Act 4: Second-Order Curvature Evaluation (SQP)
       - 4A: The Limit of Diagonal Approximations (Pure SLP vs Identity, Spectral, Diag Exact, etc.).
       - 4B: The Curvature Breakthrough (Pure SLP vs Best Diagonal vs Damped BFGS vs Exact Hessian).
       Objective: Empirically prove the indispensability of dense cross-coupling terms.

Act 5: Second-Order Synergy (Does Anisotropy Still Matter in SQP?)
       - 5A: Synergy between ATR and Exact Hessian SQP.
       - 5B: Synergy between ATR and Damped BFGS SQP.
       Objective: Assess whether anisotropic boxes continue to improve convergence even with full curvature.

Act 6: Complete Evolutionary Journey (The Grand Finale)
       - Connects Classical Baseline -> Best Linear SLP -> Best SQP BFGS -> Overall Champion Exact SQP.
       Objective: Provide the definitive executive summary plot of the research.
================================================================================
""")

    println("📈 Generating Dolan-Moré Performance Profiles...")
    println("  📁 [1/4] Gerando perfis de diagnóstico KKT em: Performance Profile Plots ALL/1_Status_KKT_OK...")
    _generate_group_plots(df, summary, dir_kkt_all; criterion=:kkt_ok)

    println("  📁 [2/4] Gerando perfis de diagnóstico Melhor f em: Performance Profile Plots ALL/2_Melhor_Objetivo_f...")
    _generate_group_plots(df, summary, dir_best_f_all; criterion=:best_f)

    println("  📁 [3/4] Gerando perfis para Artigo KKT em: Paper Performance Profile/1_Status_KKT_OK...")
    _generate_paper_plots(df, summary, dir_kkt_paper; criterion=:kkt_ok)

    println("  📁 [4/4] Gerando perfis para Artigo Melhor f em: Paper Performance Profile/2_Melhor_Objetivo_f...")
    _generate_paper_plots(df, summary, dir_best_f_paper; criterion=:best_f)
end

function _generate_group_plots(df::DataFrame, summary::DataFrame, target_dir::String; criterion::Symbol=:kkt_ok)
    all_variants = unique(df.Variant)
    get_existing(wanted::Vector{String}) = intersect(wanted, all_variants)
    crit_tag = criterion == :kkt_ok ? "KKT_OK" : "Melhor f"

    # ---------------------------------------------------------
    # GROUP 1: UNIF - The Impact of SAR and PH
    # ---------------------------------------------------------
    g1 = get_existing([
        "BASE_UNIF",            # SAR1, PH0
        "UNIF_SAR0_PH0_ATR0",   # SAR0, PH0
        "UNIF_SAR1_PH1_ATR0",   # SAR1, PH1
        "UNIF_SAR0_PH1_ATR0"    # SAR0, PH1
    ])
    plot_profile(df, g1, :Time_ms, "Group 1: UNIF - SAR vs PH (Time) [$crit_tag]", joinpath(target_dir, "Profile_G1_UNIF_SAR_PH_Time.png"); criterion=criterion)
    plot_profile(df, g1, :Iterations, "Group 1: UNIF - SAR vs PH (Iters) [$crit_tag]", joinpath(target_dir, "Profile_G1_UNIF_SAR_PH_Iters.png"); criterion=criterion)

    # ---------------------------------------------------------
    # GROUP 2: UNIF - The Impact of Anisotropy (ATR)
    # ---------------------------------------------------------
    g2 = get_existing([
        "UNIF_SAR1_PH1_ATR0",
        "UNIF_SAR1_PH1_ATR1",
        "UNIF_SAR0_PH1_ATR0",
        "UNIF_SAR0_PH1_ATR1",
        "UNIF_SAR0_PH1_ATR1_GRAD_SIGN",
        "UNIF_SAR0_PH1_ATR1_DECOMPOSED_CUT",
        "UNIF_SAR0_PH1_ATR1_SECANT_BOX"
    ])
    plot_profile(df, g2, :Time_ms, "Group 2: UNIF - Anisotropy (Time) [$crit_tag]", joinpath(target_dir, "Profile_G2_UNIF_ATR_Time.png"); criterion=criterion)
    plot_profile(df, g2, :Iterations, "Group 2: UNIF - Anisotropy (Iters) [$crit_tag]", joinpath(target_dir, "Profile_G2_UNIF_ATR_Iters.png"); criterion=criterion)
    
    # ---------------------------------------------------------
    # GROUP 3: Two-Phase - Ratio Strategies (SAR & RRU)
    # ---------------------------------------------------------
    g3 = get_existing([
        "BASE_2P",               # SAR1, RRU0
        "2P_SAR0_PH0_ATR0_RRU0", # SAR0, RRU0
        "2P_SAR1_PH0_ATR0_RRU1", # SAR1, RRU1
        "2P_SAR0_PH0_ATR0_RRU1"  # SAR0, RRU1
    ])
    plot_profile(df, g3, :Time_ms, "Group 3: 2P - Ratio Strategies (Time) [$crit_tag]", joinpath(target_dir, "Profile_G3_2P_Ratio_Time.png"); criterion=criterion)
    plot_profile(df, g3, :Iterations, "Group 3: 2P - Ratio Strategies (Iters) [$crit_tag]", joinpath(target_dir, "Profile_G3_2P_Ratio_Iters.png"); criterion=criterion)
    
    # ---------------------------------------------------------
    # GROUP 4: Two-Phase - Step Shapes (PH & ATR)
    # ---------------------------------------------------------
    g4 = get_existing([
        "2P_SAR1_PH0_ATR0_RRU1",
        "2P_SAR1_PH1_ATR0_RRU1",
        "2P_SAR1_PH1_ATR1_RRU1",
        "2P_SAR0_PH1_ATR0_RRU1",
        "2P_SAR0_PH1_ATR1_RRU1",
        "2P_SAR0_PH1_ATR1_RRU1_GRAD_SIGN",
        "2P_SAR0_PH1_ATR1_RRU1_DECOMPOSED_CUT",
        "2P_SAR0_PH1_ATR1_RRU1_SECANT_BOX"
    ])
    plot_profile(df, g4, :Time_ms, "Group 4: 2P - Step Shapes & ATR (Time) [$crit_tag]", joinpath(target_dir, "Profile_G4_2P_Shapes_Time.png"); criterion=criterion)
    plot_profile(df, g4, :Iterations, "Group 4: 2P - Step Shapes & ATR (Iters) [$crit_tag]", joinpath(target_dir, "Profile_G4_2P_Shapes_Iters.png"); criterion=criterion)

    # ---------------------------------------------------------
    # GROUP 8: Direct 2P vs UNIF across Parabolic & ATR Strategies
    # ---------------------------------------------------------
    g8 = get_existing([
        "2P_SAR0_PH1_ATR0_RRU1",
        "2P_SAR0_PH1_ATR1_RRU1",
        "2P_SAR0_PH1_ATR1_RRU1_GRAD_SIGN",
        "2P_SAR0_PH1_ATR1_RRU1_DECOMPOSED_CUT",
        "2P_SAR0_PH1_ATR1_RRU1_SECANT_BOX",
        "UNIF_SAR0_PH1_ATR0",
        "UNIF_SAR0_PH1_ATR1",
        "UNIF_SAR0_PH1_ATR1_GRAD_SIGN",
        "UNIF_SAR0_PH1_ATR1_DECOMPOSED_CUT",
        "UNIF_SAR0_PH1_ATR1_SECANT_BOX"
    ])
    plot_profile(df, g8, :Time_ms, "Group 8: 2P vs UNIF - All Parabolic/ATR (Time) [$crit_tag]", joinpath(target_dir, "Profile_G8_2P_vs_UNIF_ATR_Time.png"); criterion=criterion)
    plot_profile(df, g8, :Iterations, "Group 8: 2P vs UNIF - All Parabolic/ATR (Iters) [$crit_tag]", joinpath(target_dir, "Profile_G8_2P_vs_UNIF_ATR_Iters.png"); criterion=criterion)

    # ---------------------------------------------------------
    # GROUP 5: UNIF SQP Hessian Battle
    # ---------------------------------------------------------
    unif_pure = filter(v -> (occursin("UNIF", v) || occursin("SLP", v)) && !occursin("SQP", v), all_variants)
    best_unif = get_best_in_class(unif_pure, summary)

    sqp_suffixes = [
        "_SQP_EXACT",
        "_SQP_BFGS",
        "_SQP_DIAG_EXACT",
        "_SQP_SPECTRAL",
        "_SQP_IDENTITY",
        "_SQP_QUASI_NEWTON",
        "_SQP_RECIPROCAL",
        "_SQP_EXPONENTIAL"
    ]

    if !isnothing(best_unif)
        candidates_unif = vcat([best_unif], ["$(best_unif)$(s)" for s in sqp_suffixes])
        g5 = get_existing(candidates_unif)
        if length(g5) <= 1
            g5 = String[]
            best_unif in all_variants && push!(g5, best_unif)
            for s in sqp_suffixes
                matching = filter(v -> (occursin("UNIF", v) || occursin("SLP", v)) && endswith(v, s), all_variants)
                b = get_best_in_class(matching, summary)
                !isnothing(b) && !(b in g5) && push!(g5, b)
            end
        end
        if !isempty(g5)
            plot_profile(df, g5, :Time_ms, "Group 5: UNIF SQP Strategies (Time) [$crit_tag]", joinpath(target_dir, "Profile_G5_UNIF_SQP_Time.png"); criterion=criterion)
            plot_profile(df, g5, :Iterations, "Group 5: UNIF SQP Strategies (Iters) [$crit_tag]", joinpath(target_dir, "Profile_G5_UNIF_SQP_Iters.png"); criterion=criterion)
        end
    end

    # ---------------------------------------------------------
    # GROUP 6: 2P SQP Hessian Battle
    # ---------------------------------------------------------
    tp_pure = filter(v -> occursin("2P", v) && !occursin("SQP", v), all_variants)
    best_tp = get_best_in_class(tp_pure, summary)

    if !isnothing(best_tp)
        candidates_tp = vcat([best_tp], ["$(best_tp)$(s)" for s in sqp_suffixes])
        g6 = get_existing(candidates_tp)
        if length(g6) <= 1
            g6 = String[]
            best_tp in all_variants && push!(g6, best_tp)
            for s in sqp_suffixes
                matching = filter(v -> occursin("2P", v) && endswith(v, s), all_variants)
                b = get_best_in_class(matching, summary)
                !isnothing(b) && !(b in g6) && push!(g6, b)
            end
        end
        if !isempty(g6)
            plot_profile(df, g6, :Time_ms, "Group 6: 2P SQP Strategies (Time) [$crit_tag]", joinpath(target_dir, "Profile_G6_2P_SQP_Time.png"); criterion=criterion)
            plot_profile(df, g6, :Iterations, "Group 6: 2P SQP Strategies (Iters) [$crit_tag]", joinpath(target_dir, "Profile_G6_2P_SQP_Iters.png"); criterion=criterion)
        end
    end

    # ---------------------------------------------------------
    # GROUP 7: THE GRAND FINALE
    # ---------------------------------------------------------
    unif_sqp  = filter(v -> (occursin("UNIF", v) || occursin("SLP", v)) && occursin("SQP", v), all_variants)
    tp_sqp   = filter(v -> occursin("2P", v) && occursin("SQP", v), all_variants)

    best_unif_sqp  = get_best_in_class(unif_sqp, summary)
    best_tp_sqp   = get_best_in_class(tp_sqp, summary)

    g7 = String[]
    isnothing(best_unif)      || push!(g7, best_unif)
    isnothing(best_unif_sqp)  || push!(g7, best_unif_sqp)
    isnothing(best_tp)       || push!(g7, best_tp)
    isnothing(best_tp_sqp)   || push!(g7, best_tp_sqp)

    plot_profile(df, g7, :Time_ms, "Best of Classes (Time) [$crit_tag]", joinpath(target_dir, "Profile_G7_Finale_Time.png"); criterion=criterion)
    plot_profile(df, g7, :Iterations, "Best of Classes (Iters) [$crit_tag]", joinpath(target_dir, "Profile_G7_Finale_Iters.png"); criterion=criterion)
end

function _generate_paper_plots(df::DataFrame, summary::DataFrame, target_dir::String; criterion::Symbol=:kkt_ok)
    all_variants = unique(df.Variant)
    get_existing(wanted::Vector{String}) = intersect(wanted, all_variants)
    crit_tag = criterion == :kkt_ok ? "KKT OK" : "Best f"

    # =========================================================================
    # ACT 1: BASELINE ABLATION STUDY (SAR, PH, RRU)
    # =========================================================================
    # 1A: UNIF Base
    act1a = get_existing([
        "BASE_UNIF",
        "UNIF_SAR0_PH0_ATR0",
        "UNIF_SAR1_PH1_ATR0",
        "UNIF_SAR0_PH1_ATR0"
    ])
    plot_profile(df, act1a, :Time_ms, "Act 1A: UNIF - Impact of SAR & Parabolic Step (Time) [$crit_tag]", joinpath(target_dir, "Act_1A_UNIF_Base_Time.png"); criterion=criterion, paper_mode=true)
    plot_profile(df, act1a, :Iterations, "Act 1A: UNIF - Impact of SAR & Parabolic Step (Iters) [$crit_tag]", joinpath(target_dir, "Act_1A_UNIF_Base_Iters.png"); criterion=criterion, paper_mode=true)

    # 1B: Two-Phase Base
    act1b = get_existing([
        "BASE_2P",
        "2P_SAR0_PH0_ATR0_RRU0",
        "2P_SAR1_PH0_ATR0_RRU1",
        "2P_SAR0_PH0_ATR0_RRU1",
        "2P_SAR0_PH1_ATR0_RRU1"
    ])
    plot_profile(df, act1b, :Time_ms, "Act 1B: Two-Phase - Impact of RRU & Parabolic Step (Time) [$crit_tag]", joinpath(target_dir, "Act_1B_2P_Base_Time.png"); criterion=criterion, paper_mode=true)
    plot_profile(df, act1b, :Iterations, "Act 1B: Two-Phase - Impact of RRU & Parabolic Step (Iters) [$crit_tag]", joinpath(target_dir, "Act_1B_2P_Base_Iters.png"); criterion=criterion, paper_mode=true)

    # =========================================================================
    # ACT 2: THE POWER OF 1ST-ORDER ANISOTROPY (ISOTROPIC VS ATR STRATEGIES)
    # =========================================================================
    # 2A: UNIF Anisotropy
    act2a = get_existing([
        "UNIF_SAR0_PH1_ATR0",
        "UNIF_SAR0_PH1_ATR1",
        "UNIF_SAR0_PH1_ATR1_GRAD_SIGN",
        "UNIF_SAR0_PH1_ATR1_DECOMPOSED_CUT",
        "UNIF_SAR0_PH1_ATR1_SECANT_BOX"
    ])
    plot_profile(df, act2a, :Time_ms, "Act 2A: UNIF - Isotropic vs ATR Strategies (Time) [$crit_tag]", joinpath(target_dir, "Act_2A_UNIF_Anisotropy_Time.png"); criterion=criterion, paper_mode=true)
    plot_profile(df, act2a, :Iterations, "Act 2A: UNIF - Isotropic vs ATR Strategies (Iters) [$crit_tag]", joinpath(target_dir, "Act_2A_UNIF_Anisotropy_Iters.png"); criterion=criterion, paper_mode=true)

    # 2B: Two-Phase Anisotropy
    act2b = get_existing([
        "2P_SAR0_PH1_ATR0_RRU1",
        "2P_SAR0_PH1_ATR1_RRU1",
        "2P_SAR0_PH1_ATR1_RRU1_GRAD_SIGN",
        "2P_SAR0_PH1_ATR1_RRU1_DECOMPOSED_CUT",
        "2P_SAR0_PH1_ATR1_RRU1_SECANT_BOX"
    ])
    plot_profile(df, act2b, :Time_ms, "Act 2B: Two-Phase - Isotropic vs ATR Strategies (Time) [$crit_tag]", joinpath(target_dir, "Act_2B_2P_Anisotropy_Time.png"); criterion=criterion, paper_mode=true)
    plot_profile(df, act2b, :Iterations, "Act 2B: Two-Phase - Isotropic vs ATR Strategies (Iters) [$crit_tag]", joinpath(target_dir, "Act_2B_2P_Anisotropy_Iters.png"); criterion=criterion, paper_mode=true)

    # =========================================================================
    # ACT 3: FIRST-ORDER ARCHITECTURAL DUEL (UNIF VS TWO-PHASE)
    # =========================================================================
    act3 = get_existing([
        "UNIF_SAR0_PH1_ATR0",                   # Baseline UNIF Isotropic
        "2P_SAR0_PH1_ATR0_RRU1",                # Baseline 2P Isotropic
        "UNIF_SAR0_PH1_ATR1_DECOMPOSED_CUT",    # Best UNIF
        "UNIF_SAR0_PH1_ATR1_SECANT_BOX",        # Runner-up UNIF
        "2P_SAR0_PH1_ATR1_RRU1_SECANT_BOX",     # Best 2P
        "2P_SAR0_PH1_ATR1_RRU1_DECOMPOSED_CUT"  # Runner-up 2P
    ])
    plot_profile(df, act3, :Time_ms, "Act 3: UNIF vs Two-Phase in 1st Order (Time) [$crit_tag]", joinpath(target_dir, "Act_3_Architectures_SLP_Time.png"); criterion=criterion, paper_mode=true)
    plot_profile(df, act3, :Iterations, "Act 3: UNIF vs Two-Phase in 1st Order (Iters) [$crit_tag]", joinpath(target_dir, "Act_3_Architectures_SLP_Iters.png"); criterion=criterion, paper_mode=true)

    # =========================================================================
    # ACT 4: SECOND-ORDER EVALUATION (SQP: DIAGONAL VS DENSE CURVATURE)
    # =========================================================================
    # 4A: The Limit of Diagonal Approximations
    tp_pure = filter(v -> occursin("2P", v) && !occursin("SQP", v), all_variants)
    best_tp_slp = get_best_in_class(tp_pure, summary)
    base_diag = !isnothing(best_tp_slp) ? best_tp_slp : "2P_SAR0_PH1_ATR1_RRU1_SECANT_BOX"

    act4a = get_existing([
        base_diag,
        "$(base_diag)_SQP_IDENTITY",
        "$(base_diag)_SQP_SPECTRAL",
        "$(base_diag)_SQP_DIAG_EXACT",
        "$(base_diag)_SQP_QUASI_NEWTON",
        "$(base_diag)_SQP_RECIPROCAL",
        "$(base_diag)_SQP_EXPONENTIAL"
    ])
    plot_profile(df, act4a, :Time_ms, "Act 4A: Diagonal Hessian Approximations (Time) [$crit_tag]", joinpath(target_dir, "Act_4A_SQP_Diagonals_Time.png"); criterion=criterion, paper_mode=true)
    plot_profile(df, act4a, :Iterations, "Act 4A: Diagonal Hessian Approximations (Iters) [$crit_tag]", joinpath(target_dir, "Act_4A_SQP_Diagonals_Iters.png"); criterion=criterion, paper_mode=true)

    # 4B: The Curvature Breakthrough (Pure SLP vs Best Diagonal vs BFGS vs Exact)
    diag_candidates = filter(v -> occursin("2P", v) && (occursin("IDENTITY", v) || occursin("SPECTRAL", v) || occursin("DIAG_EXACT", v) || occursin("QUASI_NEWTON", v)), all_variants)
    best_diag = get_best_in_class(diag_candidates, summary)

    act4b = String[]
    !isnothing(best_tp_slp) && push!(act4b, best_tp_slp)
    !isnothing(best_diag) && push!(act4b, best_diag)
    
    if !isnothing(best_tp_slp) && "$(best_tp_slp)_SQP_BFGS" in all_variants
        push!(act4b, "$(best_tp_slp)_SQP_BFGS")
    else
        bfgs_vars = filter(v -> occursin("2P", v) && occursin("SQP_BFGS", v), all_variants)
        b_bfgs = get_best_in_class(bfgs_vars, summary)
        !isnothing(b_bfgs) && push!(act4b, b_bfgs)
    end

    if !isnothing(best_tp_slp) && "$(best_tp_slp)_SQP_EXACT" in all_variants
        push!(act4b, "$(best_tp_slp)_SQP_EXACT")
    else
        exact_vars = filter(v -> occursin("2P", v) && occursin("SQP_EXACT", v), all_variants)
        b_exact = get_best_in_class(exact_vars, summary)
        !isnothing(b_exact) && push!(act4b, b_exact)
    end
    act4b = get_existing(act4b)

    plot_profile(df, act4b, :Time_ms, "Act 4B: 2nd Order Curvature - Linear vs BFGS vs Exact (Time) [$crit_tag]", joinpath(target_dir, "Act_4B_Curvature_2ndOrder_Time.png"); criterion=criterion, paper_mode=true)
    plot_profile(df, act4b, :Iterations, "Act 4B: 2nd Order Curvature - Linear vs BFGS vs Exact (Iters) [$crit_tag]", joinpath(target_dir, "Act_4B_Curvature_2ndOrder_Iters.png"); criterion=criterion, paper_mode=true)

    # =========================================================================
    # ACT 5: SECOND-ORDER SYNERGY (DOES ANISOTROPY STILL MATTER IN SQP?)
    # =========================================================================
    # 5A: Synergy ATR + SQP Exact
    act5a = get_existing([
        "2P_SAR0_PH1_ATR0_RRU1_SQP_EXACT",
        "2P_SAR0_PH1_ATR1_RRU1_SQP_EXACT",
        "2P_SAR0_PH1_ATR1_RRU1_GRAD_SIGN_SQP_EXACT",
        "2P_SAR0_PH1_ATR1_RRU1_DECOMPOSED_CUT_SQP_EXACT",
        "2P_SAR0_PH1_ATR1_RRU1_SECANT_BOX_SQP_EXACT"
    ])
    plot_profile(df, act5a, :Time_ms, "Act 5A: ATR Synergy with Exact SQP (Time) [$crit_tag]", joinpath(target_dir, "Act_5A_Synergy_ATR_SQP_Exact_Time.png"); criterion=criterion, paper_mode=true)
    plot_profile(df, act5a, :Iterations, "Act 5A: ATR Synergy with Exact SQP (Iters) [$crit_tag]", joinpath(target_dir, "Act_5A_Synergy_ATR_SQP_Exact_Iters.png"); criterion=criterion, paper_mode=true)

    # 5B: Synergy ATR + SQP BFGS
    act5b = get_existing([
        "2P_SAR0_PH1_ATR0_RRU1_SQP_BFGS",
        "2P_SAR0_PH1_ATR1_RRU1_SQP_BFGS",
        "2P_SAR0_PH1_ATR1_RRU1_GRAD_SIGN_SQP_BFGS",
        "2P_SAR0_PH1_ATR1_RRU1_DECOMPOSED_CUT_SQP_BFGS",
        "2P_SAR0_PH1_ATR1_RRU1_SECANT_BOX_SQP_BFGS"
    ])
    plot_profile(df, act5b, :Time_ms, "Act 5B: ATR Synergy with BFGS SQP (Time) [$crit_tag]", joinpath(target_dir, "Act_5B_Synergy_ATR_SQP_BFGS_Time.png"); criterion=criterion, paper_mode=true)
    plot_profile(df, act5b, :Iterations, "Act 5B: ATR Synergy with BFGS SQP (Iters) [$crit_tag]", joinpath(target_dir, "Act_5B_Synergy_ATR_SQP_BFGS_Iters.png"); criterion=criterion, paper_mode=true)

    # =========================================================================
    # ACT 6: GRAND FINALE (COMPLETE EVOLUTIONARY JOURNEY)
    # =========================================================================
    unif_slp_best = get_best_in_class(filter(v -> occursin("UNIF", v) && !occursin("SQP", v), all_variants), summary)
    tp_slp_best   = get_best_in_class(filter(v -> occursin("2P", v) && !occursin("SQP", v), all_variants), summary)
    tp_bfgs_best  = get_best_in_class(filter(v -> occursin("2P", v) && occursin("BFGS", v), all_variants), summary)
    tp_exact_best = get_best_in_class(filter(v -> occursin("2P", v) && occursin("EXACT", v), all_variants), summary)

    act6 = String[]
    "2P_SAR0_PH1_ATR0_RRU1" in all_variants && push!(act6, "2P_SAR0_PH1_ATR0_RRU1") # Isotropic Baseline
    !isnothing(unif_slp_best) && push!(act6, unif_slp_best)
    !isnothing(tp_slp_best)   && push!(act6, tp_slp_best)
    !isnothing(tp_bfgs_best)  && push!(act6, tp_bfgs_best)
    !isnothing(tp_exact_best) && push!(act6, tp_exact_best)
    act6 = get_existing(act6)

    plot_profile(df, act6, :Time_ms, "Act 6: Complete Evolutionary Journey (Time) [$crit_tag]", joinpath(target_dir, "Act_6_Complete_Evolution_Time.png"); criterion=criterion, paper_mode=true)
    plot_profile(df, act6, :Iterations, "Act 6: Complete Evolutionary Journey (Iters) [$crit_tag]", joinpath(target_dir, "Act_6_Complete_Evolution_Iters.png"); criterion=criterion, paper_mode=true)
end

function get_best_in_class(class_variants::Vector{String}, summary::DataFrame)
    if isempty(class_variants) return nothing end
    sub = summary[in.(summary.Variant, Ref(class_variants)), :]
    if nrow(sub) == 0 return nothing end
    return sub[1, :Variant]
end

"""
Formats variant names into clean, publication-ready strings in en-US,
embedding the final success rate in brackets for immediate legibility.
"""
function format_paper_label(s::String, success_rate::Float64)
    rate_str = string(round(success_rate, digits=1), "%")

    if s == "BASE_UNIF" return "UNIF Classical Base [$rate_str]" end
    if s == "BASE_2P"   return "2P Classical Base [$rate_str]" end

    arch = occursin("2P", s) ? "2P" : (occursin("UNIF", s) ? "UNIF" : "")

    # SQP component
    sqp_str = ""
    if occursin("_SQP_EXACT", s)
        sqp_str = " + SQP Exact"
    elseif occursin("_SQP_BFGS", s)
        sqp_str = " + SQP BFGS"
    elseif occursin("_SQP_DIAG_EXACT", s)
        sqp_str = " + SQP Diag Exact"
    elseif occursin("_SQP_SPECTRAL", s)
        sqp_str = " + SQP Spectral"
    elseif occursin("_SQP_IDENTITY", s)
        sqp_str = " + SQP Identity"
    elseif occursin("_SQP_QUASI_NEWTON", s)
        sqp_str = " + SQP Quasi-Newton Diag"
    elseif occursin("_SQP_RECIPROCAL", s)
        sqp_str = " + SQP Reciprocal"
    elseif occursin("_SQP_EXPONENTIAL", s)
        sqp_str = " + SQP Exponential"
    end

    # ATR component / Step shape
    atr_str = ""
    if occursin("GRAD_SIGN", s)
        atr_str = " + ATR (Grad Sign)"
    elseif occursin("DECOMPOSED_CUT", s)
        atr_str = " + ATR (Decomposed Cut)"
    elseif occursin("SECANT_BOX", s)
        atr_str = " + ATR (Secant Box)"
    elseif occursin("ATR1", s)
        atr_str = " + ATR (Legacy)"
    elseif occursin("ATR0", s) || occursin("BASE", s)
        if occursin("PH1", s)
            atr_str = occursin("SQP", s) ? " Isotropic" : " Isotropic (Parabolic Step)"
        elseif occursin("SAR0_PH0", s)
            atr_str = " Isotropic (SAR0, PH0)"
        elseif occursin("SAR1_PH0", s)
            atr_str = " Isotropic (SAR1, PH0)"
        elseif occursin("SAR1_PH1", s)
            atr_str = " Isotropic (SAR1, PH1)"
        else
            atr_str = " Isotropic"
        end
    end

    # Distinction of RRU for 2P base variants
    if occursin("2P", s) && !occursin("ATR1", s) && !occursin("SQP", s)
        if occursin("RRU0", s)
            atr_str = replace(atr_str, ")" => ", RRU0)")
        elseif occursin("RRU1", s) && occursin("PH0", s)
            atr_str = replace(atr_str, ")" => ", RRU1)")
        end
    end

    return strip("$arch$atr_str$sqp_str") * " [$rate_str]"
end

"""
    plot_profile(df, solvers, metric_col, title_str, filename, color_map; criterion=:kkt_ok, f_tol=1e-5, h_tol=1e-5, paper_mode=false)

Constrói um perfil de desempenho de Dolan-Moré sob dois critérios metodológicos distintos:
- `criterion=:kkt_ok` : Mede formalmente a convergência de primeira ordem (`Status == "KKT_OK"` com viabilidade satisfeita).
- `criterion=:best_f` : Mede a qualidade do ponto encontrado, considerando como vencedores os algoritmos que alcançaram
                         o menor valor viável de f(x*) entre os concorrentes (com tolerância relativa/absoluta de 1e-5).

Quando `paper_mode=true`:
- O eixo Y exibe porcentagem formatada (0% a 100%) com grade sutil.
- As curvas na legenda são estritamente ordenadas da maior para a menor taxa de sucesso.
- Os rótulos são limpos com `format_paper_label` anexando a taxa final `[XX.X%]`.
- Métodos baseline/isotrópicos recebem linha tracejada (`:dash`) para imediata identificação como referências.
- Os vencedores recebem linhas de maior espessura (`linewidth=3.2`).
"""
function plot_profile(
    df::DataFrame,
    solvers_in::AbstractVector,
    metric_col::Symbol,
    title_str::String,
    filename::String,
    color_map::AbstractDict=Dict{String,Any}();
    criterion::Symbol=:kkt_ok,
    f_tol::Float64=1e-5,
    h_tol::Float64=1e-5,
    paper_mode::Bool=false
)
    solvers = String.(filter(!ismissing, solvers_in))
    if length(solvers) < 2
        println("  > Skipping '$title_str': Not enough valid solvers in this group.")
        return
    end

    problems = unique(df.Problem)
    n_probs = length(problems)
    n_solvers = length(solvers)

    perf_matrix = fill(Inf, n_probs, n_solvers)
    infeasible_statuses = ["INFEASIBLE_STATIONARY", "STALLED_INFEASIBLE", "FAIL", "ERROR", "TIMEOUT"]

    has_h = :h_norm in propertynames(df)

    if criterion == :kkt_ok
        # ----------------------------------------------------------------------
        # CRITÉRIO 1: STATUS KKT_OK (Convergência Formal)
        # ----------------------------------------------------------------------
        for (j, s) in enumerate(solvers)
            for (i, p) in enumerate(problems)
                row = df[(df.Variant .== s) .& (df.Problem .== p), :]
                if nrow(row) > 0
                    st = string(row[1, :Status])
                    h_val = has_h ? Float64(row[1, :h_norm]) : 0.0
                    
                    if st == "KKT_OK" && (isnan(h_val) || h_val <= h_tol)
                        val = Float64(row[1, metric_col])
                        if isfinite(val)
                            perf_matrix[i, j] = val <= 0.0 ? 1e-8 : val
                        end
                    end
                end
            end
        end
    else
        # ----------------------------------------------------------------------
        # CRITÉRIO 2: MELHOR f(x*) (Qualidade Global da Solução)
        # ----------------------------------------------------------------------
        for (i, p) in enumerate(problems)
            df_p = df[df.Problem .== p, :]
            
            # Etapa 1: Encontrar o menor f(x*) entre todos os solvers viáveis do grupo
            best_f = Inf
            for s in solvers
                row = df_p[df_p.Variant .== s, :]
                if nrow(row) > 0
                    st = string(row[1, :Status])
                    h_val = has_h ? Float64(row[1, :h_norm]) : 0.0
                    f_val = Float64(row[1, :f_final])
                    
                    if !(st in infeasible_statuses) && (isnan(h_val) || h_val <= h_tol) && isfinite(f_val)
                        best_f = min(best_f, f_val)
                    end
                end
            end

            isinf(best_f) && continue

            # Etapa 2: Preencher matriz para os que empatam com a melhor função objetivo
            for (j, s) in enumerate(solvers)
                row = df_p[df_p.Variant .== s, :]
                if nrow(row) > 0
                    st = string(row[1, :Status])
                    h_val = has_h ? Float64(row[1, :h_norm]) : 0.0
                    f_val = Float64(row[1, :f_final])
                    
                    if !(st in infeasible_statuses) && (isnan(h_val) || h_val <= h_tol) && isfinite(f_val)
                        if f_val <= best_f + f_tol * (1.0 + abs(best_f))
                            val = Float64(row[1, metric_col])
                            if isfinite(val)
                                perf_matrix[i, j] = val <= 0.0 ? 1e-8 : val
                            end
                        end
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
    plot_max = min(max(10.0, max_tau * 1.1), 1000.0)

    # Coleta dados e calcula estatísticas para ordenação da legenda
    solver_stats = []
    for (j, s) in enumerate(solvers)
        sorted_r = sort(ratios[:, j])
        filter!(x -> !isinf(x), sorted_r)
        n_solved = length(sorted_r)
        rate = (n_solved / n_probs) * 100.0
        tau1 = count(r -> r <= 1.0 + 1e-6, sorted_r)
        push!(solver_stats, (solver=s, orig_idx=j, sorted_r=sorted_r, n_solved=n_solved, rate=rate, tau1=tau1))
    end

    if paper_mode
        # Ordena a legenda rigorosamente pelo número de resolvidos; em empate, pelo tau=1
        sort!(solver_stats, by = x -> (-x.n_solved, -x.tau1))
    end

    # Configuração do gráfico
    if paper_mode
        ylabel_str = criterion == :kkt_ok ? "Solved Problems (%)" : "Best Feasible f(x*) (%)"
        p = plot(title=title_str, 
                 xlabel="Performance Ratio (τ)", 
                 ylabel=ylabel_str, 
                 legend=:bottomright, 
                 xscale=:log10, 
                 framestyle=:box,
                 dpi=300,
                 size=(900, 650),
                 titlefontsize=11,
                 guidefontsize=10,
                 tickfontsize=9,
                 legendfontsize=8,
                 ylims=(-0.02, 1.05),
                 yticks=(0.0:0.1:1.0, ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]),
                 grid=true,
                 gridalpha=0.35,
                 gridstyle=:dot)
    else
        ylabel_str = criterion == :kkt_ok ? "Fraction of Solved Problems (KKT OK)" : "Fraction with Best Feasible f(x*)"
        p = plot(title=title_str, 
                 xlabel="Performance Ratio (τ)", 
                 ylabel=ylabel_str, 
                 legend=:bottomright, 
                 xscale=:log10, 
                 framestyle=:box,
                 dpi=300,
                 size=(800, 600))
    end

    # Paleta de alta distinção para publicação
    paper_palette = [
        RGB(0.00, 0.28, 0.67), # 1: Azul Royal Intenso
        RGB(0.84, 0.15, 0.15), # 2: Vermelho Carmesim
        RGB(0.13, 0.55, 0.13), # 3: Verde Floresta
        RGB(0.90, 0.45, 0.00), # 4: Laranja Vibrante
        RGB(0.49, 0.18, 0.56), # 5: Roxo Intenso
        RGB(0.00, 0.60, 0.60), # 6: Ciano Escuro / Teal
        RGB(0.80, 0.60, 0.00), # 7: Dourado Mostarda
        RGB(0.85, 0.20, 0.55), # 8: Magenta
        RGB(0.20, 0.45, 0.65), # 9: Azul Petróleo
        RGB(0.55, 0.30, 0.10)  # 10: Marrom
    ]

    for (rank, item) in enumerate(solver_stats)
        s = item.solver
        sorted_r = item.sorted_r
        n_solved = item.n_solved
        rate = item.rate

        x_vals = [1.0]
        y_vals = [0.0]

        for (k, r) in enumerate(sorted_r)
            push!(x_vals, r)
            push!(y_vals, k / n_probs)
        end

        push!(x_vals, plot_max)
        push!(y_vals, n_solved / n_probs)

        clean_label = if paper_mode
            format_paper_label(s, rate)
        else
            replace(s, "UNIF_" => "UNIF ", "2P_" => "2P ", "BASE_UNIF" => "UNIF SAR1_PH0_ATR0", "BASE_2P" => "2P SAR1_PH0_ATR0_RRU0")
        end

        is_baseline = occursin("ATR0", s) || occursin("BASE", s)

        ls = (paper_mode && is_baseline) ? :dash : :solid
        lw = if paper_mode
            if rank == 1
                3.2
            elseif rank == 2
                2.8
            elseif is_baseline
                2.0
            else
                2.2
            end
        else
            2.5
        end

        line_color = get(color_map, s, nothing)
        if line_color === nothing && paper_mode
            color_idx = mod1(rank, length(paper_palette))
            line_color = paper_palette[color_idx]
        end

        plot_kwargs = Dict{Symbol, Any}(
            :linetype => :steppost,
            :label => clean_label,
            :linewidth => lw,
            :linestyle => ls
        )
        if line_color !== nothing
            plot_kwargs[:color] = line_color
        end

        plot!(p, x_vals, y_vals; plot_kwargs...)
    end

    savefig(p, filename)
    println("  > Saved: $(basename(filename))")
end

end # End of module