# ==============================================================================
# Proprietary Software • All Rights Reserved
# ==============================================================================
module Generic_module_Stats

export run_statistical_analysis

using CSV
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
# 2. MAIN STATISTICAL ANALYSIS ROUTINE
# ==============================================================================
"""
Main entry point for generating statistics, performance profiles, and XLSX reports.
"""
function run_statistical_analysis(results_dir::String)
    println("\n" * "=" ^ 80)
    println("📊 STARTING CONSOLIDATED ANALYSIS AND STATISTICS")
    println("=" ^ 80)

    slp_file = joinpath(results_dir, "partial_backup_slp.csv")
    tp_file = joinpath(results_dir, "partial_backup_twophase.csv")

    println("Loading databases...")
    df_slp = isfile(slp_file) ? CSV.read(slp_file, DataFrame) : DataFrame()
    df_tp = isfile(tp_file) ? CSV.read(tp_file, DataFrame) : DataFrame()

    if nrow(df_slp) == 0 && nrow(df_tp) == 0
        println("⚠️ No data found to process. Exiting analysis.")
        return
    end

    df_all = vcat(df_slp, df_tp, cols=:union)

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
    slp_variants = filter(v -> startswith(v, "SLP") || v == "BASE_SLP", all_variants)
    tp_variants  = filter(v -> startswith(v, "2P") || v == "BASE_2P", all_variants)

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
    if "BASE_SLP" in slp_variants
        for v in slp_variants
            if v != "BASE_SLP" push!(intra_pairs, ("BASE_SLP", v)) end
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

    # INTER-METHOD COMPARISON (Best 2P vs Best SLP)
    m1_w, m1_t, m1_l, m2_w, m2_t, m2_l, m3_w, m3_t, m3_l, total_inter = zeros(Int, 10)
    inter_details = []

    for prob in unique(df_valid.Problem)
        df_prob = filter(r -> r.Problem == prob, df_valid)
        slps_valid = filter(r -> r.Variant in slp_variants && r.h_norm <= eps, df_prob)
        tps_valid  = filter(r -> r.Variant in tp_variants && r.h_norm <= eps, df_prob)
        
        if nrow(slps_valid) > 0 && nrow(tps_valid) > 0
            best_slp = sort(slps_valid, [:f_final, :Iterations])[1, :]
            best_2p = sort(tps_valid, [:f_final, :Iterations])[1, :]
            total_inter += 1
            
            res_m1 = evaluate_m1(best_2p.Iterations, best_slp.Iterations, best_2p.f_final, best_slp.f_final, best_2p.h_norm, best_slp.h_norm, eps)
            res_m1 == :Win ? m1_w += 1 : (res_m1 == :Tie ? m1_t += 1 : m1_l += 1)
            res_m2 = evaluate_m2(best_2p.f_final, best_slp.f_final, best_2p.h_norm, best_slp.h_norm, eps)
            res_m2 == :Win ? m2_w += 1 : (res_m2 == :Tie ? m2_t += 1 : m2_l += 1)
            res_m3 = evaluate_m3(best_2p.Time_ms, best_slp.Time_ms, best_2p.f_final, best_slp.f_final, best_2p.h_norm, best_slp.h_norm, eps)
            res_m3 == :Win ? m3_w += 1 : (res_m3 == :Tie ? m3_t += 1 : m3_l += 1)

            push!(inter_details, (Problem = prob, Best_UNIF = best_slp.Variant, UNIF_f_final = best_slp.f_final, UNIF_Iterations = best_slp.Iterations, Best_2P = best_2p.Variant, f_final_2P = best_2p.f_final, Iterations_2P = best_2p.Iterations, Result_M1_Iters = string(res_m1), Result_M2_F_Final = string(res_m2), Result_M3_Time = string(res_m3)))
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
    
    plots_dir = joinpath(out_dir, "Performance Profile Plots")
    mkpath(plots_dir)

    # ---------------------------------------------------------
    # GROUP 1: SLP - The Impact of APR and BQ
    # ---------------------------------------------------------
    g1 = get_existing([
        "BASE_SLP",             # APR1, BQ0
        "SLP_APR0_BQ0_ATR0",    # APR0, BQ0
        "SLP_APR1_BQ1_ATR0",    # APR1, BQ1
        "SLP_APR0_BQ1_ATR0"     # APR0, BQ1
    ])
    # GROUP 1
    plot_profile(df, g1, :Time_ms, "Group 1: UNIF - APR vs BQ (Time)", joinpath(plots_dir, "Profile_G1_UNIF_APR_BQ_Time.png"))

    # ---------------------------------------------------------
    # GROUP 2: SLP - The Impact of Anisotropy (ATR)
    # ---------------------------------------------------------
    g2 = get_existing([
        "SLP_APR1_BQ1_ATR0",
        "SLP_APR1_BQ1_ATR1",
        "SLP_APR0_BQ1_ATR0",
        "SLP_APR0_BQ1_ATR1"
    ])
    # GROUP 2
    plot_profile(df, g2, :Time_ms, "Group 2: UNIF - Anisotropy (Time)", joinpath(plots_dir, "Profile_G2_UNIF_ATR_Time.png"))
    
    # ---------------------------------------------------------
    # GROUP 3: Two-Phase - Ratio Strategies (APR & URU)
    # ---------------------------------------------------------
    g3 = get_existing([
        "BASE_2P",               # APR1, URU0 (Base teórica: usa Ared/Pred, sem URU)
        "2P_APR0_BQ0_ATR0_URU0", # APR0, URU0
        "2P_APR1_BQ0_ATR0_URU1", # APR1, URU1
        "2P_APR0_BQ0_ATR0_URU1"  # APR0, URU1
    ])
    plot_profile(df, g3, :Time_ms, "Group 3: 2P - Ratio Strategies (Time)", joinpath(plots_dir, "Profile_G3_2P_Ratio_Time.png"))
    
    # ---------------------------------------------------------
    # GROUP 4: Two-Phase - Step Shapes (BQ & ATR)
    # ---------------------------------------------------------
    g4 = get_existing([
        "2P_APR1_BQ0_ATR0_URU1",
        "2P_APR1_BQ1_ATR0_URU1",
        "2P_APR1_BQ1_ATR1_URU1"
    ])
    plot_profile(df, g4, :Time_ms, "Group 4: 2P - Step Shapes (Time)", joinpath(plots_dir, "Profile_G4_2P_Shapes_Time.png"))

    # ---------------------------------------------------------
    # GROUP 5: SLP SQP Hessian Battle
    # ---------------------------------------------------------
    slp_pure = filter(v -> occursin("SLP", v) && !occursin("SQP", v), all_variants)
    best_slp = get_best_in_class(slp_pure, summary)

    if !isnothing(best_slp)
        g5 = get_existing([
            best_slp,
            "$(best_slp)_SQP_IDENTITY",
            "$(best_slp)_SQP_SPECTRAL"
        ])
        # GROUP 5
        plot_profile(df, g5, :Time_ms, "Group 5: UNIF SQP Strategies (Time)", joinpath(plots_dir, "Profile_G5_UNIF_SQP_Time.png"))
    end # <-- END ADICIONADO AQUI

    # ---------------------------------------------------------
    # GROUP 6: 2P SQP Hessian Battle
    # ---------------------------------------------------------
    tp_pure = filter(v -> occursin("2P", v) && !occursin("SQP", v), all_variants)
    best_tp = get_best_in_class(tp_pure, summary)

    if !isnothing(best_tp)
        g6 = get_existing([
            best_tp,
            "$(best_tp)_SQP_IDENTITY",
            "$(best_tp)_SQP_SPECTRAL"
        ])
        plot_profile(df, g6, :Time_ms, "Group 6: 2P SQP Strategies (Time)", joinpath(plots_dir, "Profile_G6_2P_SQP_Time.png"))
    end

    # ---------------------------------------------------------
    # GROUP 7: THE GRAND FINALE
    # ---------------------------------------------------------
    slp_sqp  = filter(v -> occursin("SLP", v) && occursin("SQP", v), all_variants)
    tp_sqp   = filter(v -> occursin("2P", v) && occursin("SQP", v), all_variants)

    best_slp_sqp  = get_best_in_class(slp_sqp, summary)
    best_tp_sqp   = get_best_in_class(tp_sqp, summary)

    g7 = String[]
    isnothing(best_slp)      || push!(g7, best_slp)
    isnothing(best_slp_sqp)  || push!(g7, best_slp_sqp)
    isnothing(best_tp)       || push!(g7, best_tp)
    isnothing(best_tp_sqp)   || push!(g7, best_tp_sqp)

    plot_profile(df, g7, :Time_ms, "Best of Classes (Time)", joinpath(plots_dir, "Profile_G7_Finale_Time.png"))
    plot_profile(df, g7, :Iterations, "Best of Classes (Iters)", joinpath(plots_dir, "Profile_G7_Finale_Iters.png"))
end

function get_best_in_class(class_variants::Vector{String}, summary::DataFrame)
    if isempty(class_variants) return nothing end
    sub = summary[in.(summary.Variant, Ref(class_variants)), :]
    if nrow(sub) == 0 return nothing end
    return sub[1, :Variant]
end

"""
Builds a Dolan-Moré performance profile (step plot) for a specific metric and subset of solvers.
"""
function plot_profile(df::DataFrame, solvers::Vector{String}, metric_col::Symbol, title_str::String, filename::String)
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

        clean_label = replace(s, "SLP_" => "UNIF ", "2P_" => "2P ","BASE_SLP" => "UNIF APR1_BQ0_ATR0", "BASE_2P" => "2P APR1_BQ0_ATR0_URU0")

        plot!(p, x_vals, y_vals, linetype=:steppost, label=clean_label, linewidth=2.5)
    end

    savefig(p, filename)
    println("  > Saved: $(basename(filename))")
end

end # End of module