module Generic_module_Stats

export run_statistical_analysis

using CSV
using DataFrames
using Statistics
using XLSX
using Plots

# ==============================================================================
# 1. AUXILIARY METRIC FUNCTIONS
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
# 2. DYNAMIC LEGEND FORMATTER
# ==============================================================================
# Translates complex variant names into clean, readable labels for plots
function format_variant_name(v::AbstractString)
    s = v
    s = replace(s, "BASE_SLP" => "SLP Base")
    s = replace(s, "BASE_2P" => "2P Base")
    s = replace(s, "2P_M1" => "2P")
    s = replace(s, "_BQ1_ATR0" => " Iso")
    s = replace(s, "_BQ1_ATR1" => " Aniso")
    s = replace(s, "_BQ0_ATR0" => " Standard")
    s = replace(s, "_URU1" => " URU")
    s = replace(s, "_URU0" => "")
    s = replace(s, "_SQP_" => " SQP-")
    return strip(s)
end

# ==============================================================================
# 3. PLOTTING FUNCTION
# ==============================================================================

function plot_performance_profile_fbest(df, variants_to_compare, title_suffix; 
                                        metric=:Time_ms, max_tau=10.0, 
                                        eps_h=1e-3, eps_f=1e-3, 
                                        strict_intersection=true)
    df_work = filter(r -> r.Variant in variants_to_compare, df)
    if nrow(df_work) == 0 return nothing end
    
    df_work[!, :Viable] = [!ismissing(r.h_norm) && r.h_norm <= eps_h for r in eachrow(df_work)]
    
    gdf_prob = groupby(df_work, :Problem)
    valid_problems = String[]
    
    for sub in gdf_prob
        if strict_intersection
            has_all_viable = true
            for v in variants_to_compare
                idx = findfirst(r -> r.Variant == v, eachrow(sub))
                if isnothing(idx) || !sub[idx, :Viable]
                    has_all_viable = false
                    break
                end
            end
            if has_all_viable push!(valid_problems, sub.Problem[1]) end
        else
            if any(sub.Viable) push!(valid_problems, sub.Problem[1]) end
        end
    end
    
    filter!(r -> r.Problem in valid_problems, df_work)
    if nrow(df_work) == 0 return nothing end

    df_feasible = filter(r -> r.Viable, df_work)
    if nrow(df_feasible) == 0 return nothing end # Guard against no feasible points

    f_bests = combine(groupby(df_feasible, :Problem), :f_final => minimum => :f_best)
    df_work = leftjoin(df_work, f_bests, on=:Problem)
    
    df_work[!, :Solved] = [!ismissing(r.f_best) && r.Viable && !ismissing(r.f_final) && r.f_final <= r.f_best + eps_f for r in eachrow(df_work)]
    df_work[!, :Metric] = [r.Solved ? Float64(r[metric]) : Inf for r in eachrow(df_work)]
    
    gdf = groupby(df_work, :Problem)
    min_metrics = Dict(k.Problem => minimum(v.Metric) for (k, v) in pairs(gdf))
    
    df_work[!, :Ratio] = [min_metrics[r.Problem] == Inf ? Inf : r.Metric / min_metrics[r.Problem] for r in eachrow(df_work)]
    
    metric_label = metric == :Iterations ? "Iterations" : (metric == :Time_ms ? "Time (ms)" : string(metric))
    num_problems = length(keys(min_metrics))

    p = plot(title="Performance Profile\n$title_suffix ($num_problems problems)\n", 
             xlabel="Factor (τ) relative to best $metric_label", ylabel="Proportion of solved problems",
             xlims=(1.0, max_tau), ylims=(0.0, 1.05), framestyle=:box,
             size = (1200, 800),           # Aumenta a área total do gráfico (largura x altura)
             legend = :outertopright,      # Joga a legenda para o lado de fora, à direita
             legend_columns = 2,           # Divide a lista em 2 colunas para não ficar tão alta
             legendfontsize = 8,           # Reduz levemente a fonte da legenda (ajuste conforme necessário)
             bottom_margin = 5Plots.mm,    # Adiciona respiro nas bordas (opcional)
             right_margin = 15Plots.mm)
    
    taus = range(1.0, stop=max_tau, length=500)
    
    for var in variants_to_compare
        df_var = filter(row -> row.Variant == var, df_work)
        rho = [sum(df_var.Ratio .<= t) / num_problems for t in taus]
        plot!(p, taus, rho, label=format_variant_name(var), linewidth=2)
    end
    
    return p
end

# ==============================================================================
# 4. MAIN STATISTICAL ANALYSIS ROUTINE
# ==============================================================================

function run_statistical_analysis(results_dir::String)
    println("\n" * "=" ^ 80)
    println("📊 STARTING CONSOLIDATED ANALYSIS AND STATISTICS")
    println("=" ^ 80)

    file_twophase = joinpath(results_dir, "partial_backup_twophase.csv")
    file_slp = joinpath(results_dir, "partial_backup_slp.csv")

    println("Loading databases...")
    df_twophase = isfile(file_twophase) ? CSV.read(file_twophase, DataFrame) : DataFrame()
    df_slp = isfile(file_slp) ? CSV.read(file_slp, DataFrame) : DataFrame()

    if nrow(df_twophase) == 0 && nrow(df_slp) == 0
        println("⚠️ No data found to process. Exiting analysis.")
        return
    end

    # Merge DataFrames
    df_all = vcat(df_slp, df_twophase, cols=:union)

    # Filter invalid results (NaN or Inf)
    invalid_mask = isnan.(df_all.f_final) .| isinf.(df_all.f_final) .| isnan.(df_all.h_norm) .| isinf.(df_all.h_norm)
    df_valid = df_all[.!invalid_mask, :]

    # Extract all variants that were actually executed
    all_variants = unique(df_valid.Variant)
    slp_variants = filter(v -> startswith(v, "SLP") || v == "BASE_SLP", all_variants)
    tp_variants  = filter(v -> startswith(v, "2P") || v == "BASE_2P", all_variants)

    # ==========================================================================
    # SCENARIO TABLES
    # ==========================================================================
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
    # INTRA-METHOD COMPARISON (Dynamic)
    # ==========================================================================
    intra_pairs = []
    
    # Pair every SLP variant against the Base SLP
    if "BASE_SLP" in slp_variants
        for v in slp_variants
            if v != "BASE_SLP" push!(intra_pairs, ("BASE_SLP", v)) end
        end
    end
    
    # Pair every 2P variant against the Base 2P
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

    # ==========================================================================
    # INTER-METHOD COMPARISON (Best 2P vs Best SLP)
    # ==========================================================================
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

            push!(inter_details, (Problem = prob, Best_SLP = best_slp.Variant, SLP_f_final = best_slp.f_final, SLP_Iterations = best_slp.Iterations, Best_2P = best_2p.Variant, f_final_2P = best_2p.f_final, Iterations_2P = best_2p.Iterations, Result_M1_Iters = string(res_m1), Result_M2_F_Final = string(res_m2), Result_M3_Time = string(res_m3)))
        end
    end

    df_res_inter = DataFrame(Comparison=["Best 2P vs Best SLP"], Total=[total_inter], M1_W=[m1_w], M1_T=[m1_t], M1_L=[m1_l], M2_W=[m2_w], M2_T=[m2_t], M2_L=[m2_l], M3_W=[m3_w], M3_T=[m3_t], M3_L=[m3_l])
    df_inter_details = DataFrame(inter_details)

    # ==========================================================================
    # DYNAMIC PLOT GENERATION
    # ==========================================================================
    plots_dir = joinpath(results_dir, "Plots_Profiles")
    mkpath(plots_dir) 

    # 1. Main Broad Groups
    comparison_groups = [
        ("All_Variants", all_variants),
        ("All_SLP", slp_variants),
        ("All_2P", tp_variants)
    ]

    for (name, vars) in comparison_groups
        if length(vars) < 2 continue end
        p = plot_performance_profile_fbest(df_valid, vars, replace(name, "_" => " "), metric=:Time_ms, strict_intersection=false)
        if !isnothing(p) 
            savefig(p, joinpath(plots_dir, "Group_$(name)_Time.png")) 
        end
    end

    # 2. Dynamic Duels (Smart Matching)
    duels = []
    
    # Duel 1: Base SLP vs Base 2P
    if "BASE_SLP" in all_variants && "BASE_2P" in all_variants
        push!(duels, ("BASE_SLP_vs_BASE_2P", ["BASE_SLP", "BASE_2P"]))
    end

    # Duel 2: LP/QP Core vs SQP Extensions
    # This automatically matches e.g. "SLP_BQ1_ATR0" against ["SLP_BQ1_ATR0_SQP_IDENTITY", "SLP_BQ1_ATR0_SQP_EXACT"]
    for core_v in all_variants
        if !contains(core_v, "SQP") && core_v != "BASE_SLP" && core_v != "BASE_2P"
            # Find all SQP variants that branch from this core variant
            sqp_versions = filter(x -> startswith(x, core_v * "_SQP"), all_variants)
            if !isempty(sqp_versions)
                group = vcat([core_v], sqp_versions)
                push!(duels, ("SQP_Impact_on_" * core_v, group))
            end
        end
    end

    # Generate the Duel Plots
    for (name, vars) in duels
        # Only plot if we have at least 2 variants
        if length(vars) < 2 continue end
        
        safe_name = replace(name, " " => "_", "(" => "", ")" => "")
        p_t = plot_performance_profile_fbest(df_valid, vars, format_variant_name(name), metric=:Time_ms, strict_intersection=false)
        if !isnothing(p_t) 
            savefig(p_t, joinpath(plots_dir, "Duel_$(safe_name)_Time.png")) 
        end
    end

    # ==========================================================================
    # XLSX EXPORT
    # ==========================================================================
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

    println("✅ Execution and Analysis complete. Results and plots saved to: $results_dir")
end

end # End of module