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

# ==============================================================================
# 1. SAR/PH CONFLICT RESOLUTION
# ==============================================================================
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
function run_statistical_analysis(results_dir::String)
    println("\n" * "=" ^ 80)
    println("📊 STARTING CONSOLIDATED ANALYSIS AND STATISTICS")
    println("=" ^ 80)

    slp_file = joinpath(results_dir, "benchmark_slp.csv")
    tp_file   = joinpath(results_dir, "benchmark_twophase.csv")

    println("Loading databases...")
    df_slp = isfile(slp_file) ? CSV.read(slp_file, DataFrame) : DataFrame()
    df_tp   = isfile(tp_file)   ? CSV.read(tp_file, DataFrame)   : DataFrame()

    if nrow(df_slp) == 0 && nrow(df_tp) == 0
        println("⚠️ No data found to process. Exiting analysis.")
        return
    end

    df_all = vcat(df_slp, df_tp, cols=:union)
    filter!(r -> !occursin("SQP", r.Variant), df_all)
    df_all = remove_redundant_sar_ph_variants(df_all)

    println("📈 Generating Dolan-Moré Performance Profiles...")
    generate_tournament_profiles(df_all, results_dir)
    println("✅ Execution and Analysis complete. Plots saved to: $results_dir")
end

# ==============================================================================
# 3. PERFORMANCE PROFILES LOGIC (DOLAN-MORÉ TOURNAMENTS)
# ==============================================================================
function find_best_variant(df::DataFrame, variants::Vector{String})
    isempty(variants) && return nothing
    
    stats = []
    for v in variants
        sub = df[df.Variant .== v, :]
        total = nrow(sub)
        total == 0 && continue
        
        # Avalia a viabilidade para critério de seleção do melhor representante
        feasible = sub[sub.h_norm .<= 1e-5, :]
        solved_count = nrow(feasible)
        mean_time = solved_count > 0 ? mean(feasible.Time_ms) : Inf
        
        push!(stats, (Variant=v, Solved=solved_count, Time=mean_time))
    end
    
    isempty(stats) && return nothing
    sort!(stats, by = x -> (-x.Solved, x.Time))
    return stats[1].Variant
end

function generate_tournament_profiles(df::DataFrame, out_dir::String)
    all_variants = unique(df.Variant)
    get_existing(wanted::Vector{String}) = intersect(wanted, all_variants)

    color_map = build_variant_color_map(all_variants)

    plots_dir = joinpath(out_dir, "Plots")
    mkpath(plots_dir)

    # ---------------------------------------------------------
    # GROUP 1: SLP - The Impact of SAR and PH
    # ---------------------------------------------------------
    g1 = get_existing([
        "BASE_SLP",          # 1. Baseline (SAR 1, PH 0)
        "SLP_SAR0_PH0_ATR0", # 2. Ablação do SAR (SAR 0, PH 0)
        "SLP_SAR1_PH1_ATR0"  # 3. Efeito da Heurística (PH 1, SAR irrelevante)
    ])
    plot_profile(df, g1, :Time_ms, "Group 1: UNIF - SAR vs PH", joinpath(plots_dir, "Profile_G1_SLP_SAR_PH.png"), color_map)

    # ---------------------------------------------------------
    # GROUP 2: SLP - The Impact of Anisotropy (ATR)
    # ---------------------------------------------------------
    g2 = get_existing([
        "SLP_SAR1_PH1_ATR0",
        "SLP_SAR1_PH1_ATR1",
        "SLP_SAR0_PH1_ATR0",
        "SLP_SAR0_PH1_ATR1"
    ])
    plot_profile(df, g2, :Time_ms, "Group 2: UNIF - Anisotropy", joinpath(plots_dir, "Profile_G2_SLP_ATR.png"), color_map)
    
    # ---------------------------------------------------------
    # GROUP 3: Two-Phase - Ratio Strategies (SAR & RRU)
    # ---------------------------------------------------------
    g3 = get_existing([
        "2P_SAR1_PH0_ATR0_RRU0", # SAR1, RRU0
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
    # GROUP 5: Best UNIF vs Best 2P (Linear Only)
    # ---------------------------------------------------------
    slp_pure = filter(v -> (occursin("SLP", v) || occursin("UNIF", v)) && !occursin("SQP", v), all_variants)
    tp_pure  = filter(v -> occursin("2P", v) && !occursin("SQP", v), all_variants)

    best_slp = find_best_variant(df, slp_pure)
    best_tp  = find_best_variant(df, tp_pure)

    g5_variants = ["SLP_SAR1_PH0_ATR0", "2P_SAR1_PH0_ATR0_RRU0"]
    isnothing(best_slp) || push!(g5_variants, best_slp)
    isnothing(best_tp)  || push!(g5_variants, best_tp)
    
    g5 = get_existing(unique(g5_variants))

    plot_profile(df, g5, :Time_ms, "Group 5: Best UNIF vs Best 2P", joinpath(plots_dir, "Profile_G5_Best_Linear.png"), color_map)
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
    f_tol = 1e-5
    h_tol = 1e-5

    for (i, p) in enumerate(problems)
        df_p = df[df.Problem .== p, :]
        
        # Etapa 1: Encontrar F_min entre os métodos viáveis
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

        # Etapa 2: Preencher a performance para os que atingiram F_min
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

        # Atualizado para lidar com as nomenclaturas SLP
        clean_label = replace(s, "SLP_" => "UNIF ", "UNIF_" => "UNIF ", "2P_" => "2P ", "BASE_SLP" => "UNIF SAR1_PH0_ATR0", "BASE_UNIF" => "UNIF SAR1_PH0_ATR0", "BASE_2P" => "2P SAR1_PH0_ATR0_RRU0")

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