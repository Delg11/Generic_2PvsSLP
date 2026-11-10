# ==============================================================================
# generate_cutest_baseline_trajectories.jl
# ==============================================================================
# Generates iteration-by-iteration baseline trajectory plots for didactic CUTEst
# problems (comparing step norm ||s^k||_∞, objective value f(x^k), projected
# gradient norm ||g_P(x^k)||_∞, and constraint violation ||h(x^k)||_2), as well
# as comparative bar charts for Chapter 3 and the SBPO 2027 paper.
# ==============================================================================

if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "100"
end

using LinearAlgebra
using Printf
using Statistics
using Dates
using CUTEst
using DataFrames
using CSV
using Plots

include(normpath(joinpath(@__DIR__, "..", "src", "worker_env.jl")))

const DEFAULT_DIDACTIC_PROBLEMS = ["HS6", "TRY-B", "HS26", "ROSENBR", "SISSER", "BT1", "HS7", "HS8", "HS9", "MARATOS"]

function compute_trajectory_metrics(prob::SharedTypes.OptimizationProblem, xs::Vector{Vector{Float64}}; div_nelem::Bool=true)
    K = length(xs)
    iters = collect(0:(K - 1))
    f_vals = zeros(K)
    h_norms = zeros(K)
    gp_norms = zeros(K)
    step_norms = fill(NaN, K)

    λ_zero = zeros(prob.ncon)
    for k in 1:K
        xk = xs[k]
        f_vals[k] = try prob.f(xk) catch; NaN end
        hk = try prob.h(xk) catch; zeros(prob.ncon) end
        h_norms[k] = isempty(hk) ? 0.0 : norm(hk)

        gk = try prob.∇f(xk) catch; fill(NaN, prob.nvar) end
        Jk = try prob.∇h(xk) catch; zeros(prob.ncon, prob.nvar) end

        if any(!isfinite, gk) || any(!isfinite, Jk)
            gp_norms[k] = NaN
        else
            proj_step = SharedTypes.compute_gradient_projection(
                xk, gk, Jk, λ_zero, prob.xl, prob.xu;
                norm_type = Inf,
                strategy = :dykstra,
                max_iter = 500,
                tol = 1e-8,
                debug = false
            )
            gpn = norm(proj_step, Inf)
            if div_nelem && prob.nvar > 0
                gpn /= prob.nvar
            end
            gp_norms[k] = gpn
        end

        if k >= 2
            step_norms[k] = norm(xs[k] .- xs[k - 1], Inf)
        end
    end

    return (iters=iters, f_vals=f_vals, h_norms=h_norms, gp_norms=gp_norms, step_norms=step_norms)
end

function safe_positive_series(v::Vector{Float64}; floor_val::Float64=1e-16)
    out = similar(v)
    for i in eachindex(v)
        out[i] = isfinite(v[i]) ? max(abs(v[i]), floor_val) : NaN
    end
    return out
end

function build_cutest_single_plots_and_panels(prob_name::String, nvar::Int, ncon::Int, traj_dict::Dict{String, Any}, out_dir::String)
    safe_name = replace(prob_name, r"[^A-Za-z0-9_\-]" => "_")

    # Clean, stroke-free styling for both markers and legend
    common_panel = (
        linewidth = 2.4,
        markerstrokewidth = 0,
        markerstrokealpha = 0.0,
        framestyle = :box,
        left_margin = 10Plots.mm,
        bottom_margin = 8Plots.mm,
        top_margin = 6Plots.mm,
        right_margin = 6Plots.mm,
        titlefontsize = 11,
        guidefontsize = 10,
        tickfontsize = 9,
        legendfontsize = 9,
        foreground_color_legend = nothing,
        background_color_legend = RGBA(1.0, 1.0, 1.0, 0.80),
        grid = true,
        gridalpha = 0.25,
    )

    t_ub = traj_dict["UNIF_BASE"]
    t_2b = traj_dict["2P_BASE"]
    t_ui = traj_dict["UNIF_ISO"]
    t_2i = traj_dict["2P_ISO"]

    all_f = vcat(t_ub.f_vals, t_2b.f_vals, t_ui.f_vals, t_2i.f_vals)
    finite_f = filter(isfinite, all_f)
    f_min = isempty(finite_f) ? 0.0 : minimum(finite_f)

    # --------------------------------------------------------------------------
    # 1. Clean 1x3 Panel for BASE (UNIF_BASE vs 2P_BASE) + Individual Single Plots
    # --------------------------------------------------------------------------
    p_f_base = plot(t_ub.iters, t_ub.f_vals;
        title="Objective f(xᵏ) — $prob_name", xlabel="Iteration k", ylabel="f(xᵏ)",
        color=:steelblue, linestyle=:dash, marker=:circle, markersize=4,
        label="UNIF_BASE", legend=:topright, common_panel...)
    plot!(p_f_base, t_2b.iters, t_2b.f_vals;
        color=:firebrick, linestyle=:solid, marker=:diamond, markersize=4,
        markerstrokewidth=0, label="2P_BASE")

    gp_ub = safe_positive_series(t_ub.gp_norms)
    gp_2b = safe_positive_series(t_2b.gp_norms)
    p_g_base = plot(t_ub.iters, gp_ub;
        title="Projected Gradient ||gₚ(xᵏ)||∞ — $prob_name", xlabel="Iteration k", ylabel="||gₚ(xᵏ)||∞",
        yaxis=:log10, color=:steelblue, linestyle=:dash, marker=:circle, markersize=4,
        label="UNIF_BASE", legend=:topright, common_panel...)
    plot!(p_g_base, t_2b.iters, gp_2b;
        color=:firebrick, linestyle=:solid, marker=:diamond, markersize=4,
        markerstrokewidth=0, label="2P_BASE")
    hline!(p_g_base, [1e-3]; color=:gray30, linestyle=:dot, linewidth=1.3, label="tolG = 10⁻³")

    s_ub = safe_positive_series(t_ub.step_norms)
    s_2b = safe_positive_series(t_2b.step_norms)
    v_ub = .!isnan.(t_ub.step_norms)
    v_2b = .!isnan.(t_2b.step_norms)
    p_s_base = plot(;
        title="Step Norm ||sᵏ||∞ — $prob_name", xlabel="Iteration k", ylabel="||sᵏ||∞",
        yaxis=:log10, legend=:topright, common_panel...)
    any(v_ub) && plot!(p_s_base, t_ub.iters[v_ub], s_ub[v_ub];
        color=:steelblue, linestyle=:dash, marker=:circle, markersize=4,
        markerstrokewidth=0, label="UNIF_BASE")
    any(v_2b) && plot!(p_s_base, t_2b.iters[v_2b], s_2b[v_2b];
        color=:firebrick, linestyle=:solid, marker=:diamond, markersize=4,
        markerstrokewidth=0, label="2P_BASE")

    # Save standalone single-metric plots (large, uncluttered 1-graph figures)
    single_dir = joinpath(out_dir, "single_plots")
    isdir(single_dir) || mkpath(single_dir)
    for (suf, plt_obj) in [("Objective_BASE", p_f_base), ("Gradient_BASE", p_g_base), ("StepNorm_BASE", p_s_base)]
        p_single = plot(plt_obj; size=(820, 540), margin=8Plots.mm, titlefontsize=13, guidefontsize=11, tickfontsize=10, legendfontsize=10)
        savefig(p_single, joinpath(single_dir, "Single_$(safe_name)_$(suf).png"))
        savefig(p_single, joinpath(single_dir, "Single_$(safe_name)_$(suf).pdf"))
    end

    plt_base_1x3 = plot(
        p_f_base, p_g_base, p_s_base;
        layout = (1, 3),
        size = (1800, 560),
        plot_title = "$prob_name (n=$nvar, m=$ncon) — Baseline Comparison: Two-Phase vs UNIF",
        plot_titlefontsize = 13,
    )
    savefig(plt_base_1x3, joinpath(out_dir, "Baseline_Trajectory_$(safe_name)_BASE.png"))
    savefig(plt_base_1x3, joinpath(out_dir, "Baseline_Trajectory_$(safe_name)_BASE.pdf"))

    # --------------------------------------------------------------------------
    # 2. Clean 1x3 Panel for ISO (UNIF_ISO vs 2P_ISO)
    # --------------------------------------------------------------------------
    p_f_iso = plot(t_ui.iters, t_ui.f_vals;
        title="Objective f(xᵏ) — $prob_name (ISO)", xlabel="Iteration k", ylabel="f(xᵏ)",
        color=:dodgerblue4, linestyle=:dash, marker=:utriangle, markersize=4,
        label="UNIF_ISO", legend=:topright, common_panel...)
    plot!(p_f_iso, t_2i.iters, t_2i.f_vals;
        color=:darkgreen, linestyle=:solid, marker=:rect, markersize=4,
        markerstrokewidth=0, label="2P_ISO")

    gp_ui = safe_positive_series(t_ui.gp_norms)
    gp_2i = safe_positive_series(t_2i.gp_norms)
    p_g_iso = plot(t_ui.iters, gp_ui;
        title="Projected Gradient ||gₚ(xᵏ)||∞ — $prob_name (ISO)", xlabel="Iteration k", ylabel="||gₚ(xᵏ)||∞",
        yaxis=:log10, color=:dodgerblue4, linestyle=:dash, marker=:utriangle, markersize=4,
        label="UNIF_ISO", legend=:topright, common_panel...)
    plot!(p_g_iso, t_2i.iters, gp_2i;
        color=:darkgreen, linestyle=:solid, marker=:rect, markersize=4,
        markerstrokewidth=0, label="2P_ISO")
    hline!(p_g_iso, [1e-3]; color=:gray30, linestyle=:dot, linewidth=1.3, label="tolG = 10⁻³")

    s_ui = safe_positive_series(t_ui.step_norms)
    s_2i = safe_positive_series(t_2i.step_norms)
    v_ui = .!isnan.(t_ui.step_norms)
    v_2i = .!isnan.(t_2i.step_norms)
    p_s_iso = plot(;
        title="Step Norm ||sᵏ||∞ — $prob_name (ISO)", xlabel="Iteration k", ylabel="||sᵏ||∞",
        yaxis=:log10, legend=:topright, common_panel...)
    any(v_ui) && plot!(p_s_iso, t_ui.iters[v_ui], s_ui[v_ui];
        color=:dodgerblue4, linestyle=:dash, marker=:utriangle, markersize=4,
        markerstrokewidth=0, label="UNIF_ISO")
    any(v_2i) && plot!(p_s_iso, t_2i.iters[v_2i], s_2i[v_2i];
        color=:darkgreen, linestyle=:solid, marker=:rect, markersize=4,
        markerstrokewidth=0, label="2P_ISO")

    plt_iso_1x3 = plot(
        p_f_iso, p_g_iso, p_s_iso;
        layout = (1, 3),
        size = (1800, 560),
        plot_title = "$prob_name (n=$nvar, m=$ncon) — Isotropic Parabolic Heuristic: Two-Phase vs UNIF",
        plot_titlefontsize = 13,
    )
    savefig(plt_iso_1x3, joinpath(out_dir, "Baseline_Trajectory_$(safe_name)_ISO.png"))
    savefig(plt_iso_1x3, joinpath(out_dir, "Baseline_Trajectory_$(safe_name)_ISO.pdf"))

    # --------------------------------------------------------------------------
    # 3. Main 2x2 Panel (4 spacious subplots instead of 9, no marker stroke)
    # --------------------------------------------------------------------------
    p_4_f = plot(t_ub.iters, t_ub.f_vals;
        title="Objective Value f(xᵏ)", xlabel="Iteration k", ylabel="f(xᵏ)",
        color=:steelblue, linestyle=:dash, marker=:circle, markersize=3.5,
        label="UNIF_BASE", legend=:topright, common_panel...)
    plot!(p_4_f, t_2b.iters, t_2b.f_vals; color=:firebrick, linestyle=:solid, marker=:diamond, markersize=3.5, markerstrokewidth=0, label="2P_BASE")
    plot!(p_4_f, t_ui.iters, t_ui.f_vals; color=:dodgerblue4, linestyle=:dot, marker=:utriangle, markersize=3.5, markerstrokewidth=0, label="UNIF_ISO")
    plot!(p_4_f, t_2i.iters, t_2i.f_vals; color=:darkgreen, linestyle=:solid, marker=:rect, markersize=3.5, markerstrokewidth=0, label="2P_ISO")

    p_4_g = plot(t_ub.iters, gp_ub;
        title="Projected Gradient Norm ||gₚ(xᵏ)||∞", xlabel="Iteration k", ylabel="||gₚ(xᵏ)||∞",
        yaxis=:log10, color=:steelblue, linestyle=:dash, marker=:circle, markersize=3.5,
        label="UNIF_BASE", legend=:topright, common_panel...)
    plot!(p_4_g, t_2b.iters, gp_2b; color=:firebrick, linestyle=:solid, marker=:diamond, markersize=3.5, markerstrokewidth=0, label="2P_BASE")
    plot!(p_4_g, t_ui.iters, gp_ui; color=:dodgerblue4, linestyle=:dot, marker=:utriangle, markersize=3.5, markerstrokewidth=0, label="UNIF_ISO")
    plot!(p_4_g, t_2i.iters, gp_2i; color=:darkgreen, linestyle=:solid, marker=:rect, markersize=3.5, markerstrokewidth=0, label="2P_ISO")
    hline!(p_4_g, [1e-3]; color=:gray30, linestyle=:dot, linewidth=1.3, label="tolG = 10⁻³")

    p_4_s = plot(;
        title="Step Norm ||sᵏ||∞", xlabel="Iteration k", ylabel="||sᵏ||∞",
        yaxis=:log10, legend=:topright, common_panel...)
    any(v_ub) && plot!(p_4_s, t_ub.iters[v_ub], s_ub[v_ub]; color=:steelblue, linestyle=:dash, marker=:circle, markersize=3.5, markerstrokewidth=0, label="UNIF_BASE")
    any(v_2b) && plot!(p_4_s, t_2b.iters[v_2b], s_2b[v_2b]; color=:firebrick, linestyle=:solid, marker=:diamond, markersize=3.5, markerstrokewidth=0, label="2P_BASE")
    any(v_ui) && plot!(p_4_s, t_ui.iters[v_ui], s_ui[v_ui]; color=:dodgerblue4, linestyle=:dot, marker=:utriangle, markersize=3.5, markerstrokewidth=0, label="UNIF_ISO")
    any(v_2i) && plot!(p_4_s, t_2i.iters[v_2i], s_2i[v_2i]; color=:darkgreen, linestyle=:solid, marker=:rect, markersize=3.5, markerstrokewidth=0, label="2P_ISO")

    if ncon > 0
        h_ub = safe_positive_series(t_ub.h_norms; floor_val=1e-16)
        h_2b = safe_positive_series(t_2b.h_norms; floor_val=1e-16)
        h_ui = safe_positive_series(t_ui.h_norms; floor_val=1e-16)
        h_2i = safe_positive_series(t_2i.h_norms; floor_val=1e-16)
        p_4_fourth = plot(t_ub.iters, h_ub;
            title="Constraint Infeasibility ||h(xᵏ)||₂", xlabel="Iteration k", ylabel="||h(xᵏ)||₂",
            yaxis=:log10, color=:steelblue, linestyle=:dash, marker=:circle, markersize=3.5,
            label="UNIF_BASE", legend=:topright, common_panel...)
        plot!(p_4_fourth, t_2b.iters, h_2b; color=:firebrick, linestyle=:solid, marker=:diamond, markersize=3.5, markerstrokewidth=0, label="2P_BASE")
        plot!(p_4_fourth, t_ui.iters, h_ui; color=:dodgerblue4, linestyle=:dot, marker=:utriangle, markersize=3.5, markerstrokewidth=0, label="UNIF_ISO")
        plot!(p_4_fourth, t_2i.iters, h_2i; color=:darkgreen, linestyle=:solid, marker=:rect, markersize=3.5, markerstrokewidth=0, label="2P_ISO")
        hline!(p_4_fourth, [1e-3]; color=:gray30, linestyle=:dot, linewidth=1.3, label="tol = 10⁻³")
    else
        df_ub = safe_positive_series(t_ub.f_vals .- f_min .+ 1e-14; floor_val=1e-14)
        df_2b = safe_positive_series(t_2b.f_vals .- f_min .+ 1e-14; floor_val=1e-14)
        df_ui = safe_positive_series(t_ui.f_vals .- f_min .+ 1e-14; floor_val=1e-14)
        df_2i = safe_positive_series(t_2i.f_vals .- f_min .+ 1e-14; floor_val=1e-14)
        p_4_fourth = plot(t_ub.iters, df_ub;
            title="Optimality Gap |f(xᵏ) - f*| + 10⁻¹⁴", xlabel="Iteration k", ylabel="|f(xᵏ) - f*|",
            yaxis=:log10, color=:steelblue, linestyle=:dash, marker=:circle, markersize=3.5,
            label="UNIF_BASE", legend=:topright, common_panel...)
        plot!(p_4_fourth, t_2b.iters, df_2b; color=:firebrick, linestyle=:solid, marker=:diamond, markersize=3.5, markerstrokewidth=0, label="2P_BASE")
        plot!(p_4_fourth, t_ui.iters, df_ui; color=:dodgerblue4, linestyle=:dot, marker=:utriangle, markersize=3.5, markerstrokewidth=0, label="UNIF_ISO")
        plot!(p_4_fourth, t_2i.iters, df_2i; color=:darkgreen, linestyle=:solid, marker=:rect, markersize=3.5, markerstrokewidth=0, label="2P_ISO")
    end

    plt_2x2 = plot(
        p_4_f, p_4_g,
        p_4_s, p_4_fourth;
        layout = (2, 2),
        size = (1600, 1050),
        plot_title = "CUTEst Trajectory Comparison: $prob_name (n=$nvar, m=$ncon) — Two-Phase vs UNIF",
        plot_titlefontsize = 14,
    )
    savefig(plt_2x2, joinpath(out_dir, "Baseline_Trajectory_$(safe_name).png"))
    savefig(plt_2x2, joinpath(out_dir, "Baseline_Trajectory_$(safe_name).pdf"))
    return plt_2x2
end

function generate_didactic_bar_charts(df::DataFrame, out_dir::String)
    probs = unique(df.Problem)
    variants = ["2P_BASE", "UNIF_BASE", "2P_ISO", "UNIF_ISO"]

    n_p = length(probs)

    p_iter = plot(; title="CUTEst Benchmark Problems: Outer Iterations",
                  xlabel="Problem", ylabel="Iterations", framestyle=:box, size=(1100, 540),
                  xticks=(1:n_p, probs), legend=:topright,
                  foreground_color_legend=nothing, background_color_legend=RGBA(1.0, 1.0, 1.0, 0.85),
                  titlefontsize=13, guidefontsize=11, tickfontsize=10, legendfontsize=10,
                  margin=7Plots.mm, grid=true, gridalpha=0.25)

    p_time = plot(; title="CUTEst Benchmark Problems: Execution Time (ms)",
                  xlabel="Problem", ylabel="Time (ms)", framestyle=:box, size=(1100, 540),
                  xticks=(1:n_p, probs), legend=:topright,
                  foreground_color_legend=nothing, background_color_legend=RGBA(1.0, 1.0, 1.0, 0.85),
                  titlefontsize=13, guidefontsize=11, tickfontsize=10, legendfontsize=10,
                  margin=7Plots.mm, grid=true, gridalpha=0.25)

    bar_width = 0.18
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]
    v_colors = [:firebrick, :steelblue, :darkgreen, :dodgerblue4]

    for (vi, vname) in enumerate(variants)
        iters_v = Float64[]
        times_v = Float64[]
        for p in probs
            sub = df[(df.Problem .== p) .& (df.Variant .== vname), :]
            push!(iters_v, isempty(sub) ? 0.0 : Float64(sub.Iterations[1]))
            push!(times_v, isempty(sub) ? 0.0 : Float64(sub.Time_ms[1]))
        end
        x_pos = (1:n_p) .+ offsets[vi]
        bar!(p_iter, x_pos, iters_v; bar_width=bar_width, color=v_colors[vi], label=vname, linecolor=:transparent, markerstrokewidth=0)
        bar!(p_time, x_pos, times_v; bar_width=bar_width, color=v_colors[vi], label=vname, linecolor=:transparent, markerstrokewidth=0)
    end

    savefig(p_iter, joinpath(out_dir, "BarChart_Cap3_Didactic_Iterations.png"))
    savefig(p_iter, joinpath(out_dir, "BarChart_Cap3_Didactic_Iterations.pdf"))
    savefig(p_time, joinpath(out_dir, "BarChart_Cap3_Didactic_Time_ms.png"))
    savefig(p_time, joinpath(out_dir, "BarChart_Cap3_Didactic_Time_ms.pdf"))
    savefig(p_iter, joinpath(out_dir, "BarChart_Cap3_Iterations.png"))
    savefig(p_iter, joinpath(out_dir, "BarChart_Cap3_Iterations.pdf"))
    savefig(p_time, joinpath(out_dir, "BarChart_Cap3_Time_ms.png"))
    savefig(p_time, joinpath(out_dir, "BarChart_Cap3_Time_ms.pdf"))
end

function generate_global_summary_bar_charts(summary_csv_path::String, out_dir::String)
    isfile(summary_csv_path) || return
    df_sum = CSV.read(summary_csv_path, DataFrame)
    isempty(df_sum) && return

    v_names = String.(df_sum.Variant)
    n_v = length(v_names)
    bar_colors = [startswith(v, "2P") || occursin("2P", v) ? :firebrick : :steelblue for v in v_names]

    if "Success_Rate_Pct" in names(df_sum)
        p_sr = bar(1:n_v, df_sum.Success_Rate_Pct;
            title="Cap 3 Global Benchmark: Convergence Rate (%)",
            xlabel="Variant", ylabel="Success Rate (%)",
            xticks=(1:n_v, v_names), xrotation=30,
            color=bar_colors, legend=false, framestyle=:box,
            size=(1100, 550), margin=8Plots.mm, ylim=(0, 105), grid=true)
        savefig(p_sr, joinpath(out_dir, "BarChart_Cap3_Global_SuccessRate.png"))
        savefig(p_sr, joinpath(out_dir, "BarChart_Cap3_Global_SuccessRate.pdf"))
    end

    if "Median_Iters" in names(df_sum)
        p_mi = bar(1:n_v, df_sum.Median_Iters;
            title="Cap 3 Global Benchmark: Median Iterations (Converged)",
            xlabel="Variant", ylabel="Median Iterations",
            xticks=(1:n_v, v_names), xrotation=30,
            color=bar_colors, legend=false, framestyle=:box,
            size=(1100, 550), margin=8Plots.mm, grid=true)
        savefig(p_mi, joinpath(out_dir, "BarChart_Cap3_Global_MedianIters.png"))
        savefig(p_mi, joinpath(out_dir, "BarChart_Cap3_Global_MedianIters.pdf"))
    end

    if "Median_Time_ms" in names(df_sum)
        p_mt = bar(1:n_v, df_sum.Median_Time_ms;
            title="Cap 3 Global Benchmark: Median Time in ms (Converged)",
            xlabel="Variant", ylabel="Median Time (ms)",
            xticks=(1:n_v, v_names), xrotation=30,
            color=bar_colors, legend=false, framestyle=:box,
            size=(1100, 550), margin=8Plots.mm, grid=true)
        savefig(p_mt, joinpath(out_dir, "BarChart_Cap3_Global_MedianTime.png"))
        savefig(p_mt, joinpath(out_dir, "BarChart_Cap3_Global_MedianTime.pdf"))
    end
end

function run_cutest_baseline_trajectories(;
    problems::Vector{String}=DEFAULT_DIDACTIC_PROBLEMS,
    max_iter::Int=500,
    out_dir::String=normpath(joinpath(@__DIR__, "..", "results", "baseline_didactic")),
    summary_csv_path::String=""
)
    isdir(out_dir) || mkpath(out_dir)

    delta0   = 0.20
    tolG     = 1e-3
    tolF     = 5e-2
    tolS     = 1e-4
    maxcount = 3

    tp_base = SharedTypes.TwoPhaseParams(
        max_outer_iter = max_iter, δ0_opt = delta0, δ0_resto = delta0,
        tol = 1e-3, tolG = tolG, tolF = tolF, tolS = tolS, maxcount = maxcount,
        verbose_out = false, verbose_in = false, debugverbose = false,
        use_unif_stopping = true, rfeas = 1e-12, δmin = 1e-4, δmax = 1.0,
        r_resto = 0.8, τ1 = 0.25, τ2 = 0.50, τ3 = 0.40, αL = 1e-8, αR = 1e-8,
        θ0 = 0.90, strong_agreement_rule = true, parabolic_heuristic = false,
        anisotropic_trust_region = false, restoration_ratio_update = false,
        use_quadratic = false, non_monotone_M = 1
    )

    tp_iso = SharedTypes.TwoPhaseParams(
        max_outer_iter = max_iter, δ0_opt = delta0, δ0_resto = delta0,
        tol = 1e-3, tolG = tolG, tolF = tolF, tolS = tolS, maxcount = maxcount,
        verbose_out = false, verbose_in = false, debugverbose = false,
        use_unif_stopping = true, rfeas = 1e-12, δmin = 1e-4, δmax = 1.0,
        r_resto = 0.8, τ1 = 0.25, τ2 = 0.50, τ3 = 0.40, αL = 1e-8, αR = 1e-8,
        θ0 = 0.90, strong_agreement_rule = true, parabolic_heuristic = true,
        parabolic_min_increase_ratio = 1.5,
        anisotropic_trust_region = false, restoration_ratio_update = false,
        use_quadratic = false, non_monotone_M = 1
    )

    unif_base = SharedTypes.UNIFParams(
        maxiter = max_iter, delta0 = 0.10, tolG = tolG, tolF = tolF, tolS = 1e-4,
        maxcount = maxcount, verbose = false,
        strong_agreement_rule = true, parabolic_heuristic = false,
        anisotropic_trust_region = false, use_quadratic = false
    )

    unif_iso = SharedTypes.UNIFParams(
        maxiter = max_iter, delta0 = 0.10, tolG = tolG, tolF = tolF, tolS = 1e-4,
        maxcount = maxcount, verbose = false,
        strong_agreement_rule = false, parabolic_heuristic = true,
        anisotropic_trust_region = false, use_quadratic = false
    )

    configs = [
        ("2P_BASE",   :twophase, tp_base),
        ("2P_ISO",    :twophase, tp_iso),
        ("UNIF_BASE", :unif,     unif_base),
        ("UNIF_ISO",  :unif,     unif_iso)
    ]

    df_summary = DataFrame(
        Problem = String[], nvar = Int[], ncon = Int[], Variant = String[],
        Status = String[], Iterations = Int[], Time_ms = Float64[],
        f_final = Float64[], h_norm = Float64[], gpnorm_final = Float64[]
    )

    # Warm-up JIT on first problem
    if !isempty(problems)
        try
            nlp_w = CUTEstModel{Float64}(problems[1]; decode=true)
            prob_w = SharedTypes.build_optimization_problem(nlp_w)
            x0_w = clamp.(nlp_w.meta.x0, prob_w.xl, prob_w.xu)
            Generic_module_Twophase.two_phase_optimization(prob_w, x0_w, tp_base; history=true)
            Generic_module_unif.solve_unif_trust_region(prob_w, x0_w, unif_base; history=true)
            finalize(nlp_w)
        catch
        end
    end

    for prob_name in problems
        println("📊 Generating trajectory baseline for CUTEst problem: $prob_name")
        local nlp = nothing
        try
            nlp = CUTEstModel{Float64}(prob_name; decode=true)
            prob = SharedTypes.build_optimization_problem(nlp)
            x0 = clamp.(nlp.meta.x0, prob.xl, prob.xu)
            nvar, ncon = prob.nvar, prob.ncon

            traj_dict = Dict{String, Any}()

            for (vname, kind, params) in configs
                if kind == :twophase
                    t_sec = @elapsed begin
                        y, λ, θ, iters, status_code, x_hist, _ = Generic_module_Twophase.two_phase_optimization(prob, x0, params; history=true)
                    end
                    xs = x_hist === nothing || isempty(x_hist) ? [copy(x0), copy(y)] : x_hist
                    t_metrics = compute_trajectory_metrics(prob, xs; div_nelem=true)
                    traj_dict[vname] = t_metrics
                    st_str = get_status_name(status_code)
                    push!(df_summary, (prob_name, nvar, ncon, vname, st_str, iters, t_sec * 1000.0, t_metrics.f_vals[end], t_metrics.h_norms[end], t_metrics.gp_norms[end]))
                else
                    t_sec = @elapsed begin
                        y, λ, θ, iters, status_code, x_hist, _ = Generic_module_unif.solve_unif_trust_region(prob, x0, params; history=true)
                    end
                    xs = x_hist === nothing || isempty(x_hist) ? [copy(x0), copy(y)] : x_hist
                    t_metrics = compute_trajectory_metrics(prob, xs; div_nelem=false)
                    traj_dict[vname] = t_metrics
                    st_str = get_status_name(status_code)
                    push!(df_summary, (prob_name, nvar, ncon, vname, st_str, iters, t_sec * 1000.0, t_metrics.f_vals[end], t_metrics.h_norms[end], t_metrics.gp_norms[end]))
                end
            end

            build_cutest_single_plots_and_panels(prob_name, nvar, ncon, traj_dict, out_dir)
            safe_name = replace(prob_name, r"[^A-Za-z0-9_\-]" => "_")

            # Save per-iteration CSV for LaTeX/pgfplots inspection if desired
            df_traj = DataFrame(Variant=String[], Iteration=Int[], f_x=Float64[], gpnorm=Float64[], step_norm=Float64[], h_norm=Float64[])
            for (vname, _, _) in configs
                tm = traj_dict[vname]
                for idx in eachindex(tm.iters)
                    push!(df_traj, (vname, tm.iters[idx], tm.f_vals[idx], tm.gp_norms[idx], tm.step_norms[idx], tm.h_norms[idx]))
                end
            end
            CSV.write(joinpath(out_dir, "Trajectory_$(safe_name).csv"), df_traj)
        catch err
            println("⚠️ Failed on $prob_name: $err")
        finally
            isnothing(nlp) || finalize(nlp)
        end
    end

    CSV.write(joinpath(out_dir, "Baseline_Didactic_Summary.csv"), df_summary)
    if !isempty(df_summary)
        generate_didactic_bar_charts(df_summary, out_dir)
    end
    if !isempty(summary_csv_path)
        generate_global_summary_bar_charts(summary_csv_path, out_dir)
    end

    println("✅ Didactic CUTEst trajectories and bar charts saved in: $out_dir")
    return df_summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    let probs = copy(DEFAULT_DIDACTIC_PROBLEMS),
        out_dir = normpath(joinpath(@__DIR__, "..", "results", "baseline_didactic")),
        summary_csv = "",
        i = 1
        while i <= length(ARGS)
            if ARGS[i] == "--problems" && i < length(ARGS)
                probs = String.(strip.(split(ARGS[i + 1], ",")))
                i += 2
            elseif ARGS[i] == "--out-dir" && i < length(ARGS)
                out_dir = ARGS[i + 1]
                i += 2
            elseif ARGS[i] == "--summary-csv" && i < length(ARGS)
                summary_csv = ARGS[i + 1]
                i += 2
            else
                i += 1
            end
        end
        run_cutest_baseline_trajectories(; problems=probs, out_dir=out_dir, summary_csv_path=summary_csv)
    end
end
