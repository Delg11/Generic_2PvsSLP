# ==============================================================================
# SCRIPT DE GERAÇÃO DE VÍDEOS: ROSENBROCK (UNIF vs TWO-PHASE COM ATR SECANT_BOX)
# ==============================================================================

using Dates
using LinearAlgebra
using Printf
using NLPModels
using CUTEst
using Plots
gr()

include(normpath(joinpath(@__DIR__, "..", "src", "worker_env.jl")))
include(normpath(joinpath(@__DIR__, "..", "src", "Generic_Visualization.jl")))

using .Generic_Visualization

output_base = normpath(joinpath(@__DIR__, "..", "visualizations"))
mkpath(output_base)

println("\n" * "="^90)
println("🎬 GERADOR DE VÍDEOS DE OTIMIZAÇÃO: PROBLEMA ROSENBR")
println("Variantes: UNIF (ATR: secant_box) vs Two-Phase (ATR: secant_box)")
println("Destino: $output_base")
println("="^90)

nlp = CUTEstModel("ROSENBR")
prob = SharedTypes.build_optimization_problem(nlp)
x0 = clamp.(nlp.meta.x0, prob.xl, prob.xu)

# ------------------------------------------------------------------------------
# 1. Execução e Vídeo do UNIF
# ------------------------------------------------------------------------------
println("\n[1/2] 🚀 Executando UNIF (SLP + ATR secant_box)...")
unif_params = SharedTypes.UNIFParams(
    maxiter = 500,
    delta0 = 0.1,
    tolG = 1e-4,
    tolF = 1e-4,
    tolS = 1e-4,
    maxcount = 3,
    verbose = false,
    strong_agreement_rule = false,
    parabolic_heuristic = true,
    anisotropic_trust_region = true,
    atr_strategy = :secant_box,
    use_quadratic = false
)

t_unif = @elapsed begin
    y_un, λ_un, θ_un, iters_un, status_un, hist_un, log_un = Generic_module_unif.solve_unif_trust_region(
        prob, x0, unif_params; history=true
    )
end

status_un_str = status_un == 0 ? "KKT_OK" : (status_un == 1 ? "SMALL_STEP" : (status_un == 2 ? "MAX_IT" : "FAIL_$status_un"))
@printf("UNIF concluído em %.2f s | Iter: %d | Status: %s | f(x*): %.4e | Log steps: %d\n",
        t_unif, iters_un, status_un_str, prob.f(y_un), length(log_un))

unif_dir = joinpath(output_base, "rosenbr_unif_secant_box")
vid_unif_path = Generic_Visualization.visualizar_frames_3plots(
    prob, log_un;
    save_frames = true,
    save_video = true,
    fps_video = 12,
    hold_time_sec = 3,
    output_dir = unif_dir,
    video_name = "rosenbr_unif_secant_box.mp4",
    prob_name = "ROSENBR",
    algo_name = "UNIF (ATR: secant_box)",
    grid_density = 65,
    dpi = 120
)

# ------------------------------------------------------------------------------
# 2. Execução e Vídeo do Two-Phase
# ------------------------------------------------------------------------------
println("\n[2/2] 🚀 Executando Two-Phase (SLP + ATR secant_box)...")
tp_params = SharedTypes.TwoPhaseParams(
    max_outer_iter = 500,
    max_iter_opt = 500,
    δ0_opt = 0.1,
    δ0_resto = 0.1,
    tolG = 1e-4,
    tolF = 1e-4,
    tolS = 1e-4,
    maxcount = 3,
    verbose_out = false,
    verbose_in = false,
    debugverbose = false,
    use_unif_stopping = true,
    rfeas = 1e-12,
    δmin = 1e-12,
    δmax = 1e16,
    r_resto = 0.9,
    τ1 = 0.1,
    τ2 = 0.25,
    τ3 = 2.0,
    αL = 1e-8,
    αR = 1e-8,
    θ0 = 0.90,
    strong_agreement_rule = false,
    parabolic_heuristic = true,
    anisotropic_trust_region = true,
    atr_strategy = :secant_box,
    restoration_ratio_update = true,
    use_quadratic = false,
    non_monotone_M = 1
)

t_tp = @elapsed begin
    y_tp, λ_tp, θ_tp, iters_tp, status_tp, hist_tp, log_tp = Generic_module_Twophase.two_phase_optimization(
        prob, x0, tp_params; history=true
    )
end

status_tp_str = status_tp == 0 ? "KKT_OK" : (status_tp == 1 ? "SMALL_STEP" : (status_tp == 2 ? "MAX_IT" : "FAIL_$status_tp"))
@printf("Two-Phase concluído em %.2f s | Iter: %d | Status: %s | f(x*): %.4e | Log steps: %d\n",
        t_tp, iters_tp, status_tp_str, prob.f(y_tp), length(log_tp))

tp_dir = joinpath(output_base, "rosenbr_2p_secant_box")
vid_tp_path = Generic_Visualization.visualizar_frames_3plots(
    prob, log_tp;
    save_frames = true,
    save_video = true,
    fps_video = 12,
    hold_time_sec = 3,
    output_dir = tp_dir,
    video_name = "rosenbr_2p_secant_box.mp4",
    prob_name = "ROSENBR",
    algo_name = "Two-Phase (ATR: secant_box)",
    grid_density = 65,
    dpi = 120
)

finalize(nlp)

println("\n" * "="^90)
println("🎉 GERAÇÃO DE VÍDEOS CONCLUÍDA COM SUCESSO!")
println("📁 Vídeo UNIF:      $vid_unif_path ($(round(filesize(vid_unif_path) / 1024 / 1024, digits=2)) MB)")
println("📁 Vídeo Two-Phase: $vid_tp_path ($(round(filesize(vid_tp_path) / 1024 / 1024, digits=2)) MB)")
println("="^90)

