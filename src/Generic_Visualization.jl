module Generic_Visualization

export visualizar_frames_3plots, visualizar_frames_dinamicos, visualize_optimization

# ==============================================================================
# VISUALIZAÇÃO E ANIMAÇÃO DE OTIMIZAÇÃO (GENERIC VISUALIZATION)
# ==============================================================================
# Módulo de alta performance para renderizar frames e vídeos 2D / 1D de
# algoritmos de otimização (Two-Phase e UNIF) com suporte avançado a regiões
# de confiança anisotrópicas (ATR), cortes 1D exatos e compilação direta via FFMPEG.
# ==============================================================================

using Dates
using LinearAlgebra
using Plots
using Printf
using FFMPEG

const DEFAULT_OUTPUT_ROOT = "Outputs_Otimizacao"

# ==============================================================================
# FUNÇÕES AUXILIARES MATEMÁTICAS E DE RENDERIZAÇÃO
# ==============================================================================

"""
    _extrair_delta(step) -> (Float64, Float64, Float64)

Extrai os componentes de raio da região de confiança (dx, dy) e o valor escalar representativo.
Compatível com raios escalares e vetoriais anisotrópicos (ATR).
"""
function _extrair_delta(step)
    if hasproperty(step, :delta)
        d = step.delta
        if d isa AbstractVector
            n = length(d)
            dx = n >= 1 ? Float64(d[1]) : 0.1
            dy = n >= 2 ? Float64(d[2]) : dx
            return dx, dy, Float64(maximum(d))
        elseif d isa Real
            val = Float64(d)
            return val, val, val
        end
    end
    return 0.1, 0.1, 0.1
end

"""
    _calcular_fronteira_atr(delta_x::Float64, delta_y::Float64, d::AbstractVector{<:Real}) -> Float64

Calcula a distância exata ao longo da direção unitária `d` até a fronteira da
caixa anisotrópica [-delta_x, delta_x] × [-delta_y, delta_y]:
    t_box = min( delta_x / |d_x|, delta_y / |d_y| )
"""
function _calcular_fronteira_atr(delta_x::Float64, delta_y::Float64, d::AbstractVector{<:Real})
    dx_abs = abs(d[1]) + 1e-12
    dy_abs = abs(d[2]) + 1e-12
    return min(delta_x / dx_abs, delta_y / dy_abs)
end

"""
    gerar_niveis_inteligentes(prob, full_log; densidade=2) -> Vector{Float64}

Gera níveis de contorno que cobrem as bacias e valores de f(x) visitados pela trajetória,
robusto para funções com valores positivos, negativos ou nulos.
"""
function gerar_niveis_inteligentes(prob, full_log; densidade=2)
    z_vals = Float64[]
    for step in full_log
        push!(z_vals, Float64(prob.f(step.x_from)))
        push!(z_vals, Float64(prob.f(step.x_to)))
    end

    if isempty(z_vals)
        return collect(range(0.0, 100.0, length=20))
    end

    z_min = minimum(z_vals)
    z_max = maximum(z_vals)
    
    # Se os valores forem todos positivos e cobrirem ordens de grandeza
    if z_min > 1e-4 && (z_max / z_min > 10.0)
        sort!(z_vals)
        unique_z = Float64[z_vals[1]]
        for v in z_vals[2:end]
            if v > unique_z[end] * 1.05
                push!(unique_z, v)
            end
        end
        push!(unique_z, z_max * 1.5, z_max * 5.0)

        niveis = Float64[]
        for i in 1:(length(unique_z) - 1)
            a, b = unique_z[i], unique_z[i + 1]
            push!(niveis, a)
            if b > a * 2.0 && a > 1e-6
                r_log = range(log10(a), log10(b); length=densidade + 2)
                for k in 2:(length(r_log) - 1)
                    push!(niveis, 10.0 ^ r_log[k])
                end
            end
        end
        push!(niveis, unique_z[end])
        return unique(sort(niveis))
    else
        # Escala linear/híbrida para problemas gerais ou negativos
        margem = max(abs(z_max - z_min) * 0.1, 1.0)
        return collect(range(z_min - margem * 0.2, z_max + margem, length=25))
    end
end

"""
    _definir_estilo_passo(phase::Symbol, status::Symbol) -> (Symbol, String, Symbol, Int)

Retorna a tupla de estilo visual: (cor, rotulo, forma_marcador, tamanho_marcador).
"""
function _definir_estilo_passo(phase::Symbol, status::Symbol)
    if phase == :restoration
        cor = (status == :accepted) ? :magenta : :purple
        lbl = (status == :accepted) ? "RESTAURAÇÃO (Aceito)" : "RESTAURAÇÃO (Rej.)"
        frm = (status == :accepted) ? :diamond : :x
        ms  = (status == :accepted) ? 9 : 7
    else
        cor = (status == :accepted) ? :green : :red
        lbl = (status == :accepted) ? "OTIMIZAÇÃO (Aceito)" : "OTIMIZAÇÃO (Rej.)"
        frm = (status == :accepted) ? :circle : :x
        ms  = (status == :accepted) ? 8 : 7
    end
    return cor, lbl, frm, ms
end

"""
    _calcular_bounding_box_global(full_log) -> Dict{Symbol, Float64}

Pré-calcula a caixa delimitadora global estática da trajetória e de todas as caixas de confiança.
Garante que a malha global seja calculada UMA ÚNICA VEZ para todos os frames.
"""
function _calcular_bounding_box_global(full_log)
    min_x, max_x = Inf, -Inf
    min_y, max_y = Inf, -Inf

    for step in full_log
        dx, dy, _ = _extrair_delta(step)
        for pt in (step.x_from, step.x_to)
            min_x = min(min_x, pt[1] - dx, pt[1] + dx)
            max_x = max(max_x, pt[1] - dx, pt[1] + dx)
            min_y = min(min_y, pt[2] - dy, pt[2] + dy)
            max_y = max(max_y, pt[2] - dy, pt[2] + dy)
        end
    end

    if !isfinite(min_x)
        min_x, max_x, min_y, max_y = -2.0, 2.0, -2.0, 2.0
    end

    span_x = max(max_x - min_x, 0.5)
    span_y = max(max_y - min_y, 0.5)

    return Dict(
        :min_x => min_x - span_x * 0.15,
        :max_x => max_x + span_x * 0.15,
        :min_y => min_y - span_y * 0.15,
        :max_y => max_y + span_y * 0.15
    )
end

"""
    _avaliar_malha_2d(f, xs, ys) -> Matrix{Float64}

Avalia uma função f([x, y]) em uma grade 2D sem alocações no loop interno.
"""
function _avaliar_malha_2d(f, xs, ys)
    Z = Matrix{Float64}(undef, length(ys), length(xs))
    buf = [0.0, 0.0]
    for (j, x) in enumerate(xs)
        buf[1] = x
        for (k, y) in enumerate(ys)
            buf[2] = y
            Z[k, j] = Float64(f(buf))
        end
    end
    return Z
end

# ==============================================================================
# RENDERIZADORES DE CENÁRIO (GLOBAL E ZOOM)
# ==============================================================================

function _montar_cenario_global_2d(prob, meus_niveis, xs_glob, ys_glob, Z_glob, restricoes_glob,
                                   trajetoria_x, trajetoria_y, rej_x, rej_y, rej_pts_x, rej_pts_y,
                                   params_passo)
    cx, cy = params_passo.cx, params_passo.cy
    dest_x, dest_y = params_passo.dest_x, params_passo.dest_y
    dx, dy = params_passo.dx, params_passo.dy
    cor_passo, forma_passo, ms_passo = params_passo.cor, params_passo.forma, params_passo.ms
    is_final = params_passo.is_final
    grad_curr = params_passo.grad_curr

    xlims = (xs_glob[1], xs_glob[end])
    ylims = (ys_glob[1], ys_glob[end])

    # 1. Base pré-avaliada de contornos (zero re-cálculo)
    p = contour(xs_glob, ys_glob, Z_glob; levels=meus_niveis, fill=true, c=:viridis, 
                alpha=0.45, legend=false, xlims=xlims, ylims=ylims, framestyle=:box, aspect_ratio=:equal)

    # 2. Restrições globais pré-calculadas
    for Z_c in restricoes_glob
        contour!(p, xs_glob, ys_glob, Z_c; levels=[0.0], c=:cyan, lw=2.5, alpha=0.9)
    end

    # 3. Rastro de passos rejeitados passados (vetorizado com NaN)
    if !isempty(rej_x)
        plot!(p, rej_x, rej_y; c=:red, alpha=0.35, style=:dot, lw=1.0, label="")
        scatter!(p, rej_pts_x, rej_pts_y; c=:red, alpha=0.35, shape=:x, ms=4, label="")
    end

    # 4. Trajetória aceita (vetorizada)
    if length(trajetoria_x) > 1
        plot!(p, trajetoria_x, trajetoria_y; c=:black, alpha=0.75, lw=1.5, label="")
        scatter!(p, trajetoria_x, trajetoria_y; c=:black, alpha=0.75, ms=3, label="")
    end

    # 5. Caixa da Região de Confiança Atual (Anisotrópica)
    box_x = [cx - dx, cx + dx, cx + dx, cx - dx, cx - dx]
    box_y = [cy - dy, cy - dy, cy + dy, cy + dy, cy - dy]
    plot!(p, box_x, box_y; c=:blue, style=:dash, lw=2.0, label="")

    # 6. Passo Atual
    if !is_final
        plot!(p, [cx, dest_x], [cy, dest_y]; c=cor_passo, arrow=true, lw=2.5, label="")
        scatter!(p, [cx], [cy]; c=:blue, ms=6, label="")
    end
    scatter!(p, [dest_x], [dest_y]; c=cor_passo, shape=forma_passo, ms=ms_passo, label="")

    # 7. Vetor Gradiente Negativo (direção de descida clássica)
    norm_g = norm(grad_curr)
    if norm_g > 1e-8
        span_diag = max(xlims[2] - xlims[1], ylims[2] - ylims[1])
        v_curr = (-grad_curr ./ norm_g) * (span_diag * 0.12)
        plot!(p, [cx, cx + v_curr[1]], [cy, cy + v_curr[2]]; c=:magenta, arrow=true, lw=2.0, label="")
    end

    title!(p, "Visão Global (Histórico Completo)")
    return p
end

function _montar_cenario_local_2d(prob, meus_niveis, params_passo; grid_pts=60)
    cx, cy = params_passo.cx, params_passo.cy
    dest_x, dest_y = params_passo.dest_x, params_passo.dest_y
    dx, dy = params_passo.dx, params_passo.dy
    cor_passo, forma_passo, ms_passo = params_passo.cor, params_passo.forma, params_passo.ms
    is_final = params_passo.is_final
    grad_curr = params_passo.grad_curr

    # Enquadramento inteligente ao redor do ponto e da caixa ATR
    step_span = norm([dest_x - cx, dest_y - cy], Inf)
    max_span = max(dx, dy, step_span, 1e-3) * 1.6
    xlims = (cx - max_span, cx + max_span)
    ylims = (cy - max_span, cy + max_span)

    xs_loc = range(xlims[1], xlims[2], length=grid_pts)
    ys_loc = range(ylims[1], ylims[2], length=grid_pts)
    Z_loc = _avaliar_malha_2d(prob.f, xs_loc, ys_loc)

    p = contour(xs_loc, ys_loc, Z_loc; levels=meus_niveis, fill=true, c=:viridis, 
                alpha=0.45, legend=false, xlims=xlims, ylims=ylims, framestyle=:box, aspect_ratio=:equal)

    # Restrições locais
    if hasproperty(prob, :h)
        for k in 1:prob.ncon
            Z_con = _avaliar_malha_2d((x) -> prob.h(x)[k], xs_loc, ys_loc)
            contour!(p, xs_loc, ys_loc, Z_con; levels=[0.0], c=:cyan, lw=2.5, alpha=0.9)
        end
    end

    # Caixa ATR no Zoom Local
    box_x = [cx - dx, cx + dx, cx + dx, cx - dx, cx - dx]
    box_y = [cy - dy, cy - dy, cy + dy, cy + dy, cy - dy]
    plot!(p, box_x, box_y; c=:blue, style=:dash, lw=2.2, label="")

    # Passo e Pontos
    if !is_final
        plot!(p, [cx, dest_x], [cy, dest_y]; c=cor_passo, arrow=true, lw=2.8, label="")
        scatter!(p, [cx], [cy]; c=:blue, ms=7, label="")
    end
    scatter!(p, [dest_x], [dest_y]; c=cor_passo, shape=forma_passo, ms=ms_passo, label="")

    # Gradiente negativo local
    norm_g = norm(grad_curr)
    if norm_g > 1e-8
        v_curr = (-grad_curr ./ norm_g) * (max_span * 0.45)
        plot!(p, [cx, cx + v_curr[1]], [cy, cy + v_curr[2]]; c=:magenta, arrow=true, lw=2.2, label="")
    end

    title!(p, "Zoom Local (Região de Confiança)")
    return p
end

# ==============================================================================
# FUNÇÕES PRINCIPAIS EXPORTADAS
# ==============================================================================

"""
    visualizar_frames_3plots(prob, full_log; kwargs...)

Gera o painel completo de visualização composto por 4 subplots por iteração:
1. Top-Left: Visão Global (histórico de aceitos/rejeitados e contornos globais pré-calculados).
2. Top-Right: Zoom Local na iteração atual, exibindo o retângulo da caixa ATR e curvas de nível.
3. Bottom-Left: Corte 1D na direção do gradiente/descida, com fronteira exata da caixa ATR.
4. Bottom-Right: Corte 1D na direção do passo LP, exibindo o ponto destino e a fronteira da caixa.

# Opções principais:
- `save_video::Bool=true`: Compila o vídeo em MP4 via FFMPEG.
- `save_frames::Bool=true`: Salva os arquivos PNG individuais de cada frame.
- `fps_video::Int=10`: Taxa de quadros do vídeo gerado.
- `hold_time_sec::Int=3`: Tempo de pausa exibindo o frame final da convergência.
- `algo_name::String=""`: Nome do algoritmo/variante para o título (ex: "UNIF (ATR: secant_box)").
- `prob_name::String=""`: Nome do problema (ex: "ROSENBR").
"""
function visualizar_frames_3plots(
    prob, full_log;
    mode::Symbol=:combined,
    save_frames::Bool=true,
    save_video::Bool=true,
    add_audio::Bool=false,
    audio_path::String="",
    hold_time_sec::Int=3,
    fps_video::Int=10,
    output_dir::String="",
    video_name::String="",
    algo_name::String="",
    prob_name::String="",
    grid_density::Int=60,
    dpi::Int=120
)
    if isempty(full_log)
        println("⚠️ O log de passos está vazio. Nenhuma visualização gerada.")
        return
    end

    # Definição de diretórios
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    RUN_DIR = isempty(output_dir) ? joinpath(DEFAULT_OUTPUT_ROOT, "run_$timestamp") : output_dir
    FRAMES_DIR = joinpath(RUN_DIR, "frames")
    mkpath(FRAMES_DIR)

    video_filename = isempty(video_name) ? "otimizacao_output.mp4" : video_name
    if !endswith(video_filename, ".mp4")
        video_filename *= ".mp4"
    end
    NOME_VIDEO_FINAL = joinpath(RUN_DIR, video_filename)

    p_name = isempty(prob_name) ? (hasproperty(prob, :name) ? prob.name : "Problema") : prob_name
    a_name = isempty(algo_name) ? "SLP" : algo_name

    println("\n" * "="^80)
    println("🎥 Renderizando Visualização Otimizada: $p_name | $a_name")
    println("📁 Diretório: $RUN_DIR")
    println("⚙️  Passos no log: $(length(full_log)) | FPS: $fps_video | Grid: $(grid_density)x$(grid_density)")
    println("="^80)

    # 1. Pré-cálculo da grade global e contornos estáticos (1 única avaliação!)
    bbox = _calcular_bounding_box_global(full_log)
    xs_glob = range(bbox[:min_x], bbox[:max_x], length=grid_density)
    ys_glob = range(bbox[:min_y], bbox[:max_y], length=grid_density)
    
    print("🎨 Pré-calculando malha e contornos globais... ")
    meus_niveis = gerar_niveis_inteligentes(prob, full_log; densidade=2)
    Z_glob = _avaliar_malha_2d(prob.f, xs_glob, ys_glob)
    
    restricoes_glob = []
    if hasproperty(prob, :h) && prob.ncon > 0
        for k in 1:prob.ncon
            push!(restricoes_glob, _avaliar_malha_2d((x) -> prob.h(x)[k], xs_glob, ys_glob))
        end
    end
    println("OK")

    # Arrays de trajetórias vetorizadas
    trajetoria_x = Float64[full_log[1].x_from[1]]
    trajetoria_y = Float64[full_log[1].x_from[2]]
    rej_x = Float64[]
    rej_y = Float64[]
    rej_pts_x = Float64[]
    rej_pts_y = Float64[]

    num_steps = length(full_log)
    total_frames = num_steps + 1

    print("🖼️  Renderizando $(total_frames) frames ")

    for i in 1:total_frames
        is_final = (i > num_steps)
        step = is_final ? full_log[end] : full_log[i]

        dx, dy, d_scalar = _extrair_delta(step)

        if is_final
            cx = dest_x = step.x_to[1]
            cy = dest_y = step.x_to[2]
            cor_passo, label_passo, forma_passo, ms_passo = :yellow, "CONVERGÊNCIA FINAL", :star5, 12
            grad_curr = prob.∇f(step.x_to)
            f_curr = Float64(prob.f(step.x_to))
        else
            cx, cy = step.x_from[1], step.x_from[2]
            dest_x, dest_y = step.x_to[1], step.x_to[2]
            cor_passo, label_passo, forma_passo, ms_passo = _definir_estilo_passo(step.phase, step.status)
            grad_curr = prob.∇f(step.x_from)
            f_curr = Float64(prob.f(step.x_from))
        end

        params_passo = (
            i = i, cx = cx, cy = cy, dest_x = dest_x, dest_y = dest_y,
            dx = dx, dy = dy, d_scalar = d_scalar, cor = cor_passo,
            forma = forma_passo, ms = ms_passo, is_final = is_final,
            grad_curr = grad_curr, label = label_passo
        )

        # Plot 1: Visão Global
        p1 = _montar_cenario_global_2d(
            prob, meus_niveis, xs_glob, ys_glob, Z_glob, restricoes_glob,
            trajetoria_x, trajetoria_y, rej_x, rej_y, rej_pts_x, rej_pts_y, params_passo
        )

        # Plot 2: Zoom Local
        p2 = _montar_cenario_local_2d(prob, meus_niveis, params_passo; grid_pts=grid_density)

        # --- CORTES 1D MATEMATICAMENTE EXATOS PARA ATR ---
        origem = is_final ? step.x_to : step.x_from
        norm_g = norm(grad_curr)

        # Plot 3: Corte 1D na Direção do Gradiente / Descida Projetada
        dir_corte = (norm_g > 1e-8) ? (-grad_curr ./ norm_g) : [1.0, 0.0]
        t_box_grad = _calcular_fronteira_atr(dx, dy, dir_corte)
        span_grad = max(t_box_grad * 1.5, 1e-3)
        t_vals_g = range(-span_grad, span_grad, length=100)
        f_slice_g = [Float64(prob.f(origem + t * dir_corte)) for t in t_vals_g]
        m_slice_g = [f_curr - t * norm_g for t in t_vals_g]

        p3 = plot(t_vals_g, f_slice_g, lw=2.2, c=:black, label="f(x + t·d_grad)", legend=:topright)
        plot!(p3, t_vals_g, m_slice_g, c=:magenta, style=:dot, lw=1.8, label="Modelo Linear")
        scatter!(p3, [0.0], [f_curr], c=:blue, ms=5, label="x_k")
        vline!(p3, [-t_box_grad, t_box_grad], c=:red, style=:dash, lw=1.5, 
               label=@sprintf("Fronteira ATR (±%.1e)", t_box_grad))
        title!(p3, "Corte na Direção de Descida"); xlabel!(p3, "Parâmetro t"); ylabel!(p3, "f")

        # Plot 4: Corte 1D na Direção do Passo LP
        step_vec = [dest_x - cx, dest_y - cy]
        len_step = norm(step_vec)
        dir_step = (len_step > 1e-8) ? (step_vec ./ len_step) : [1.0, 0.0]
        t_box_step = _calcular_fronteira_atr(dx, dy, dir_step)
        span_step = max(t_box_step, len_step) * 1.5
        span_step = max(span_step, 1e-3)
        t_vals_s = range(-span_step, span_step, length=100)
        f_slice_s = [Float64(prob.f(origem + t * dir_step)) for t in t_vals_s]
        slope_step = dot(grad_curr, dir_step)
        m_slice_s = [f_curr + t * slope_step for t in t_vals_s]

        p4 = plot(t_vals_s, f_slice_s, lw=2.2, c=:black, label="f(x + t·d_passo)", legend=:topright)
        plot!(p4, t_vals_s, m_slice_s, c=:magenta, style=:dot, lw=1.8, label="Modelo Linear")
        scatter!(p4, [0.0], [f_curr], c=:blue, ms=5, label="x_k")
        scatter!(p4, [len_step], [Float64(prob.f([dest_x, dest_y]))], c=cor_passo, shape=forma_passo, 
                 ms=ms_passo, label=label_passo)
        vline!(p4, [-t_box_step, t_box_step], c=:red, style=:dash, lw=1.5, 
               label=@sprintf("Fronteira ATR (±%.1e)", t_box_step))
        title!(p4, "Corte na Direção do Passo"); xlabel!(p4, "Distância t"); ylabel!(p4, "f")

        # --- TÍTULO DO PAINEL PRINCIPAL COM MÉTRICAS COMPLETAS ---
        atr_ratio = (min(dx, dy) > 1e-12) ? (max(dx, dy) / min(dx, dy)) : 1.0
        
        # Formatação de métricas específicas
        if hasproperty(step, :ared) && hasproperty(step, :pred)
            ared_val, pred_val = Float64(step.ared), Float64(step.pred)
            ratio_rho = abs(pred_val) > 1e-12 ? (ared_val / pred_val) : 0.0
            metric_str = @sprintf("Ared: %.2e | Pred: %.2e | Razão ρ: %.2f | ||s||: %.2e", 
                                  ared_val, pred_val, ratio_rho, len_step)
        elseif hasproperty(step, :Lagrange) && hasproperty(step, :Merit)
            norm_h = (hasproperty(prob, :h) && prob.ncon > 0) ? norm(prob.h([cx, cy])) : 0.0
            metric_str = @sprintf("Mérito: %.2e | Lagrange: %.2e | ||h(x)||: %.2e | ||s||: %.2e", 
                                  step.Merit, step.Lagrange, norm_h, len_step)
        else
            metric_str = @sprintf("||s||: %.2e", len_step)
        end

        main_title = string(
            "[$p_name] $a_name  |  Iter $(step.iter)  |  $label_passo\n",
            @sprintf("x = [%.4f, %.4f]  |  f(x) = %.4e  |  δ = [%.1e, %.1e] (Razão: %.1fx)\n", cx, cy, f_curr, dx, dy, atr_ratio),
            metric_str, "\n\n"
        )

        p_final = plot(p1, p2, p3, p4; layout=@layout([a b; c d]), size=(1280, 960), 
                       plot_title=main_title, dpi=dpi, plot_titlevspan=0.12)

        frame_file = joinpath(FRAMES_DIR, @sprintf("frame_%05d.png", i))
        savefig(p_final, frame_file)

        # Atualiza trajetórias acumuladas
        if !is_final
            if step.status == :accepted
                push!(trajetoria_x, dest_x)
                push!(trajetoria_y, dest_y)
            else
                push!(rej_x, cx, dest_x, NaN)
                push!(rej_y, cy, dest_y, NaN)
                push!(rej_pts_x, dest_x)
                push!(rej_pts_y, dest_y)
            end
        end

        if i % 10 == 0 || i == total_frames
            print(".")
        end
    end
    println(" OK!")

    # Duplica o último frame para segurar a visualização do ponto final no vídeo
    if hold_time_sec > 0
        ultimo_frame = joinpath(FRAMES_DIR, @sprintf("frame_%05d.png", total_frames))
        num_hold = fps_video * hold_time_sec
        for h in 1:num_hold
            frame_idx = total_frames + h
            dest_hold = joinpath(FRAMES_DIR, @sprintf("frame_%05d.png", frame_idx))
            cp(ultimo_frame, dest_hold; force=true)
        end
    end

    # 2. Compilação Direta e Rápida do Vídeo via FFMPEG
    if save_video
        println("🎬 Compilando vídeo MP4 via FFMPEG...")
        try
            FFMPEG.ffmpeg() do exe
                pattern = joinpath(FRAMES_DIR, "frame_%05d.png")
                run(`$exe -y -framerate $fps_video -i $pattern -c:v libx264 -pix_fmt yuv420p $NOME_VIDEO_FINAL`)
            end
            println("✅ Vídeo gravado com sucesso em:\n   $NOME_VIDEO_FINAL")

            if add_audio && !isempty(audio_path) && isfile(audio_path)
                println("🎵 Adicionando trilha sonora...")
                temp_audio_vid = joinpath(RUN_DIR, "temp_video_audio.mp4")
                try
                    FFMPEG.ffmpeg() do exe
                        run(`$exe -y -i $NOME_VIDEO_FINAL -i $audio_path -c:v copy -map 0:v -map 1:a -shortest $temp_audio_vid`)
                    end
                    mv(temp_audio_vid, NOME_VIDEO_FINAL; force=true)
                    println("✅ Áudio incorporado com sucesso!")
                catch e
                    println("⚠️ Falha ao embutir áudio: $e")
                end
            end
        catch err
            println("❌ Erro durante compilação do FFMPEG: $err")
        end
    end

    if !save_frames
        println("🧹 Limpando frames temporários individuais...")
        rm(FRAMES_DIR; recursive=true, force=true)
    else
        println("🖼️  Frames individuais preservados em: $FRAMES_DIR")
    end

    return NOME_VIDEO_FINAL
end

"""
    visualizar_frames_dinamicos(prob, full_log; kwargs...)

Versão mais compacta com 2 subplots lado a lado (Visão Global + Zoom Local).
"""
function visualizar_frames_dinamicos(
    prob, full_log;
    mode::Symbol=:combined,
    save_frames::Bool=true,
    save_video::Bool=true,
    fps_video::Int=10,
    output_dir::String="",
    video_name::String="dinamico.mp4",
    algo_name::String="",
    prob_name::String=""
)
    return visualizar_frames_3plots(
        prob, full_log;
        mode=mode, save_frames=save_frames, save_video=save_video,
        fps_video=fps_video, output_dir=output_dir, video_name=video_name,
        algo_name=algo_name, prob_name=prob_name
    )
end

"""
    visualize_optimization(prob, full_log; kwargs...)

Alias ergonômico moderno para `visualizar_frames_3plots`.
"""
const visualize_optimization = visualizar_frames_3plots

# Alias retrocompatível com módulos antigos
const visualizar_frames_3plotsModule = Generic_Visualization

end # module
