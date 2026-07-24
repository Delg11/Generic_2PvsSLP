module visualizar_frames_3plotsModule

export visualizar_frames_3plots, visualizar_frames_dinamicos

# ==============================================================================
# EXEMPLOS DE USO
# ==============================================================================

# 1. Apenas plota e gera os PNGs na pasta (sem vídeo):
# visualizar_frames_3plots(problem, full_log, save_video=false, save_frames=true)

# 2. Gera frames apenas para as últimas 10 iterações (ótimo para analisar a convergência final):
# visualizar_frames_3plots(problem, last(full_log, 10), save_video=false, save_frames=true)

# 3. Gera frames pulando a cada 1000 passos (ótimo para históricos muito longos):
# visualizar_frames_3plots(problem, full_log[begin:1000:end], save_video=false, save_frames=true)

# 4. Apenas gera um MP4 silencioso (sem salvar as dezenas de imagens soltas na pasta):
# visualizar_frames_3plots(problem, full_log, save_frames=false, save_video=true, add_audio=false)

# 5. Gera o pacote completo (frames + vídeo ÉPICO com a música do Interstellar):
# caminho_musica = raw"C:\Users\gdelg\OneDrive\Documentos\GitHub\TopPT-SLP-Codigo-Limpo\Interstellar (Main Theme Piano) - Gacabe & Jecabe.mp3"
# visualizar_frames_3plots(problem, full_log, save_frames=true, save_video=true, add_audio=true, audio_path=caminho_musica)

using Dates
using LinearAlgebra
using Plots
using Printf
using FFMPEG

const OUTPUT_DIR_DEFAULT = "frames"

# ==============================================================================
# FUNÇÕES AUXILIARES GERAIS
# ==============================================================================

"""
Gera um vetor de níveis que inclui obrigatoriamente os valores de f(x) visitados,
preenchendo os espaços vazios com intermediários logarítmicos.
"""
function gerar_niveis_inteligentes(prob, full_log; densidade=2)
    z_values = Float64[0.1] # Fundo do vale
    
    for step in full_log
        push!(z_values, prob.f(step.x_from), prob.f(step.x_to))
    end
    append!(z_values, [1e4, 1e5, 1e6]) # Paredes
    
    sort!(z_values)
    unique_z = Float64[z_values[1]]
    
    for v in z_values[2:end]
        if v > unique_z[end] * 1.01
            push!(unique_z, v)
        end
    end
    
    niveis_finais = Float64[]
    for i in 1:(length(unique_z) - 1)
        a, b = unique_z[i], unique_z[i + 1]
        push!(niveis_finais, a)
        
        if b > a * 2.0 && a > 1e-6
            range_logs = range(log10(a), log10(b); length=densidade+2)
            for k in 2:(length(range_logs) - 1)
                push!(niveis_finais, 10.0 ^ range_logs[k])
            end
        end
    end
    push!(niveis_finais, unique_z[end])
    return niveis_finais
end

function _definir_estilo_passo(phase, status)
    if phase == :restoration
        cor = (status == :accepted) ? :magenta : :purple
        lbl = (status == :accepted) ? "RESTAURAÇÃO (Aceito)" : "RESTAURAÇÃO (Rej.)"
        frm = (status == :accepted) ? :diamond : :x
        ms  = (status == :accepted) ? 10 : 8
    else
        cor = (status == :accepted) ? :green : :red
        lbl = (status == :accepted) ? "OTIMIZAÇÃO (Aceito)" : "OTIMIZAÇÃO (Rej.)"
        frm = (status == :accepted) ? :circle : :x
        ms  = 8
    end
    return cor, lbl, frm, ms
end

function _extrair_delta(step)
    if step.delta isa Vector
        return step.delta[1], step.delta[2], maximum(step.delta)
    end
    return step.delta, step.delta, step.delta
end

function _atualizar_limites!(limites, trajetoria_x, trajetoria_y, cx, cy, dest_x, dest_y, dx, dy)
    pontos_x = [cx, dest_x, cx - dx, cx + dx]
    pontos_y = [cy, dest_y, cy - dy, cy + dy]
    
    limites[:min_x] = min(limites[:min_x], minimum(trajetoria_x; init=Inf), minimum(pontos_x))
    limites[:max_x] = max(limites[:max_x], maximum(trajetoria_x; init=-Inf), maximum(pontos_x))
    limites[:min_y] = min(limites[:min_y], minimum(trajetoria_y; init=Inf), minimum(pontos_y))
    limites[:max_y] = max(limites[:max_y], maximum(trajetoria_y; init=-Inf), maximum(pontos_y))
end

# ==============================================================================
# PLOTAGENS ESPECÍFICAS
# ==============================================================================

function _montar_cenario_2d(tipo_visao, prob, step, full_log, trajetoria_x, trajetoria_y, 
                            meus_niveis, limites_globais, params_passo)
    
    cx, cy, dest_x, dest_y = params_passo.cx, params_passo.cy, params_passo.dest_x, params_passo.dest_y
    dx, dy, d_scalar = params_passo.dx, params_passo.dy, params_passo.d_scalar
    cor_passo, forma_passo, ms_passo = params_passo.cor, params_passo.forma, params_passo.ms
    is_final = params_passo.is_final
    grad_curr = params_passo.grad_curr

    # 1. Determina Limites da Câmera
    if tipo_visao == :global
        vx_min, vx_max = limites_globais[:min_x], limites_globais[:max_x]
        vy_min, vy_max = limites_globais[:min_y], limites_globais[:max_y]
        mostrar_rastro = true
    else
        max_span = max(d_scalar, norm([dest_x, dest_y] - [cx, cy], Inf)) * 1.2
        vx_min, vx_max = cx - max_span, cx + max_span
        vy_min, vy_max = cy - max_span, cy + max_span
        mostrar_rastro = false
    end

    span_x, span_y = max(vx_max - vx_min, 0.1), max(vy_max - vy_min, 0.1)
    xlims = (vx_min - span_x*0.1, vx_max + span_x*0.1)
    ylims = (vy_min - span_y*0.1, vy_max + span_y*0.1)
    grid_dens = max(span_x, span_y) / 100.0

    # 2. Contour Plot Base
    p = contour(xlims[1]:grid_dens:xlims[2], ylims[1]:grid_dens:ylims[2], (x, y) -> prob.f([x, y]); 
                levels=meus_niveis, fill=true, c=:viridis, alpha=0.5, legend=false, 
                xlims=xlims, ylims=ylims, framestyle=:box, aspect_ratio=:equal)

    # 3. Restrições
    if hasproperty(prob, :h)
        h_test = prob.h([cx, cy])
        num_restricoes = length(h_test)
        for k in 1:num_restricoes
            contour!(p, xlims[1]:grid_dens:xlims[2], ylims[1]:grid_dens:ylims[2], 
                     (x, y) -> prob.h([x, y])[k]; levels=[0.0], c=:cyan, lw=3, alpha=0.9)
        end
    end

    # 4. Rastro (Histórico)
    if mostrar_rastro
        lim_k = is_final ? length(full_log) : (params_passo.i - 1)
        for k in 1:lim_k
            old = full_log[k]
            if old.status != :accepted && old.phase == :optimization
                plot!(p, [old.x_from[1], old.x_to[1]], [old.x_from[2], old.x_to[2]]; c=:red, alpha=0.3, style=:dot)
                scatter!(p, [old.x_to[1]], [old.x_to[2]]; c=:red, alpha=0.3, shape=:x, ms=4)
            end
        end
        if length(trajetoria_x) > 1
            plot!(p, trajetoria_x, trajetoria_y; c=:black, alpha=0.6, lw=1.5)
            scatter!(p, trajetoria_x, trajetoria_y; c=:black, alpha=0.6, ms=3)
        end
    end

    # 5. Passo Atual
    plot!(p, [cx-dx, cx+dx, cx+dx, cx-dx, cx-dx], [cy-dy, cy-dy, cy+dy, cy+dy, cy-dy]; c=:blue, style=:dash, lw=2.0)
    
    if !is_final
        plot!(p, [cx, dest_x], [cy, dest_y]; c=cor_passo, arrow=true, lw=3)
        scatter!(p, [cx], [cy]; c=:blue, ms=7)
    end
    scatter!(p, [dest_x], [dest_y]; c=cor_passo, shape=forma_passo, ms=ms_passo)

    # 6. Gradiente
    norm_g = norm(grad_curr)
    if norm_g > 1e-8
        scale_grad = max(span_x, span_y) * 0.15
        v_curr = (grad_curr ./ norm_g) * scale_grad
        plot!(p, [cx, cx + v_curr[1]], [cy, cy + v_curr[2]]; c=:magenta, arrow=true, lw=2)
    end

    title!(p, tipo_visao == :global ? "Visão Global (Histórico)" : "Zoom Local (Iteração)")
    return p
end

# ==============================================================================
# FUNÇÕES PRINCIPAIS (EXPORTADAS)
# ==============================================================================

"""
Gera frames da otimização com diferentes modos de visualização:
- `mode=:history` : Visão Global com rastro completo.
- `mode=:zoom`    : Visão Local focada na iteração atual.
- `mode=:combined`: Lado a lado (Global + Zoom).
"""
function visualizar_frames_dinamicos(prob, full_log; mode::Symbol=:combined)
    rm(OUTPUT_DIR_DEFAULT; recursive=true, force=true)
    mkpath(OUTPUT_DIR_DEFAULT)
    println("\n🎥 Renderizando frames dinâmicos (Modo: $mode)...")

    meus_niveis = gerar_niveis_inteligentes(prob, full_log; densidade=3)
    trajetoria_x, trajetoria_y = Float64[], Float64[]
    limites_globais = Dict(:min_x => Inf, :max_x => -Inf, :min_y => Inf, :max_y => -Inf)

    if !isempty(full_log)
        push!(trajetoria_x, full_log[1].x_from[1])
        push!(trajetoria_y, full_log[1].x_from[2])
    end

    for (i, step) in enumerate(full_log)
        cx, cy = step.x_from[1], step.x_from[2]
        dest_x, dest_y = step.x_to[1], step.x_to[2]
        dx, dy, d_scalar = _extrair_delta(step)
        cor_passo, label_passo, forma_passo, ms_passo = _definir_estilo_passo(step.phase, step.status)

        _atualizar_limites!(limites_globais, trajetoria_x, trajetoria_y, cx, cy, dest_x, dest_y, dx, dy)

        params = (i=i, cx=cx, cy=cy, dest_x=dest_x, dest_y=dest_y, dx=dx, dy=dy, 
                  d_scalar=d_scalar, cor=cor_passo, forma=forma_passo, ms=ms_passo, 
                  is_final=false, grad_curr=prob.∇f(step.x_from))

        plots_comb = []
        if mode in [:history, :combined]; push!(plots_comb, _montar_cenario_2d(:global, prob, step, full_log, trajetoria_x, trajetoria_y, meus_niveis, limites_globais, params)); end
        if mode in [:zoom, :combined];    push!(plots_comb, _montar_cenario_2d(:local, prob, step, full_log, trajetoria_x, trajetoria_y, meus_niveis, limites_globais, params)); end

        final_size = mode == :combined ? (1200, 500) : (800, 600)
        main_title = @sprintf("Iter %d | %s | %s\nLag: %.1e | Merit: %.1e | δ(max): %.1e\n\n", 
                              step.iter, step.phase, uppercase(string(step.status)), step.Lagrange, step.Merit, d_scalar)

        p_final = plot(plots_comb...; layout=(1, length(plots_comb)), size=final_size, plot_title=main_title, dpi=150, plot_titlevspan=0.15)
        savefig(p_final, joinpath(OUTPUT_DIR_DEFAULT, @sprintf("frame_%04d.png", i)))

        if step.status == :accepted
            push!(trajetoria_x, dest_x)
            push!(trajetoria_y, dest_y)
        end
        if i % 10 == 0; print("."); end
    end
    println("\n✅ Concluído!")
end

"""
Gera 3 gráficos por iteração:
1. Global (Histórico) | 2. Zoom | 3. Corte Gradiente | 4. Corte Passo
"""
function visualizar_frames_3plots(
    prob, full_log; 
    mode::Symbol=:combined,
    save_frames::Bool=true,
    save_video::Bool=true,
    add_audio::Bool=false,
    audio_path::String="",
    hold_time_sec::Int=3,
    fps_video::Int=5
)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    RUN_DIR = joinpath("Outputs_Otimizacao", "run_$timestamp")
    FRAMES_DIR = joinpath(RUN_DIR, "frames")
    NOME_VIDEO_FINAL = joinpath(RUN_DIR, "otimizacao_output.mp4")

    if save_frames || save_video; mkpath(FRAMES_DIR); end

    println("\n🎥 Renderizando frames 3Plots (Modo: $mode)...")
    anim = Animation()
    meus_niveis = gerar_niveis_inteligentes(prob, full_log; densidade=3)

    trajetoria_x, trajetoria_y = Float64[], Float64[]
    limites_globais = Dict(:min_x => Inf, :max_x => -Inf, :min_y => Inf, :max_y => -Inf)

    if !isempty(full_log)
        push!(trajetoria_x, full_log[1].x_from[1])
        push!(trajetoria_y, full_log[1].x_from[2])
    end

    num_steps = length(full_log)
    
    for i in 1:(num_steps + 1)
        is_final = i > num_steps
        step = is_final ? full_log[end] : full_log[i]
        
        # Configuração de Passo (Final vs Normal)
# Configuração de Passo (Final vs Normal)
        if is_final
            # CORREÇÃO: Separar as atribuições de X e Y
            cx = dest_x = step.x_to[1] 
            cy = dest_y = step.x_to[2]
            
            dx, dy, d_scalar = _extrair_delta(step)
            cor_passo, label_passo, forma_passo, ms_passo = :yellow, "PONTO FINAL", :star5, 12
            grad_curr, f_curr = prob.∇f(step.x_to), prob.f(step.x_to)
        else
            cx, cy = step.x_from[1], step.x_from[2]
            dest_x, dest_y = step.x_to[1], step.x_to[2]
            dx, dy, d_scalar = _extrair_delta(step)
            cor_passo, label_passo, forma_passo, ms_passo = _definir_estilo_passo(step.phase, step.status)
            grad_curr, f_curr = prob.∇f(step.x_from), prob.f(step.x_from)
        end

        _atualizar_limites!(limites_globais, trajetoria_x, trajetoria_y, cx, cy, dest_x, dest_y, dx, dy)

        params_passo = (i=i, cx=cx, cy=cy, dest_x=dest_x, dest_y=dest_y, dx=dx, dy=dy, 
                        d_scalar=d_scalar, cor=cor_passo, forma=forma_passo, ms=ms_passo, 
                        is_final=is_final, grad_curr=grad_curr, label=label_passo)

        plots_comb = []
        if mode in [:history, :combined]; push!(plots_comb, _montar_cenario_2d(:global, prob, step, full_log, trajetoria_x, trajetoria_y, meus_niveis, limites_globais, params_passo)); end
        if mode in [:zoom, :combined];    push!(plots_comb, _montar_cenario_2d(:local, prob, step, full_log, trajetoria_x, trajetoria_y, meus_niveis, limites_globais, params_passo)); end

        # --- CORTES 1D ---
        norm_g = norm(grad_curr)
        range_lim = max(d_scalar * 1.5, 1e-3)
        origem_corte = is_final ? step.x_to : step.x_from

        # Plot 3: Corte Gradiente
        dir_corte = (norm_g > 1e-8) ? (-grad_curr ./ norm_g) : [1.0, 0.0]
        t_vals = range(-range_lim, range_lim, length=100)
        f_vals_slice = [prob.f(origem_corte + t * dir_corte) for t in t_vals]
        
        p3 = plot(t_vals, f_vals_slice, lw=2, c=:black, label="f(x + t⋅d)", legend=:top)
        plot!(p3, t_vals, f_curr .- t_vals .* norm_g, c=:magenta, style=:dot, alpha=0.6, label="Linear Model")
        scatter!(p3, [0.0], [f_curr], c=:blue, ms=5, label="x_k")
        vline!(p3, [-d_scalar, d_scalar], c=:red, style=:dash, label="Trust Region")
        title!(p3, "Corte no Gradiente"); xlabel!(p3, "Passo t")
        push!(plots_comb, p3)

        # Plot 4: Corte Passo
        step_vec = [dest_x - cx, dest_y - cy]
        len_step = norm(step_vec)
        dir_step = (len_step > 1e-8) ? (step_vec ./ len_step) : [1.0, 0.0]
        f_vals_step = [prob.f(origem_corte + t * dir_step) for t in t_vals]
        
        p4 = plot(t_vals, f_vals_step, lw=2, c=:black, label="f(x + t⋅d_step)", legend=:top)
        plot!(p4, t_vals, f_curr .+ t_vals .* dot(grad_curr, dir_step), c=:magenta, style=:dot, alpha=0.6, label="Linear Proj.")
        scatter!(p4, [0.0], [f_curr], c=:blue, ms=5, label="x_k")
        scatter!(p4, [len_step], [prob.f([dest_x, dest_y])], c=cor_passo, shape=forma_passo, ms=ms_passo, label=label_passo)
        vline!(p4, [-d_scalar, d_scalar], c=:red, style=:dash, label="Trust Region")
        title!(p4, "Corte na Direção do Passo"); xlabel!(p4, "Distância t")
        push!(plots_comb, p4)

        # --- MONTAGEM FINAL ---
        if hasproperty(step, :ared) && hasproperty(step, :pred)
            main_title = @sprintf("Iter %d | %s\nAred: %.1e | Pred: %.1e | δ(max): %.1e\n\n", step.iter, label_passo, step.ared, step.pred, d_scalar)
        else
            main_title = @sprintf("Iter %d | %s | %s\nLag: %.1e | Merit: %.1e | δ(max): %.1e\n\n", step.iter, step.phase, uppercase(string(step.status)), step.Lagrange, step.Merit, d_scalar)
        end

        p_final = plot(plots_comb..., layout=@layout([a b; c d]), size=(1200, 900), plot_title=main_title, dpi=120, plot_titlevspan=0.12)

        if save_video
            repeticoes = is_final ? (fps_video * hold_time_sec) : 1
            for _ in 1:repeticoes; frame(anim, p_final); end
        end
        if save_frames; savefig(p_final, joinpath(FRAMES_DIR, @sprintf("frame_%04d.png", i))); end
        
        if !is_final && step.status == :accepted
            push!(trajetoria_x, dest_x)
            push!(trajetoria_y, dest_y)
        end
        if i % 10 == 0; print("."); end
    end

    if save_frames; println("\n✅ Frames salvos em: $FRAMES_DIR"); end

    # --- PROCESSAMENTO DE VÍDEO ---
    if save_video
        temp_video = joinpath(RUN_DIR, "temp_video.mp4")
        mp4(anim, temp_video, fps=fps_video) 

        if add_audio && !isempty(audio_path) && isfile(audio_path)
            println("\n🎵 Adicionando trilha sonora...")
            try
                FFMPEG.ffmpeg_exe(`-y -i $temp_video -i $audio_path -c:v copy -map 0:v -map 1:a -shortest $NOME_VIDEO_FINAL`)
                println("✅ Vídeo final com áudio salvo em:\n   $NOME_VIDEO_FINAL")
                rm(temp_video; force=true) 
            catch e
                println("❌ Erro ao rodar FFMPEG: $e")
                mv(temp_video, NOME_VIDEO_FINAL; force=true)
            end
        else
            if add_audio; println("\n⚠️ Áudio não encontrado ou caminho vazio. Gerando sem som."); end
            mv(temp_video, NOME_VIDEO_FINAL; force=true)
            println("✅ Vídeo silencioso salvo em:\n   $NOME_VIDEO_FINAL")
        end
    end
end

end # module