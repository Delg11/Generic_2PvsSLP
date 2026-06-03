module Generic_module_slp

export solve_slp_trust_region

using NLPModels, Printf, JuMP, Gurobi, LinearAlgebra
using ..SharedTypes

"""
Log message helper
"""
function log_message!(msg; verbose=true)
    if verbose
        println(msg)
    end
end


function quadratic_backtracking_step!(
    δ_current::Union{Float64, Vector{Float64}},
    f_current::Float64,
    f_new::Float64,
    slope::Float64,
    params,
    s_norm::Float64,
    step_norm_sq::Float64,
    action::Symbol=:decrease;
    # Argumentos Opcionais
    anisotropic::Bool=false,
    s_vec::Vector{Float64}=Float64[],
    grad::Vector{Float64}=Float64[],
    grad_old::Vector{Float64}=Float64[],
    mode::Symbol=:shrink, # :shrink ou :reshape
    # Configurações renomeadas
    min_reduction_ratio::Float64=0.1,
    max_reduction_ratio::Float64=0.5,
    min_increase_ratio::Float64=1.0,
    max_increase_ratio::Float64=2.0,
)

    # --- 1. FALLBACK ISOTRÓPICO (Scalar ou Vetor Uniforme) ---
    if !anisotropic || !(δ_current isa Vector)
        δ_scalar = (δ_current isa Vector) ? maximum(δ_current) : δ_current

        b_global = slope
        a_global = f_new - f_current - slope

        safe_size = 1.0

        if action == :increase
            local_max = (step_norm_sq < δ_scalar^2) ? 2.0 : max_increase_ratio

            if a_global <= params.ratio_safeguard_tol
                safe_size = local_max
            else
                theo = -b_global / (2.0 * a_global)
                safe_size = clamp(theo, min_increase_ratio, local_max)
            end
        else # :decrease
            if a_global < params.ratio_safeguard_tol
                fallback = max(δ_scalar * params.τ1, s_norm * params.τ2)
                d_val = clamp(fallback, params.δmin, params.δmax)
                return fill(d_val, length(δ_current))
            end

            theo = -b_global / (2.0 * a_global)
            safe_size = clamp(theo, min_reduction_ratio, max_reduction_ratio)
        end

        δ_val = clamp(safe_size * δ_scalar, params.δmin, params.δmax)
        return (δ_current isa Vector) ? fill(δ_val, length(δ_current)) : δ_val
        # return fill(δ_val, length(δ_current))
    end

    # --- 2. CÁLCULO ANISOTRÓPICO ---
    n = length(δ_current)
    δ_new = copy(δ_current)
    multipliers = zeros(n)

    # === FASE A: CÁLCULO FÍSICO (Multiplicadores Teóricos) ===
    for i in 1:n
        b_i = grad[i] * s_vec[i]
        a_i = f_new - f_current - b_i

        if a_i < params.ratio_safeguard_tol
            multipliers[i] = (action == :increase) ? max_increase_ratio : 1.0
        else
            multipliers[i] = -b_i / (2.0 * a_i)
        end
    end

    # === FASE B: ESTRATÉGIA (Aplicação de Regras) ===
    has_history = !isempty(grad_old) && length(grad_old) == n

    if action == :decrease
        if mode == :reshape
            # ESTRATÉGIA 1: Mudar a escala (Reshape)
            val_max, idx_max = findmax(multipliers)

            # Aplica Clamp Seletivo
            for i in 1:n
                if i == idx_max
                    multipliers[i] = 1.0 # VENCEDOR: Mantém tamanho (Ancoragem)
                else
                    # PERDEDORES: Redução forçada
                    multipliers[i] = clamp(multipliers[i], min_reduction_ratio, max_reduction_ratio)
                end
            end

            if params.verbose_out
                @printf("   [Reshape] Escala alterada: Dim %d mantida. Outras reduzidas.\n", idx_max)
            end

        elseif mode == :shrink
            # ESTRATÉGIA 2: Reduzir região sem alterar escala (Uniform Shrink)
            b_global = slope
            a_global = f_new - f_current - slope
            
            # Taxa de redução baseada no passo direcional global
            if a_global < params.ratio_safeguard_tol
                uniform_ratio = max_reduction_ratio
            else
                theo = -b_global / (2.0 * a_global)
                uniform_ratio = clamp(theo, min_reduction_ratio, max_reduction_ratio)
            end
            
            # Aplica o MESMO multiplicador para todas as dimensões
            fill!(multipliers, uniform_ratio)

            if params.verbose_out
                @printf("   [Uniform Shrink] Escala mantida: Multiplicador global de %.2f\n", uniform_ratio)
            end
        end
    else # action == :increase
        for i in 1:n
            multipliers[i] = clamp(multipliers[i], min_increase_ratio, max_increase_ratio)
        end
    end

    # === FASE C: ATUALIZAÇÃO ===
    δ_new .= clamp.(δ_current .* multipliers, params.δmin, params.δmax)

    if params.verbose_out
        vec_str = string("[", join([@sprintf("%.2f", x) for x in multipliers], ", "), "]")
        @printf("   [Aniso %s] Multiplicadores: %s\n", string(action), vec_str)
    end

    return δ_new
end


# ==============================================================================
# MAIN SLP SOLVER (Logic adapted from StrSLP.jl)
# ==============================================================================
function solve_slp_trust_region(prob::OptimizationProblem, x0::Vector{Float64}, params_slp::SLPParams)
    # --- Inicialização ---
    x = clamp.(copy(x0), prob.xl, prob.xu)
    # n, m = prob.n, prob.m
    n = length(x)
    m = length(prob.h(x))

    F = prob.f(x)
    grad = prob.∇f(x)
    h_val = prob.h(x)
    jac = prob.∇h(x)
    if params_slp.anisotropic_trust_region
        delta = fill(params_slp.delta0, n)
    else
        delta = params_slp.delta0
    end
    iter = 0
    itrej = 0
    countG, countF, countS = 0, 0, 0
    opstop = -1

    is_feasible=false
    # --- HISTÓRICOS ---
    # 1. Apenas os passos aceitos (Trajetória "Limpa")
    x_accepted = Vector{Vector{Float64}}()
    push!(x_accepted, copy(x))

    # 2. Histórico Completo (Tentativas, Rejeições, Deltas)
    full_log = Vector{StepLog_SLP}()

    theta = 1.0;
    theta1 = 1.0;
    theta2 = 1.0;
    thetaMax = 1.0
    GUROBI_ENV = Gurobi.Env()
    lambda = zeros(m)

    if params_slp.verbose
        log_message!(repeat("=", 120))
        # @printf("%-4s %-12s %-8s %-8s %-10s %-6s %-12s\n", "It", "F(x)", "delta", "||s||", "Ared", "Acc", "Cnt(G|F|S)")
        @printf("%-4s %-12s %-10s %-8s %-8s %-8s %-8s %-10s %-6s %-12s\n", 
        "It", "F(x)", "aredfsb", "delta", "||s||", "Ared", "Pred", "Theta", "Acc", "Cnt")
        log_message!(repeat("-", 120))
    end

    # ==========================================================================
    # STEP 0: INITIAL OPTIMALITY CHECK
    # ==========================================================================
    # Gradiente Projetado (para limites de caixa)
    # Lógica: x - proj(x - grad)
    proj_dir = clamp.(x .- grad, prob.xl, prob.xu)
    gpnorm = norm(proj_dir .- x, Inf)

    # Viabilidade
    feas_violation = (m > 0) ? maximum(max.(0.0, h_val)) : 0.0
    # Viabilidade Bounds
    bound_violation = maximum(max.(0.0, x .- prob.xu, prob.xl .- x))

    tol_check = 1e-6
    if (gpnorm < tol_check) && (feas_violation < tol_check) && (bound_violation < tol_check)
        if params_slp.verbose
            ;
            println("✅ CONVERGED AT START: KKT satisfied.");
        end
        return x, lambda, theta, 0, 0, x_accepted, full_log
    end
    # ==========================================================================
    # MAIN LOOP (Condição idêntica ao MATLAB)
    # ==========================================================================
    while ((countG < params_slp.maxcount || countF < params_slp.maxcount) && (countS < params_slp.maxcount) && (iter < params_slp.maxiter))

        params_slp.debugverbose && println("\n🔍 [DEBUG-SLP] === Iteration $(iter+1) started ===")
        params_slp.debugverbose && println("🔍 [DEBUG-SLP] Current State: F(x) = $F, aredfsb = $feas_violation, delta = $delta")

        # -------------------------------------------------
        # 1. Definir e Resolver Modelo LP
        # -------------------------------------------------
        model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => params_slp.output_flag))

        @variable(model, s[i = 1:n])
        @objective(model, Min, dot(grad, s))

        # Restrições Linearizadas: jac * s <= -h(x)
        lp_cons = nothing
        if m > 0
            lp_cons = @constraint(model, jac * s .== -h_val)
        end

        # Box Constraints Trust Region
        sL = max.(prob.xl .- x, -delta)
        sU = min.(prob.xu .- x, delta)
        for i in 1:n
            set_lower_bound(s[i], sL[i]);
            set_upper_bound(s[i], sU[i]);
        end

        optimize!(model)
        status = termination_status(model)

        s_sol = zeros(n)
        lp_obj_val = 0.0
        current_phase = :optimization

        if status == MOI.OPTIMAL
            s_sol = value.(s)
            lp_obj_val = objective_value(model)
            if m > 0
                lambda = dual.(lp_cons);
            end
            params_slp.debugverbose && println("🔍 [DEBUG-SLP] Subproblem solved optimally. lp_obj_val = $lp_obj_val")
        else
            params_slp.debugverbose && println("⚠️ [DEBUG-SLP] Subproblem infeasible/failed. Entering RESTORATION phase.")
            # Modo de Recuperação (Relaxamento)
            current_phase = :restoration
            model2 = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => params_slp.output_flag))
            @variable(model2, s2[i = 1:n])
            @variable(model2, z >= 0)

            sL2 = max.(prob.xl .- x, -0.8 .* delta)
            sU2 = min.(prob.xu .- x, 0.8 .* delta)
            for i in 1:n
                set_lower_bound(s2[i], sL2[i]);
                set_upper_bound(s2[i], sU2[i]);
            end

            @objective(model2, Min, z)
            if m > 0
                @constraint(model2, jac * s2 .== -h_val .+ z);
            end

            # # Restaura a influência do gradiente para evitar passos em falso
            # @objective(model2, Min, dot(grad, s2) + 1e5 * z)
            # if m > 0
            #     @constraint(model2, jac * s2 .== -h_val .+ z);
            # end

            optimize!(model2)
            if termination_status(model2) == MOI.OPTIMAL
                s_sol = value.(s2)
                lp_obj_val = dot(grad, s_sol)
                params_slp.debugverbose && println("🔍 [DEBUG-SLP] Restoration solved. lp_obj_val = $lp_obj_val")
            else
                params_slp.debugverbose && println("🚫 [DEBUG-SLP] Restoration failed completely. s_sol = 0")
                s_sol = zeros(n)
            end
        end

        snorm = norm(s_sol, Inf)

        # -------------------------------------------------
        # 2. Calcular Pred e Ared
        # -------------------------------------------------
        x_trial = clamp.(x .+ s_sol, prob.xl, prob.xu)
        F_trial = prob.f(x_trial)
        h_trial = prob.h(x_trial)

        vio_curr = m > 0 ? sum(max.(0.0, h_val)) : 0.0
        vio_trial = m > 0 ? sum(max.(0.0, h_trial)) : 0.0

        # Violação PREDITA pelo modelo linear: h(x) + J*s <= 0
        if m > 0
            lin_approx = h_val .+ jac * s_sol
            vio_lin_trial = sum(max.(0.0, lin_approx))
        else
            vio_lin_trial = 0.0
        end

        predopt = -lp_obj_val
        predfsb = vio_curr - vio_lin_trial
        aredopt = F - F_trial
        aredfsb = vio_curr - vio_trial

        params_slp.debugverbose && println("\n🔍 [DEBUG-SLP] --- Pred & Ared Breakdown ---")
        params_slp.debugverbose && println("   [Opt]  aredopt (F - F_trial)      = $aredopt  |  predopt (-lp_obj) = $predopt")
        params_slp.debugverbose && println("   [Fsb]  aredfsb (vio_curr - trial) = $aredfsb  |  predfsb (vio_curr) = $predfsb")

        # Atualização Theta
        thetaMin = min(theta1, theta2)
        thetaLarge = (1 + (1e6 / ((iter + 1)^(1.1)))) * thetaMin

        if predopt > 0.5 * predfsb
            thetaSup = 1.0
        else
            denom = predfsb - predopt
            thetaSup = abs(denom) < 1e-12 ? 1e10 : (0.5 * predfsb) / denom
        end
        theta_old = theta
        theta = min(thetaLarge, thetaSup)
        theta = min(theta, thetaMax)
        theta2 = theta1;
        theta1 = theta


        params_slp.debugverbose && println("🔍 [DEBUG-SLP] Theta Logic: predopt > 0.5*predfsb? $(predopt > 0.5 * predfsb)")
        params_slp.debugverbose && println("🔍 [DEBUG-SLP] Theta Updated: $theta_old -> $theta")

        pred = theta * predopt + (1 - theta) * predfsb
        ared = theta * aredopt + (1 - theta) * aredfsb

        rho = abs(pred) < 1e-12 ? (ared >= 0 ? Inf : -Inf) : (ared / pred)
        params_slp.debugverbose && println("🔍 [DEBUG-SLP] Final Ared = $ared  |  Final Pred = $pred")
        params_slp.debugverbose && println("🔍 [DEBUG-SLP] Ratio (ared/pred) ρ = $rho")

        # -------------------------------------------------
        # 3. Aceitação e Atualização
        # -------------------------------------------------
        step_accepted = false
        threshold = params_slp.eta * pred
        if abs(pred) < 1e-12
            step_accepted = (ared >= 0)
            params_slp.debugverbose && println("🔍 [DEBUG-SLP] Check: pred is ~0. Accepted if ared >= 0. Result: $step_accepted")
        else
            # step_accepted = (ared >= params_slp.eta * pred)
            step_accepted = (ared >= threshold)
            params_slp.debugverbose && println("🔍 [DEBUG-SLP] Check: ared ($ared) >= threshold ($threshold). Result: $step_accepted")
        end

        # Gravando a tentativa antes de atualizar
        delta_log = delta isa Vector ? copy(delta) : delta
        log_entry = StepLog_SLP(
            iter + 1,       # Número da iteração atual
            copy(x),        # De onde saiu
            copy(x_trial),  # Para onde tentou ir
            step_accepted ? :accepted : :rejected,
            current_phase,
            delta_log,
            ared,
            pred,
        )
        push!(full_log, log_entry)
        # ===================================

        accept_symbol = step_accepted ? "✓" : "✗"

        # if params_slp.verbose
        #     # Formatação compacta para debug
        #     counters_str = "$(countG)|$(countF)|$(countS)"
        #     @printf("%-4d %-12.5e %-8.2e %-8.2e %-10.3e %-6s %-12s\n", iter+1, F, delta, snorm, ared, accept_symbol, counters_str)
        # end
        slope_val = dot(grad, s_sol)
        if step_accepted
            params_slp.debugverbose && println("✅ [DEBUG-SLP] >> STEP ACCEPTED <<")
            x = x_trial
            Fold=F
            F = F_trial
            grad = prob.∇f(x)
            h_val = h_trial
            jac = prob.∇h(x)
            thetaMax = 1.0

            # Trust Region Update
            delta_old = delta
            if ared >= params_slp.rho * pred
                if params_slp.backtracking_quadratic
                    delta = quadratic_backtracking_step!(
                                            delta, Fold, F_trial, slope_val, params_slp, snorm, norm(s_sol)^2, :increase;
                                            anisotropic=params_slp.anisotropic_trust_region, s_vec=s_sol, grad=grad
                                        )
                else
                delta = min.(params_slp.alphaA * delta, 1.0)
                end
                params_slp.debugverbose && println("📈 [DEBUG-SLP] Great step! Trust Region expanded: $delta_old -> $delta (ρ >= $(params_slp.rho))")
            else
                params_slp.debugverbose && println("🔄 [DEBUG-SLP] Good step, but not great. Trust Region kept at: $delta (ρ < $(params_slp.rho))")
            end
            delta = max.(delta, 1e-4)

            # Grava no histórico limpo
            push!(x_accepted, copy(x))
            itrej = 0

        else
            delta_old = delta
            # Passo rejeitado: reduz a região de confiança
            if params_slp.backtracking_quadratic
                
                # === NOVA LÓGICA: Alternância Shrink / Reshape ===
                # Se a iteração for par, usa :shrink. Se for ímpar, usa :reshape.
                current_mode = (iter % 2 == 0) ? :shrink : :reshape
                
                params_slp.debugverbose && println("📉 [DEBUG-SLP] Redução Anisotrópica - Modo: $current_mode (Iter: $iter)")

                delta = quadratic_backtracking_step!(
                    delta, F, F_trial, slope_val, params_slp, snorm, norm(s_sol)^2, :decrease;
                    anisotropic=params_slp.anisotropic_trust_region, 
                    s_vec=s_sol, 
                    grad=grad,
                    mode=current_mode # Passando o modo dinâmico aqui
                )
            else
                # Fallback tradicional (seguro para escalar e vetor)
                delta = max.(params_slp.alphaR * snorm, 0.1 .* delta)
            end
            params_slp.debugverbose && println("🚫 [DEBUG-SLP] >> STEP REJECTED <<")
            params_slp.debugverbose && println("📉 [DEBUG-SLP] Trust Region shrunk: $delta_old -> $delta")
            itrej += 1
        end

        iter += 1
        # -------------------------------------------------
        # 4. Cálculo de Métricas de Parada
        # -------------------------------------------------
        # Gradiente Lagrangeano Projetado
        Lgrad = copy(grad)
        if m > 0 && !iszero(lambda)
            Lgrad .+= jac' * lambda
        end

        proj_dir_k = clamp.(x .- Lgrad, prob.xl, prob.xu)
        gpnorm = norm(proj_dir_k .- x, Inf)

        # Atualização dos Contadores (igual ao MATLAB)
        if gpnorm <= params_slp.tolG
            countG += 1
        else
            countG = 0
        end

        if abs(ared) <= params_slp.tolF
            countF += 1
        else
            countF = 0
        end

        if snorm <= params_slp.tolS
            countS += 1
        else
            countS = 0
        end

        # -------------------------------------------------
        # 5. Log da Iteração (Com os contadores)
        # -------------------------------------------------
        if params_slp.verbose
            # Formata string "G|F|S"
            counters_str = "$(countG)|$(countF)|$(countS)"
            delta_disp = delta isa Vector ? maximum(delta) : delta
            @printf("%-4d %-12.5e %-10.3e %-8.2e %-8.2e %-8.3e %-8.3e %-10.3e %-6s %-12s\n", iter, F, vio_curr, delta_disp, snorm, ared, pred, theta, accept_symbol, counters_str)
        end
        is_feasible = vio_curr <= 1e-3
            end # Fim do While


    if (countG >= params_slp.maxcount && countF >= params_slp.maxcount)
        if is_feasible
            opstop = 0 # KKT satisfied & feasible
            if params_slp.verbose
                println("\n✅ Converged: KKT & Reduction criteria met.");
            end
        else
            opstop = 3 # Infeasible stationary point
            if params_slp.verbose
                println("\n⚠️ Converged to an INFEASIBLE stationary point (||h(x)|| = $vio_curr).");
            end
        end
    elseif (countS >= params_slp.maxcount)
        if is_feasible
            opstop = 1 # Step norm small & feasible
            if params_slp.verbose
                println("\n✅ Converged: Step size small.");
            end
        else
            opstop = 4 # Stalled infeasible
            if params_slp.verbose
                println("\n❌ Stalled: Step size small but point is INFEASIBLE (||h(x)|| = $vio_curr).");
            end
        end
    else
        opstop = 2 # Max iter
        if params_slp.verbose
            println("\n⚠️ Max iterations reached.");
        end
    end
    # end # Fim do While
    return x, lambda, theta, iter, opstop, x_accepted, full_log
end

end