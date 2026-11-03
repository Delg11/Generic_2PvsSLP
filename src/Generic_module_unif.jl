module Generic_module_unif

export solve_unif_trust_region

using NLPModels, Printf, JuMP, Gurobi, LinearAlgebra, RipQP,SparseArrays, QuadraticModels
using ..SharedTypes
using ..Generic_module_ComputeB
"""
Log message helper
"""
function log_message!(msg; verbose=true)
    if verbose
        println(msg)
    end
end

# ==============================================================================
# MAIN UNIF SOLVER (Logic adapted from StrUNIF.jl)
# ==============================================================================
function solve_unif_trust_region(prob::OptimizationProblem, x0::Vector{Float64}, params_unif::UNIFParams; history::Bool=false)
    # --- Inicialização ---
    x = clamp.(copy(x0), prob.xl, prob.xu)
    # n, m = prob.n, prob.m
    n = length(x)
    m = prob.ncon
    s_sol = zeros(n)
    F = prob.f(x)
    grad = prob.∇f(x)
    grad_old = copy(grad)
    h_val = prob.h(x)
    jac = prob.∇h(x)
    jac_old = copy(jac)
    if params_unif.anisotropic_trust_region
        delta = fill(params_unif.delta0, n)
    else
        delta = params_unif.delta0
    end
    iter = 0
    itrej = 0
    countG, countF, countS = 0, 0, 0
    opstop = -1

    is_feasible=false
    # --- HISTÓRICOS ---
    # 1. Apenas os passos aceitos (Trajetória "Limpa")
    x_accepted = history ? Vector{Vector{Float64}}() : Vector{Float64}[]
    history && push!(x_accepted, copy(x))

    # 2. Histórico Completo (Tentativas, Rejeições, Deltas)
    full_log = history ? Vector{StepLog_UNIF}() : StepLog_UNIF[]

    theta = 1.0;
    theta1 = 1.0;
    theta2 = 1.0;
    thetaMax = 1.0
    # GUROBI_ENV = Gurobi.Env(output_flag=0)
    lambda = zeros(m)

    if params_unif.verbose
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
    proj_step = compute_gradient_projection(
        x, grad, jac, lambda, prob.xl, prob.xu;
        norm_type = params_unif.norm_gpnorm,
        strategy = params_unif.gpnorm_strategy,
        max_iter = params_unif.gpnorm_max_iter,
        tol = params_unif.gpnorm_tol,
        debug = params_unif.gpnorm_debug
    )
    gpnorm = norm(proj_step, params_unif.norm_gpnorm)

    # Viabilidade
    feas_violation = (m > 0) ? norm(h_val, Inf) : 0.0
    # Viabilidade Bounds
    bound_violation = maximum(max.(0.0, x .- prob.xu, prob.xl .- x))

    tol_check = 1e-6
    if (gpnorm < tol_check) && (feas_violation < tol_check) && (bound_violation < tol_check)
        if params_unif.verbose
            ;
            println("✅ CONVERGED AT START: KKT satisfied.");
        end
        return x, lambda, theta, 0, 0, x_accepted, full_log
    end
    # ==========================================================================
    # MAIN LOOP
    # ==========================================================================

    s_last_accepted = zeros(n) # Isola o vetor de passo correto
    B = spdiagm(0 => ones(n))  # Declaração inicial no escopo superior
    g_eff = -proj_step
    g_eff_old = copy(g_eff)
    d_curv = ones(n)
    jac_diff_l1 = zeros(n)

    sub_time_limit = params_unif.max_time > 0 ? min(5.0, params_unif.max_time) : 5.0
    qp_method = params_unif.use_quadratic ? -1 : 1
    qp_presolve = params_unif.use_quadratic ? -1 : 0

    # Modelos JuMP reutilizáveis para os subproblemas
    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV),
        MOI.Silent() => (params_unif.output_flag == 0),
        "Threads" => 1,
        "Method" => qp_method,
        "Presolve" => qp_presolve,
        "NonConvex" => 0,
        "TimeLimit" => sub_time_limit
    ))
    @variable(model, s[1:n])
    if !params_unif.use_quadratic
        @objective(model, Min, sum(0.0 * s[j] for j in 1:n))
    end
    lp_cons = m > 0 ? @constraint(model, [i = 1:m], 0.0 * sum(s[j] for j in 1:n) == 0.0) : nothing

    model2 = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV),
        MOI.Silent() => (params_unif.output_flag == 0),
        "Threads" => 1,
        "Method" => 1,
        "Presolve" => 0,
        "TimeLimit" => sub_time_limit
    ))
    @variable(model2, s2[1:n])
    if m > 0
        @variable(model2, z_plus[1:m] >= 0)
        @variable(model2, z_minus[1:m] >= 0)
        @objective(model2, Min, sum(z_plus) + sum(z_minus))
        resto_cons = @constraint(model2, [i = 1:m], 0.0 * sum(s2[j] for j in 1:n) - z_plus[i] + z_minus[i] == 0.0)
    else
        @objective(model2, Min, 0.0)
        resto_cons = nothing
    end

    t_start_alg = time()
    while ((countG < params_unif.maxcount || countF < params_unif.maxcount) && (countS < params_unif.maxcount) && (iter < params_unif.maxiter))
        if params_unif.max_time > 0 && (time() - t_start_alg) > params_unif.max_time
            params_unif.verbose && println("⏱️ TIMEOUT: Execution exceeded $(params_unif.max_time)s limit")
            return x, lambda, theta, iter, 5, x_accepted, full_log
        end
        if any(!isfinite, x)
            params_unif.debugverbose && println("❌ [DEBUG-UNIF] Breakdown: NaN/Inf detected in x vector.")
            return x, lambda, theta, iter, 4, x_accepted, full_log
        end

        params_unif.debugverbose && println("\n🔍 [DEBUG-UNIF] === Iteration $(iter+1) started ===")
        params_unif.debugverbose && println("🔍 [DEBUG-UNIF] Current State: F(x) = $F, aredfsb = $feas_violation, delta = $delta")

        if params_unif.use_quadratic
            if norm(s_last_accepted) > 1e-12 # Usa apenas o passo validado
                if m > 0
                    y_diff = (grad .- grad_old) .+ ((jac .- jac_old)' * lambda)
                else
                    y_diff = grad .- grad_old
                end
                B = Generic_module_ComputeB.compute_B(params_unif.B_update_strategy, n, s_last_accepted, y_diff, params_unif.σ, prob, x, lambda, B)
            else
                B = params_unif.σ * spdiagm(0 => ones(n))
            end
        end

        # Box Constraints Trust Region
        sL = zeros(n)
        sU = zeros(n)
        for i in 1:n
            d_val = (delta isa Vector) ? delta[i] : delta
            d_val = isfinite(d_val) ? d_val : params_unif.delta0
            xi = isfinite(x[i]) ? x[i] : 0.0
            lx = isfinite(prob.xl[i]) ? (prob.xl[i] - xi) : -Inf
            ux = isfinite(prob.xu[i]) ? (prob.xu[i] - xi) : Inf
            lb = max(lx, -d_val)
            ub = min(ux, d_val)
            sL[i] = isfinite(lb) ? lb : -d_val
            sU[i] = isfinite(ub) ? ub : d_val
            if sL[i] > sU[i]; sL[i] = sU[i]; end
        end

        s_sol = zeros(n)
        lp_obj_val = 0.0
        current_phase = :optimization
        success_opt = false



        # 1. Preparação da Matriz Hessiana
        local Hqp
        params_unif.use_quadratic && (Hqp = B)

        # 2. Resolução do Subproblema Principal
        if params_unif.use_quadratic && params_unif.quadratic_solver == :ripqp
            # -------------------------------------------------
            # Fluxo RipQP
            # -------------------------------------------------
            qm = QuadraticModel(grad, Hqp; A = jac, lcon = -h_val, ucon = -h_val, lvar = sL, uvar = sU, c0 = 0.0)
            
            stats = ripqp(qm, display=(params_unif.output_flag == 1))
            # println(stats)
            if stats.status == :first_order || stats.status == :acceptable
                s_sol = stats.solution
                lp_obj_val = stats.objective
                if m > 0
                    lambda = stats.multipliers[1:m]
                end
                success_opt = true
                params_unif.debugverbose && println("🔍 [DEBUG-SQP] Subproblem solved optimally (RipQP). obj_val = $lp_obj_val")
            end
            
        else
            # -------------------------------------------------
            # Fluxo Original (JuMP + Gurobi)
            # -------------------------------------------------
            for i in 1:n
                set_lower_bound(s[i], sL[i])
                set_upper_bound(s[i], sU[i])
            end

            # Safeguard against non-finite coefficients (NaN/Inf)
            if any(!isfinite, grad) || (params_unif.use_quadratic && (Hqp !== nothing && any(!isfinite, Hqp)))
                success_opt = false
            else
                if params_unif.use_quadratic
                    @objective(model, Min, 0.5 * dot(s, Hqp * s) + dot(grad, s))
                else
                    for j in 1:n
                        set_objective_coefficient(model, s[j], grad[j])
                    end
                end

                # Restrições Linearizadas: jac * s == -h(x)
                if m > 0
                    for i in 1:m
                        for j in 1:n
                            set_normalized_coefficient(lp_cons[i], s[j], jac[i, j])
                        end
                        set_normalized_rhs(lp_cons[i], -h_val[i])
                    end
                end

                if params_unif.max_time > 0
                    rem_t = params_unif.max_time - (time() - t_start_alg)
                    rem_t <= 0.0 && return x, lambda, theta, iter, 5, x_accepted, full_log
                    set_optimizer_attribute(model, "TimeLimit", clamp(rem_t, 0.5, 5.0))
                end
                optimize!(model)
                
                if termination_status(model) == MOI.OPTIMAL
                    s_sol = [value(s[i]) for i in 1:n]
                    lp_obj_val = objective_value(model)
                    if has_duals(model)
                        if m > 0
                            lambda = [dual(lp_cons[i]) for i in 1:m]
                        end
                    else
                        if m > 0
                            lambda = zeros(m)
                        end
                    end
                    success_opt = true
                    mode_str = params_unif.use_quadratic ? "SQP" : "UNIF"
                    params_unif.debugverbose && println("🔍 [DEBUG-$mode_str] Subproblem solved optimally (Gurobi). obj_val = $lp_obj_val")
                end
            end
        end

        # 3. Restauration Phase (if subproblem fails)
        if !success_opt
            params_unif.debugverbose && println("⚠️ [DEBUG-UNIF] Subproblem infeasible/failed. Entering RESTORATION phase.")
            
            current_phase = :restoration
            sL2 = zeros(n)
            sU2 = zeros(n)
            for i in 1:n
                d_val = (delta isa Vector) ? 0.8 * delta[i] : 0.8 * delta
                d_val = isfinite(d_val) ? d_val : 0.8 * params_unif.delta0
                xi = isfinite(x[i]) ? x[i] : 0.0
                lx = isfinite(prob.xl[i]) ? (prob.xl[i] - xi) : -Inf
                ux = isfinite(prob.xu[i]) ? (prob.xu[i] - xi) : Inf
                lb = max(lx, -d_val)
                ub = min(ux, d_val)
                sL2[i] = isfinite(lb) ? lb : -d_val
                sU2[i] = isfinite(ub) ? ub : d_val
                if sL2[i] > sU2[i]; sL2[i] = sU2[i]; end
                set_lower_bound(s2[i], sL2[i])
                set_upper_bound(s2[i], sU2[i])
            end

            if m > 0
                for i in 1:m
                    for j in 1:n
                        set_normalized_coefficient(resto_cons[i], s2[j], jac[i, j])
                    end
                    set_normalized_rhs(resto_cons[i], -h_val[i])
                end
            end

            if params_unif.max_time > 0
                rem_t = params_unif.max_time - (time() - t_start_alg)
                rem_t <= 0.0 && return x, lambda, theta, iter, 5, x_accepted, full_log
                set_optimizer_attribute(model2, "TimeLimit", clamp(rem_t, 0.5, 5.0))
            end
            optimize!(model2)
            
            if termination_status(model2) == MOI.OPTIMAL
                s_sol = [value(s2[i]) for i in 1:n]
                lp_obj_val = dot(grad, s_sol)
                params_unif.debugverbose && println("🔍 [DEBUG-UNIF] Restoration solved. lp_obj_val = $lp_obj_val")
            else
                params_unif.debugverbose && println("🚫 [DEBUG-UNIF] Restoration failed completely. s_sol = 0")
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


        # Norma L1 para restrições de igualdade (abs)
        vio_curr = m > 0 ? sum(abs.(h_val)) : 0.0
        vio_trial = m > 0 ? sum(abs.(h_trial)) : 0.0

        # Violação PREDITA pelo modelo linear: h(x) + J*s == 0
        if m > 0
            lin_approx = h_val .+ jac * s_sol
            vio_lin_trial = sum(abs.(lin_approx))
        else
            vio_lin_trial = 0.0
        end

        predopt = -lp_obj_val
        predfsb = vio_curr - vio_lin_trial
        aredopt = F - F_trial
        aredfsb = vio_curr - vio_trial

        params_unif.debugverbose && println("\n🔍 [DEBUG-UNIF] --- Pred & Ared Breakdown ---")
        params_unif.debugverbose && println("   [Opt]  aredopt (F - F_trial)      = $aredopt  |  predopt (-lp_obj) = $predopt")
        params_unif.debugverbose && println("   [Fsb]  aredfsb (vio_curr - trial) = $aredfsb  |  predfsb (vio_curr) = $predfsb")

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


        params_unif.debugverbose && println("🔍 [DEBUG-UNIF] Theta Logic: predopt > 0.5*predfsb? $(predopt > 0.5 * predfsb)")
        params_unif.debugverbose && println("🔍 [DEBUG-UNIF] Theta Updated: $theta_old -> $theta")

        pred = theta * predopt + (1 - theta) * predfsb
        ared = theta * aredopt + (1 - theta) * aredfsb

        rho = abs(pred) < 1e-12 ? (ared >= 0 ? Inf : -Inf) : (ared / pred)
        params_unif.debugverbose && println("🔍 [DEBUG-UNIF] Final Ared = $ared  |  Final Pred = $pred")
        params_unif.debugverbose && println("🔍 [DEBUG-UNIF] Ratio (ared/pred) ρ = $rho")

        # -------------------------------------------------
        # 3. Aceitação e Atualização
        # -------------------------------------------------
        step_accepted = false
        threshold = params_unif.eta * pred
        if abs(pred) < 1e-12
            step_accepted = (ared >= 0)
            params_unif.debugverbose && println("🔍 [DEBUG-UNIF] Check: pred is ~0. Accepted if ared >= 0. Result: $step_accepted")
        else
            # step_accepted = (ared >= params_unif.eta * pred)
            step_accepted = (ared >= threshold)
            params_unif.debugverbose && println("🔍 [DEBUG-UNIF] Check: ared ($ared) >= threshold ($threshold). Result: $step_accepted")
        end

        # Gravando a tentativa antes de atualizar se history ativo
        if history
            delta_log = delta isa Vector ? copy(delta) : delta
            log_entry = StepLog_UNIF(
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
        end
        # ===================================

        accept_symbol = step_accepted ? "✓" : "✗"

        # if params_unif.verbose
        #     # Formatação compacta para debug
        #     counters_str = "$(countG)|$(countF)|$(countS)"
        #     @printf("%-4d %-12.5e %-8.2e %-8.2e %-10.3e %-6s %-12s\n", iter+1, F, delta, snorm, ared, accept_symbol, counters_str)
        # end
        slope_val = dot(grad, s_sol)
        use_new_atr = params_unif.anisotropic_trust_region && params_unif.atr_strategy != :legacy_parabolic
        local proj_step_k::Vector{Float64}
        proj_step_ready = false

        if step_accepted
            params_unif.debugverbose && println("✅ [DEBUG-UNIF] >> STEP ACCEPTED <<")
            x = x_trial
            Fold = F
            F = F_trial
            grad_old = grad
            grad = prob.∇f(x)
            h_val = h_trial
            jac_old = jac
            jac = prob.∇h(x)
            vio_curr = vio_trial
            feas_violation = (m > 0) ? norm(h_val, Inf) : 0.0
            theta_used = theta
            thetaMax = 1.0
            # Atualiza o cache do passo para a equação secante da próxima iteração
            s_last_accepted = copy(s_sol)

            if use_new_atr
                proj_step_k = compute_gradient_projection(
                    x, grad, jac, lambda, prob.xl, prob.xu;
                    norm_type = params_unif.norm_gpnorm,
                    strategy = params_unif.gpnorm_strategy,
                    max_iter = params_unif.gpnorm_max_iter,
                    tol = params_unif.gpnorm_tol,
                    debug = params_unif.gpnorm_debug
                )
                proj_step_ready = true
                g_eff_old = g_eff
                g_eff = -proj_step_k
                if m > 0
                    @inbounds for j in 1:n
                        col_sum = 0.0
                        for i in 1:m
                            col_sum += abs(jac[i, j] - jac_old[i, j])
                        end
                        jac_diff_l1[j] = col_sum
                    end
                else
                    fill!(jac_diff_l1, 0.0)
                end
            end

            # Trust Region Update
            delta_old = copy(delta)

            if !params_unif.parabolic_heuristic && !use_new_atr
                if params_unif.strong_agreement_rule
                    # Lógica tradicional baseada na razão de redução
                    if ared >= params_unif.rho * pred
                        delta = min.(params_unif.alphaA * delta, params_unif.δmax)
                        params_unif.debugverbose && println("📈 [DEBUG-UNIF] Great step! Trust Region expanded: $delta_old -> $delta (ρ >= $(params_unif.rho))")
                    else
                        params_unif.debugverbose && println("🔄 [DEBUG-UNIF] Good step, but not great. Trust Region kept at: $delta (ρ < $(params_unif.rho))")
                    end
                else
                    # Expansão incondicional (Ablação do SAR ativada)
                    delta = min.(params_unif.alphaA * delta, params_unif.δmax)
                    params_unif.debugverbose && println("📈 [DEBUG-UNIF] Trust Region unconditionally expanded: $delta_old -> $delta")
                end
            else
                # Lógica de expansão via modelo quadrático / estratégia anisotrópica
                params_unif.debugverbose && println("🔍 [DEBUG-UNIF] Applying Parabolic/ATR Strategy for Trust Region expansion.")
                delta = parabolic_heuristic_step!(
                    delta, Fold, F_trial, slope_val, params_unif, snorm, norm(s_sol)^2, :increase;
                    anisotropic=params_unif.anisotropic_trust_region, 
                    s_vec=s_sol, 
                    grad=grad_old,                
                    min_reduction_ratio=params_unif.parabolic_min_reduction_ratio,
                    max_reduction_ratio=params_unif.parabolic_max_reduction_ratio,
                    min_increase_ratio=params_unif.parabolic_min_increase_ratio,
                    max_increase_ratio=params_unif.parabolic_max_increase_ratio,
                    ared_val=ared,
                    pred_val=pred,
                    g_prev=g_eff_old,
                    g_curr=g_eff,
                    jac_diff_col_l1=jac_diff_l1,
                    theta_merit=theta_used,
                    d_curv=d_curv,
                )
            end
            
            # Limite inferior de segurança para o delta
            delta_floor = use_new_atr ? params_unif.δmin : 1e-4
            delta = max.(delta, delta_floor)

            # Grava no histórico limpo se history ativo
            history && push!(x_accepted, copy(x))
            itrej = 0

        else
            delta_old = copy(delta)
            # Passo rejeitado: reduz a região de confiança
            if params_unif.parabolic_heuristic || use_new_atr
                
                # === Alternância Shrink / Reshape ===
                # Se a iteração for par, usa :shrink. Se for ímpar, usa :reshape.
                current_mode = (iter % 2 == 0) ? :shrink : :reshape
                
                params_unif.debugverbose && println("📉 [DEBUG-UNIF] Redução Anisotrópica - Modo: $current_mode (Iter: $iter)")

                delta = parabolic_heuristic_step!(
                    delta, F, F_trial, slope_val, params_unif, snorm, norm(s_sol)^2, :decrease;
                    anisotropic=params_unif.anisotropic_trust_region, 
                    s_vec=s_sol, 
                    grad=grad,
                    mode=current_mode,
                    min_reduction_ratio=params_unif.parabolic_min_reduction_ratio,
                    max_reduction_ratio=params_unif.parabolic_max_reduction_ratio,
                    min_increase_ratio=params_unif.parabolic_min_increase_ratio,
                    max_increase_ratio=params_unif.parabolic_max_increase_ratio,
                    ared_val=ared,
                    pred_val=pred,
                    g_prev=g_eff,
                    d_curv=d_curv,
                )
            else
                # Fallback tradicional (seguro para escalar e vetor)
                delta = max.(params_unif.alphaR * snorm, 0.1 .* delta)
            end
            params_unif.debugverbose && println("🚫 [DEBUG-UNIF] >> STEP REJECTED <<")
            params_unif.debugverbose && println("📉 [DEBUG-UNIF] Trust Region shrunk: $delta_old -> $delta")
            itrej += 1
        end

        iter += 1
        # -------------------------------------------------
        # 4. Cálculo de Métricas de Parada
        # -------------------------------------------------
        if !proj_step_ready
            proj_step_k = compute_gradient_projection(
                x, grad, jac, lambda, prob.xl, prob.xu;
                norm_type = params_unif.norm_gpnorm,
                strategy = params_unif.gpnorm_strategy,
                max_iter = params_unif.gpnorm_max_iter,
                tol = params_unif.gpnorm_tol,
                debug = params_unif.gpnorm_debug
            )
            if use_new_atr && step_accepted
                g_eff = -proj_step_k
            end
        end
        gpnorm = norm(proj_step_k, params_unif.norm_gpnorm)

        # Atualização dos Contadores (igual ao MATLAB)
        if gpnorm <= params_unif.tolG
            countG += 1
        else
            countG = 0
        end

        if abs(ared) <= params_unif.tolF
            countF += 1
        else
            countF = 0
        end

        if snorm <= params_unif.tolS
            countS += 1
        else
            countS = 0
        end

        # -------------------------------------------------
        # 5. Log da Iteração (Com os contadores)
        # -------------------------------------------------
        if params_unif.verbose
            # Formata string "G|F|S"
            counters_str = "$(countG)|$(countF)|$(countS)"
            delta_disp = delta isa Vector ? maximum(delta) : delta
            @printf("%-4d %-12.5e %-10.3e %-8.2e %-8.2e %-8.3e %-8.3e %-10.3e %-6s %-12s\n", iter, F, vio_curr, delta_disp, snorm, ared, pred, theta, accept_symbol, counters_str)
        end
        is_feasible = (m > 0) ? (norm(h_val, Inf) <= 1e-3) : true
            end # Fim do While


    if (countG >= params_unif.maxcount && countF >= params_unif.maxcount)
        if is_feasible
            opstop = 0 # KKT satisfied & feasible
            if params_unif.verbose
                println("\n✅ Converged: KKT & Reduction criteria met.");
            end
        else
            opstop = 3 # Infeasible stationary point
            if params_unif.verbose
                println("\n⚠️ Converged to an INFEASIBLE stationary point (||h(x)|| = $vio_curr).");
            end
        end
    elseif (countS >= params_unif.maxcount)
        if is_feasible
            opstop = 1 # Step norm small & feasible
            if params_unif.verbose
                println("\n✅ Converged: Step size small.");
            end
        else
            opstop = 4 # Stalled infeasible
            if params_unif.verbose
                println("\n❌ Stalled: Step size small but point is INFEASIBLE (||h(x)|| = $vio_curr).");
            end
        end
    else
        opstop = 2 # Max iter
        if params_unif.verbose
            println("\n⚠️ Max iterations reached.");
        end
    end
    # end # Fim do While
    return x, lambda, theta, iter, opstop, x_accepted, full_log
end

end