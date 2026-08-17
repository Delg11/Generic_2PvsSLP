module Generic_module_unif

export solve_unif_trust_region

using NLPModels, Printf, JuMP, Gurobi, LinearAlgebra, RipQP,SparseArrays, QuadraticModels
using ..SharedTypes
include("Generic_module_ComputeB.jl")
"""
Log message helper
"""
function log_message!(msg; verbose=true)
    if verbose
        println(msg)
    end
end


function parabolic_heuristic_step!(
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
                    return (δ_current isa Vector) ? fill(d_val, length(δ_current)) : d_val
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

function compute_gradient_projection(
    x::Vector{Float64},
    Fgrad::Vector{Float64},
    jac_h::AbstractMatrix{Float64},
    λ::Vector{Float64},
    xl::Vector{Float64},
    xu::Vector{Float64};
    norm_type::Union{Float64, Int} = Inf,
    strategy::Symbol=:dykstra,
    max_iter::Int=500,
    tol::Float64=1e-8,
    debug::Bool=false
)
    n = length(x)
    m = size(jac_h, 1)
    
    run_lagrangian = (strategy == :lagrangian) || debug
    run_qp         = (strategy == :qp)         || debug
    run_dykstra    = (strategy == :dykstra)    || debug

    buf_lag = run_lagrangian ? zeros(n) : Float64[]
    buf_qp  = run_qp         ? zeros(n) : Float64[]
    buf_dyk = run_dykstra    ? zeros(n) : Float64[]
    qp_succeeded = true

    # 1. Método: Lagrangiano (dependente de lambda)
    if run_lagrangian
        if m > 0
            Lgrad = Fgrad .+ jac_h' * λ
        else
            Lgrad = Fgrad
        end
        for i in 1:n
            buf_lag[i] = clamp(x[i] - Lgrad[i], xl[i], xu[i]) - x[i]
        end
    end

    if run_qp || run_dykstra
        v = x .- Fgrad
        b_target = m > 0 ? jac_h * x : Float64[]

        function execute_dykstra!(out_buffer)
            p_dyk = copy(v)
            q = zeros(n)
            r = zeros(n)
            y_val = zeros(n)
            
            local fact_J
            if m > 0
                J_Jt = jac_h * jac_h'
                fact_J = lu(J_Jt + 1e-12 * I)
            end
            
            for k in 1:max_iter
                p_old = copy(p_dyk)
                
                # Projeção 1: Limites da caixa (Box)
                for i in 1:n
                    val = p_dyk[i] + q[i]
                    y_val[i] = clamp(val, xl[i], xu[i])
                    q[i] = val - y_val[i]
                end
                
                # Projeção 2: Hiperplanos
                if m > 0
                    y_plus_r = y_val .+ r
                    res = jac_h * y_plus_r .- b_target
                    lambda_proj = fact_J \ res
                    
                    p_dyk .= y_plus_r .- jac_h' * lambda_proj
                    r .= y_plus_r .- p_dyk
                else
                    p_dyk .= y_val
                end
                
                max_diff = norm(p_dyk .- p_old, Inf)
                if max_diff < tol
                    break
                end
            end
            
            for i in 1:n
                out_buffer[i] = p_dyk[i] - x[i]
            end
        end

        # 2. Método: QP via JuMP
        if run_qp
            model = Model(Gurobi.Optimizer)
            set_silent(model)
            @variable(model, xl[i] <= p_var[i=1:n] <= xu[i])
            @objective(model, Min, 0.5 * sum((p_var[i] - v[i])^2 for i in 1:n))
            
            if m > 0
                @constraint(model, jac_h * p_var .== b_target)
            end
            
            optimize!(model)
            status = termination_status(model)
            if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
                p_opt = value.(p_var)
                for i in 1:n
                    buf_qp[i] = p_opt[i] - x[i]
                end
            else
                qp_succeeded = false
                execute_dykstra!(buf_qp) # Fallback
            end
        end

        # 3. Método: Dykstra Nativo
        if run_dykstra
            execute_dykstra!(buf_dyk)
        end
    end

    if debug
        n_lag = norm(buf_lag, norm_type)
        n_qp  = run_qp ? norm(buf_qp, norm_type) : 0.0
        n_dyk = run_dykstra ? norm(buf_dyk, norm_type) : 0.0
        
        println("--- DEBUG: Gradient Projection Comparison ---")
        @printf("Lagrangian norm : %.8e\n", n_lag)
        @printf("QP norm         : %.8e%s\n", n_qp, qp_succeeded ? "" : " (FALLBACK: Dykstra usado)")
        @printf("Dykstra norm    : %.8e\n", n_dyk)
        println("---------------------------------------------")
    end
        
    if strategy == :lagrangian
        return buf_lag
    elseif strategy == :qp
        return buf_qp
    elseif strategy == :dykstra
        return buf_dyk
    else
        error("Estratégia inválida: $strategy")
    end
end

# ==============================================================================
# MAIN UNIF SOLVER (Logic adapted from StrUNIF.jl)
# ==============================================================================
function solve_unif_trust_region(prob::OptimizationProblem, x0::Vector{Float64}, params_unif::UNIFParams)
    # --- Inicialização ---
    x = clamp.(copy(x0), prob.xl, prob.xu)
    # n, m = prob.n, prob.m
    n = length(x)
    m = length(prob.h(x))
    s_sol = zeros(n)
    F = prob.f(x)
    grad = prob.∇f(x)
    grad_old = copy(grad)
    h_val = prob.h(x)
    jac = prob.∇h(x)
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
    x_accepted = Vector{Vector{Float64}}()
    push!(x_accepted, copy(x))

    # 2. Histórico Completo (Tentativas, Rejeições, Deltas)
    full_log = Vector{StepLog_UNIF}()

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
    feas_violation = (m > 0) ? maximum(max.(0.0, h_val)) : 0.0
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

    while ((countG < params_unif.maxcount || countF < params_unif.maxcount) && (countS < params_unif.maxcount) && (iter < params_unif.maxiter))

        params_unif.debugverbose && println("\n🔍 [DEBUG-UNIF] === Iteration $(iter+1) started ===")
        params_unif.debugverbose && println("🔍 [DEBUG-UNIF] Current State: F(x) = $F, aredfsb = $feas_violation, delta = $delta")

        if params_unif.use_quadratic
            if norm(s_last_accepted) > 1e-12 # Usa apenas o passo validado
                y_diff = grad .- grad_old
                B = Generic_module_ComputeB.compute_B(params_unif.B_update_strategy, n, s_last_accepted, y_diff, params_unif.σ, prob, x, lambda, B)
            else
                B = params_unif.σ * spdiagm(0 => ones(n))
            end
        end

        # Box Constraints Trust Region
        sL = max.(prob.xl .- x, -delta)
        sU = min.(prob.xu .- x, delta)

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
            model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), "OutputFlag" => params_unif.output_flag))

            @variable(model, s[i = 1:n])
            
            if params_unif.use_quadratic
                @objective(model, Min, 0.5 * dot(s, Hqp * s) + dot(grad, s))
            else
                @objective(model, Min, dot(grad, s))
            end

            # Restrições Linearizadas: jac * s == -h(x)
            lp_cons = nothing
            if m > 0
                lp_cons = @constraint(model, jac * s .== -h_val)
            end

            for i in 1:n
                set_lower_bound(s[i], sL[i])
                set_upper_bound(s[i], sU[i])
            end

            optimize!(model)
            
            if termination_status(model) == MOI.OPTIMAL
                s_sol = value.(s)
                lp_obj_val = objective_value(model)
                if has_duals(model)
                    if m > 0
                        lambda = dual.(lp_cons)
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

        # 3. Restauration Phase (if subproblem fails)
        if !success_opt
            params_unif.debugverbose && println("⚠️ [DEBUG-UNIF] Subproblem infeasible/failed. Entering RESTORATION phase.")
            
            current_phase = :restoration
            model2 = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), "OutputFlag" => params_unif.output_flag))
            
            @variable(model2, s2[i = 1:n])

            sL2 = max.(prob.xl .- x, -0.8 .* delta)
            sU2 = min.(prob.xu .- x, 0.8 .* delta)
            for i in 1:n
                set_lower_bound(s2[i], sL2[i])
                set_upper_bound(s2[i], sU2[i])
            end

            if m > 0
                @variable(model2, z_plus[1:m] >= 0)
                @variable(model2, z_minus[1:m] >= 0)
                
                @objective(model2, Min, sum(z_plus) + sum(z_minus))
                
                @constraint(model2, jac * s2 .== -h_val .+ z_plus .- z_minus)
            else
                @objective(model2, Min, 0.0)
            end

            optimize!(model2)
            
            if termination_status(model2) == MOI.OPTIMAL
                s_sol = value.(s2)
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

        # Gravando a tentativa antes de atualizar
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
        # ===================================

        accept_symbol = step_accepted ? "✓" : "✗"

        # if params_unif.verbose
        #     # Formatação compacta para debug
        #     counters_str = "$(countG)|$(countF)|$(countS)"
        #     @printf("%-4d %-12.5e %-8.2e %-8.2e %-10.3e %-6s %-12s\n", iter+1, F, delta, snorm, ared, accept_symbol, counters_str)
        # end
        slope_val = dot(grad, s_sol)
        if step_accepted
            params_unif.debugverbose && println("✅ [DEBUG-UNIF] >> STEP ACCEPTED <<")
            x = x_trial
            Fold = F
            F = F_trial
            grad_old = grad
            grad = prob.∇f(x)
            h_val = h_trial
            jac = prob.∇h(x)
            thetaMax = 1.0
            # Atualiza o cache do passo para a equação secante da próxima iteração
            s_last_accepted = copy(s_sol)
            # Trust Region Update
            delta_old = copy(delta)

            if !params_unif.parabolic_heuristic
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
                # Lógica de expansão via modelo quadrático
                params_unif.debugverbose && println("🔍 [DEBUG-UNIF] Applying Parabolic Heuristic for Trust Region expansion.")
                delta = parabolic_heuristic_step!(
                    delta, Fold, F_trial, slope_val, params_unif, snorm, norm(s_sol)^2, :increase;
                    anisotropic=params_unif.anisotropic_trust_region, 
                    s_vec=s_sol, 
                    grad=grad,                
                    min_reduction_ratio=params_unif.parabolic_min_reduction_ratio,
                    max_reduction_ratio=params_unif.parabolic_max_reduction_ratio,
                    min_increase_ratio=params_unif.parabolic_min_increase_ratio,
                    max_increase_ratio=params_unif.parabolic_max_increase_ratio,
                )
            end
            
            # Limite inferior de segurança para o delta
            delta = max.(delta, 1e-4)

            # Grava no histórico limpo
            push!(x_accepted, copy(x))
            itrej = 0

        else
            delta_old = copy(delta)
            # Passo rejeitado: reduz a região de confiança
            if params_unif.parabolic_heuristic
                
                # === Alternância Shrink / Reshape ===
                # Se a iteração for par, usa :shrink. Se for ímpar, usa :reshape.
                current_mode = (iter % 2 == 0) ? :shrink : :reshape
                
                params_unif.debugverbose && println("📉 [DEBUG-UNIF] Redução Anisotrópica - Modo: $current_mode (Iter: $iter)")

                delta = parabolic_heuristic_step!(
                    delta, F, F_trial, slope_val, params_unif, snorm, norm(s_sol)^2, :decrease;
                    anisotropic=params_unif.anisotropic_trust_region, 
                    s_vec=s_sol, 
                    grad=grad,
                    mode=current_mode
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
        # # Gradiente Lagrangeano Projetado
        # Lgrad = copy(grad)
        # if m > 0 && !iszero(lambda)
        #     Lgrad .+= jac' * lambda
        # end

        # proj_dir_k = clamp.(x .- Lgrad, prob.xl, prob.xu)
        # gpnorm = norm(proj_dir_k .- x, Inf)
        # -------------------------------------------------
        # 4. Cálculo de Métricas de Parada
        # -------------------------------------------------
        proj_step_k = compute_gradient_projection(
            x, grad, jac, lambda, prob.xl, prob.xu;
            norm_type = params_unif.norm_gpnorm,
            strategy = params_unif.gpnorm_strategy,
            max_iter = params_unif.gpnorm_max_iter,
            tol = params_unif.gpnorm_tol,
            debug = params_unif.gpnorm_debug
        )
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
        is_feasible = vio_curr <= 1e-3
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