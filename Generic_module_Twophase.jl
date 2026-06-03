module Generic_module_Twophase

export create_optimized_model, restoration_phase, optimization_phase, two_phase_optimization

using JuMP, Gurobi, Printf, LinearAlgebra
using ..SharedTypes
# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
"""
Força a região de confiança a manter uma proporção mínima saudável.
Se uma dimensão for menor que `factor` vezes a maior dimensão, ela é inflada.
Isso evita o efeito "agulha" que trava o algoritmo em vales diagonais.
"""
function enforce_aspect_ratio!(δ::Vector{Float64}, min_ratio::Float64=1e-3) # rescale scale
    max_delta = maximum(δ)
    threshold = max_delta * min_ratio

    changed = false
    for i in 1:length(δ)
        if δ[i] < threshold
            δ[i] = threshold # Re-infla a dimensão colapsada
            changed = true
        end
    end
    return changed
end
"""
Log message to console and/or file
"""
function log_message!(msg, io; verbose=true)
    if verbose
        println(msg)
    end
    if io !== nothing
        println(io, msg)
        flush(io)  # Ensure immediate write to disk
    end
end

"""
Compute merit function value
"""
@inline function compute_merit_functions(λ::Vector{Float64}, θ::Float64, h::Vector{Float64}, F::Float64)
    return θ * (F + dot(h, λ)) + (1 - θ) * norm(h)
end

# ==============================================================================
# JUMP MODEL CREATION
# ==============================================================================

"""
Create JuMP model for restoration or optimization phase
"""
function create_optimized_model(solver_choice::Symbol, phase::Symbol, n::Int, m::Int; env::Union{Gurobi.Env, Nothing}=nothing)
    # Create base model
    model = if solver_choice == :gurobi
        env = (env === nothing) ? Gurobi.Env() : env
        Model(optimizer_with_attributes(() -> Gurobi.Optimizer(env), "OutputFlag" => 0, "TimeLimit" => 1))
    elseif solver_choice == :highs
        Model(optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "threads" => Threads.nthreads(), "primal_feasibility_tolerance" => 1e-8, "dual_feasibility_tolerance" => 1e-8))
    else
        error("Unsupported solver: $solver_choice")
    end

    model.obj_dict = Dict{Symbol, Any}()
    model.obj_dict[:phase] = phase

    if phase == :restauracao
        # Algorithm 4.1: Restoration subproblem
        # min e'w + eps * e'|d|  
        # s.t.  A(z^ℓ)d + E(z^ℓ)w = -h(z^ℓ), |d| ≤ w_aux, z ∈ Ω, w ≥ 0

        @variable(model, d[1:n])              # Step d = z - z^ℓ
        @variable(model, w[1:m] >= 0)         # Constraint slacks (size m)
        @variable(model, w_aux[1:n] >= 0)     # Auxiliary for |d| (size n)

        # Constraints (will be updated later)
        @constraint(model, con[i = 1:m], 0 == 0)

        # |d[j]| ≤ w_aux[j]  =>  d[j] ≤ w_aux[j] and -d[j] ≤ w_aux[j]
        @constraint(model, d_upper[j = 1:n], d[j] <= w_aux[j])
        @constraint(model, d_lower[j = 1:n], -d[j] <= w_aux[j])

        # Objective: minimize w primarily, then |d|
        eps = 1e-6
        @objective(model, Min, sum(w[i] for i in 1:m) + eps * sum(w_aux[j] for j in 1:n))

        # Store references
        model.obj_dict[:d] = d
        model.obj_dict[:w] = w
        model.obj_dict[:w_aux] = w_aux
        model.obj_dict[:con] = con

    elseif phase == :otimizacao
        # Algorithm 4.2: Optimization subproblem
        # min ∇L(y^k,λ')'(x-y^k)  s.t.  J_h(y^k)(x-y^k) = 0

        @variable(model, s[1:n])

        # Placeholder constraints
        lp_con = @constraint(model, con[i = 1:m], 0.0 * sum(s[j] for j in 1:n) == 0.0)

        # Placeholder objective
        @objective(model, Min, sum(0.0 * s[i] for i in 1:n))

        model.obj_dict[:s] = s
        model.obj_dict[:con] = lp_con
    else
        error("Unsupported phase: $phase. Use :restauracao or :otimizacao")
    end

    return model
end

# ==============================================================================
# ALGORITHM 4.1: RESTORATION PHASE
# ==============================================================================

"""
Solve restoration subproblem
min e'w s.t. A(z^ℓ)d + E(z^ℓ)w = -h(z^ℓ), |d_i| ≤ δ, d ∈ [xl-z, xu-z]
"""
function solve_restoration_subproblem!(model::JuMP.Model, problem::OptimizationProblem, z::Vector{Float64}, δ::Union{Float64, Vector{Float64}})
    n = length(z)

    # Evaluate Jacobian and constraints at z
    A = problem.∇h(z)  # m × n
    h_z = problem.h(z)  # m × 1
    m = length(h_z)

    # Retrieve variables from model dictionary
    d = model.obj_dict[:d]
    w = model.obj_dict[:w]
    con = model.obj_dict[:con]

    # Update bounds: max(xl-z, -δ) ≤ d ≤ min(xu-z, δ)
    for i in 1:n
        # Seleciona o delta correto (escalar ou componente i do vetor)
        d_val = (δ isa Vector) ? δ[i] : δ

        lb = max(problem.xl[i] - z[i], -d_val)
        ub = min(problem.xu[i] - z[i], d_val)
        set_lower_bound(d[i], lb)
        set_upper_bound(d[i], ub)
    end
    # Determine E(z^ℓ): A(z^ℓ)d + E(z^ℓ)w = -h(z^ℓ)
    # To keep w ≥ 0, we need: E(z^ℓ) = -sign(h(z^ℓ))
    E = zeros(m)
    for i in 1:m
        if h_z[i] > 0
            E[i] = -1.0
        elseif h_z[i] < 0
            E[i] = 1.0
        else
            E[i] = 0.0
        end
    end

    # Update constraints: A(z^ℓ)d + E(z^ℓ)w = -h(z^ℓ)
    for i in 1:m
        # Zero out existing coefficients
        for j in 1:n
            set_normalized_coefficient(con[i], d[j], 0.0)
        end
        set_normalized_coefficient(con[i], w[i], 0.0)

        # Set coefficients for d[j]
        for j in 1:n
            set_normalized_coefficient(con[i], d[j], A[i, j])
        end

        # Set coefficient for w[i]
        set_normalized_coefficient(con[i], w[i], E[i])

        # RHS = -h(z)
        set_normalized_rhs(con[i], -h_z[i])
    end

    # Solve
    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        d_sol = value.(d)
        return d_sol, true
    else
        return zeros(n), false
    end
end

"""
Algorithm 4.1: Restoration Phase
"""

function restoration_phase(
    problem::OptimizationProblem, 
    z::Vector{Float64}, 
    δ::Union{Float64, Vector{Float64}}, 
    model::JuMP.Model; 
    params, 
    verbose::Bool=params.verbose_in, 
    buffers::OptimizedBuffers, 
    full_log::Vector{StepLog_TwoPhase}=Vector{StepLog_TwoPhase}(), 
    iterglobal::Int
)
    # =========================================================
    # PASSO 1: Computar ctarget e ϵ_c baseados no x^k original
    # =========================================================
    h_init = problem.h(z)
    norm_h_init = norm(h_init)
    
    # c(x^k) = 1/2 * ||h(x^k)||^2
    c_init = 0.5 * norm_h_init^2 
    
    # ctarget = 1/2 * r^2 * ||h(x^k)||^2
    r = params.r_resto 
    c_target = 0.5 * (r^2) * (norm_h_init^2) 
    
    # ϵ_c = rfeas * ||h(x^k)||
    eps_c = params.rfeas * norm_h_init 
    
    # =========================================================
    # PASSO 2: Inicializar ℓ <- 0 e z^0 
    # =========================================================
    ℓ = 0
    c_prev = c_init # Usado para a heurística de trust region
    # Estamos tomando z^0=x^k, então as condições a seguir são satisfeitas sem precisar de testes:
    # 1. ||h(z^0)|| <= ||h(x^k)||
    # 2. ||z^0-x^k|| <= βc * ||h(x^k)||
    verbose && println("🔄 Starting restoration phase...")
    params.debugverbose && println("DEBUG: Initial state - δ=$δ, norm_h_init=$norm_h_init, c_target=$c_target, eps_c=$eps_c")

    # Use temporary buffer to avoid modifying z directly
    buffers.z_temp .= z
    buffers.z_old .= z

    # LOOP EXTERNO (Iterações ℓ)
    while ℓ < params.max_iter_resto
        
        # Avalia c(z^ℓ)
        h_z = problem.h(buffers.z_temp)
        c_z = 0.5 * norm(h_z)^2

        # Computa o gradiente projetado para a condição de parada: || P_Ω(z - ∇c(z)) - z ||
        A_z = problem.∇h(buffers.z_temp)
        grad_c = A_z' * h_z
        z_minus_grad = buffers.z_temp .- grad_c
        projected_step = clamp.(z_minus_grad, problem.xl, problem.xu)
        proj_grad_norm = norm(projected_step .- buffers.z_temp)

        params.debugverbose && println("DEBUG: Iter $ℓ - c(z)=$c_z, proj_grad_norm=$proj_grad_norm")

        # =========================================================
        # PASSO 3: Critério de parada
        # =========================================================
        if c_z <= c_target || proj_grad_norm <= eps_c
            verbose && println("✅ Restoration stopped in $ℓ iterations.")
            if c_z <= c_target
                params.debugverbose && println("DEBUG: Target feasibility reduction achieved (c_z <= c_target).")
            else
                params.debugverbose && println("DEBUG: Stationarity reached (proj_grad_norm <= eps_c).")
            end
            
            z .= buffers.z_temp
            return buffers.z_temp, δ, true
        end

        # =========================================================
        # PASSO 4: Inicializar j <- 0 e escolher δ
        # =========================================================
        j = 0
        success_step = false

        # LOOP INTERNO (Iterações j - Busca pelo passo aceito)
        while j < params.max_iter_resto
            
            # =========================================================
            # PASSO 5: Resolver o subproblema para computar z^{ℓ,j}
            # =========================================================
            buffers.s_val, success_solver = solve_restoration_subproblem!(model, problem, buffers.z_temp, δ)

            if success_solver
                buffers.z_old .= buffers.z_temp
                
                # Test new point z^{ℓ,j}
                buffers.z_temp .= buffers.z_temp .+ buffers.s_val
                buffers.z_temp .= clamp.(buffers.z_temp, problem.xl, problem.xu)

                # Evaluate new c(z)
                h_new = problem.h(buffers.z_temp)
                c_new = 0.5 * norm(h_new)^2

                # =========================================================
                # PASSO 6: Condição de Descida sobre c(z)
                # c(z^{ℓ,j}) ≤ c(z^ℓ) - α_R * ||z^{ℓ,j} - z^ℓ||^2
                # =========================================================
                step_norm_sq = dot(buffers.z_temp .- buffers.z_old, buffers.z_temp .- buffers.z_old)
                reduction_threshold = params.αR * step_norm_sq

                is_accepted = c_new <= c_z - reduction_threshold

                # Gravação de Log
                log_entry = StepLog_TwoPhase(
                    iterglobal, copy(buffers.z_old), copy(buffers.z_temp), 
                    is_accepted ? :accepted : :rejected, :restoration, δ, 0.0, 0.0
                )
                push!(full_log, log_entry)

                if is_accepted
                    success_step = true
                    c_prev = c_z # Atualiza o estado anterior para a próxima iteração
                    break        # Passo aceito: sai do laço interno j
                else
                    # REJEIÇÃO: Restaura z e encolhe δ estritamente (Teoria Passo 6)
                    buffers.z_temp .= buffers.z_old
                    δ = max.(params.τ1 .* δ, params.δmin)
                end
            else
                # Se o solver falhar (ex: instabilidade numérica), trata como rejeição
                buffers.z_temp .= buffers.z_old
                δ = max.(params.τ1 .* δ, params.δmin)
            end

            j += 1
        end # Fim do loop j

        # Checa se o loop j estourou o limite sem encontrar passo aceitável
        if !success_step || j >= params.max_iter_resto
            verbose && println("⚠️ Restoration failure at iteration $ℓ (Too many inner iterations)")
            z .= buffers.z_temp
            return z, δ, false
        end

        # =========================================================
        # PASSO 7: Atualização Adaptativa do δ 
        # (Feito APÓS a aceitação do passo para preparar o próximo ℓ)
        # =========================================================
        if params.use_ratio_update && ℓ > 0 && c_prev > params.ratio_safeguard_tol && @isdefined(c_new)
            ratio = clamp(c_new / c_prev, params.ratio_min_factor, params.ratio_max_factor)
            δ = clamp.(ratio .* δ, params.δmin, params.δmax)
        end

        # =========================================================
        # PASSO 7 (Teoria): z^{ℓ+1} = z^{ℓ,j}, ℓ <- ℓ+1, voltar ao Passo 3
        # =========================================================
        # O z^{ℓ+1} já está em buffers.z_temp devido ao sucesso no loop interno
        ℓ += 1
        
    end # Fim do loop ℓ

    # Se atingiu o limite máximo de iterações externas ℓ
    z .= buffers.z_temp
    return z, δ, ℓ < params.max_iter_resto
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

"""
Solve optimization subproblem
min ∇f'·s  s.t. ∇h'·s = -h, bounds
"""
function solve_optimization_subproblem!(model::JuMP.Model, n::Int, δ::Union{Float64, Vector{Float64}}, problem::OptimizationProblem, use_quadratic::Bool, buffers::OptimizedBuffers)
    m = length(buffers.h_old)
    grad_L = buffers.∇f + transpose(buffers.∇h) * buffers.λ
    if use_quadratic
        # Use RipQP for quadratic model
        H = problem.∇²L(buffers.x_temp, zeros(m))
        Hqp = 0.5 * H + spdiagm(0 => -0.1 * ones(n))

        # Calculate bounds: max(xl-x, -δ) ≤ s ≤ min(xu-x, δ)
        for i in 1:n
            d_val = (δ isa Vector) ? δ[i] : δ

            lx = problem.xl[i] - buffers.x_temp[i]
            ux = problem.xu[i] - buffers.x_temp[i]
            buffers.lvar[i] = max(lx, -d_val)
            buffers.uvar[i] = min(ux, d_val)
        end

        qm = QuadraticModel(buffers.∇f, Hqp; A=buffers.∇h, lcon=(-buffers.h_old), ucon=(-buffers.h_old), lvar=buffers.lvar, uvar=buffers.uvar, c0=0.0, name="OptSubproblem")

        stats = ripqp(qm)

        if stats.status == :first_order
            copy!(buffers.s_val, stats.solution)
            buffers.λ = stats.multipliers[1:m]
            return true, -dot(buffers.∇f, stats.solution)
        end
    end

    # JuMP-based solution

    s = model[:s]
    con = model[:con]

    # Update bounds: max(xl-x, -δ) ≤ s ≤ min(xu-x, δ)
    @inbounds for i in 1:n
        d_val = (δ isa Vector) ? δ[i] : δ

        lx = problem.xl[i] - buffers.x_temp[i]
        ux = problem.xu[i] - buffers.x_temp[i]

        lower = max(lx, -d_val)
        upper = min(ux, d_val)

        set_lower_bound(s[i], lower)
        set_upper_bound(s[i], upper)
    end

    # Update objective
    if use_quadratic
        H = problem.∇²L(buffers.x_temp, zeros(m))
        @objective(model, Min, 0.5 * dot(s, H * s) + dot(buffers.∇f, s))
    else
        # @objective(model, Min, sum(buffers.∇f[i] * s[i] for i in 1:n))
        # SLP: min ∇L' * s
        @objective(model, Min, dot(grad_L, s))
    end

    # Update constraints: ∇h'·s = -h
    @inbounds for i in 1:m
        # set_normalized_rhs(con[i], -buffers.h_old[i])
        set_normalized_rhs(con[i], 0.0)

        for j in 1:n
            set_normalized_coefficient(con[i], s[j], buffers.∇h[i, j])
        end
    end

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        # Extract solution
        @inbounds for i in 1:n
            buffers.s_val[i] = value(s[i])
        end

        obj_val = objective_value(model)

        # Extract dual variables
        for i in 1:m
            buffers.λ[i] = dual(con[i])
        end

        return true, obj_val
    end

    return false, 0.0
end



"""
Algorithm 4.2: Optimization Phase (Non-Monotone Support)
"""
function optimization_phase(
    problem::OptimizationProblem,
    x_k::Vector{Float64},      # Passo 0: O x antes da restauração (x^k)
    λ_k::Vector{Float64},      # Passo 0: O lambda anterior (λ^k)
    norm_h_xk::Float64,        # Passo 0: ||h(x^k)||
    y::Vector{Float64},        # Passo 0: Ponto após restauração (y^k)
    δ::Union{Float64, Vector{Float64}},
    model_opt::JuMP.Model,
    params;
    θ_k::Float64=0.5,          # Passo 0: Parâmetro de penalidade da iteração k
    use_quadratic::Bool=false,
    verbose::Bool=params.verbose_in,
    buffers::OptimizedBuffers,
    full_log::Vector{StepLog_TwoPhase}=Vector{StepLog_TwoPhase}(),
    iterglobal::Int,
    # === Histórico para Busca Não-Monótona ===
    lagrangian_history::Vector{Float64}=Float64[],
    merit_history::Vector{Float64}=Float64[]
)
    # =========================================================
    # PASSO 1: Inicializar j <- 0, escolher δ_j e λ'
    # =========================================================
    j = 0
    n = length(y)
    m = length(problem.h(y))
    θ = θ_k
    consecutive_rejects = 0

    verbose && println("🎯 Starting optimization phase...")

    # Avaliações no ponto y^k 
    buffers.x_temp .= y
    F_y = problem.f(y)
    h_y = problem.h(y)
    norm_h_y = norm(h_y)
    
    # Avaliações no ponto x^k
    F_xk = problem.f(x_k)
    h_xk = problem.h(x_k)

    params.debugverbose && println("🔍 [DEBUG-OPT] Initial state: F(y^k)=$F_y, ||h(y^k)||=$norm_h_y, δ=$δ")
    params.debugverbose && println("🔍 [DEBUG-OPT] History sizes - Lagrangian: $(length(lagrangian_history)), Merit: $(length(merit_history))")

    # Loop Principal de Otimização (Iterações j)
    while j < params.max_iter_opt
        params.debugverbose && println("\n🔍 [DEBUG-OPT] === Iteration j=$j started ===")

        # Preparar gradientes em y^k para o subproblema
        buffers.∇f = problem.∇f(buffers.x_temp)
        buffers.∇h = problem.∇h(buffers.x_temp)

        # =========================================================
        # PASSO 2: Encontrar z^{ℓ,j} resolvendo o subproblema
        # PASSO 3: Escolher λ 
        # =========================================================
        success, f_model = solve_optimization_subproblem!(model_opt, n, δ, problem, use_quadratic, buffers)
        buffers.λ .= 0.0
        if !success
            δ = max.(params.τ2 .* δ, params.δmin)
            j += 1
            params.debugverbose && println("🚫 [DEBUG-OPT] Subproblem failure! Could not find a valid step.")
            verbose && println("🔄 Subproblem failure, reducing δ to $δ (j=$j)")
            continue
        end

        # z^j = y^k + s
        buffers.x_old .= buffers.x_temp
        buffers.x_temp .+= buffers.s_val

        # Avaliar funções no novo ponto z^j
        F_zj = problem.f(buffers.x_temp)
        h_zj = problem.h(buffers.x_temp)
        norm_h_zj = norm(h_zj)

        step_norm_sq = dot(buffers.s_val, buffers.s_val)
        
        params.debugverbose && println("🔍 [DEBUG-OPT] Subproblem solved. ||s|| = $(sqrt(step_norm_sq)), f_model predicted red: $f_model")
        params.debugverbose && println("🔍 [DEBUG-OPT] New point evals: F(z^j)=$F_zj, ||h(z^j)||=$norm_h_zj, λ=$(buffers.λ)")

        # =========================================================
        # PASSO 4: Teste de Descida da Lagrangiana
        # L(z^j, λ) ≤ L(y^k, λ) - α_L ||z^j - y^k||^2
        # =========================================================
        L_y_lambda = F_y + dot(buffers.λ, h_y) 
        L_zj_lambda = F_zj + dot(buffers.λ, h_zj)

        # Heurística Não-Monótona para a Lagrangiana
        ref_lagrangian = isempty(lagrangian_history) ? L_y_lambda : max(L_y_lambda, maximum(lagrangian_history))
        lagrangian_threshold = ref_lagrangian - params.αL * step_norm_sq

        params.debugverbose && println("🔍 [DEBUG-OPT] --- Lagrangian Check ---")
        params.debugverbose && println("🔍 [DEBUG-OPT] L(y^k, λ): $L_y_lambda")
        params.debugverbose && println("🔍 [DEBUG-OPT] L(z^j, λ): $L_zj_lambda")
        params.debugverbose && println("🔍 [DEBUG-OPT] Ref Lagrangian: $ref_lagrangian")
        params.debugverbose && println("🔍 [DEBUG-OPT] Threshold to beat: $lagrangian_threshold (Drop req: $(params.αL * step_norm_sq))")

        if L_zj_lambda > lagrangian_threshold
            # -- FALHA NO PASSO 4 --
            gap = L_zj_lambda - lagrangian_threshold
            params.debugverbose && println("🚫 [DEBUG-OPT] REJECTED by Lagrangian! L(z^j) is $gap ABOVE the threshold.")

            # LOG DA REJEIÇÃO PELA LAGRANGIANA RECUPERADO AQUI
            push!(full_log, StepLog_TwoPhase(
                iterglobal, copy(buffers.x_old), copy(buffers.x_temp), 
                :rejected, :optimization, δ, L_zj_lambda, compute_merit_functions(buffers.λ, θ_k, h_zj, F_zj)
            ))

            consecutive_rejects += 1
            buffers.x_temp .= buffers.x_old # Reverte o passo
            
            if params.use_slp_stopping && ((δ isa Vector) ? maximum(δ) : δ) <= params.δmin
                verbose && println("⚠️ δ too small, aborting optimization")
                return y, buffers.λ, θ_k, δ, false
            end

            # Lógica de encolhimento de δ após falha na Lagrangiana
            if !params.backtracking_quadratic
                s_ref = (δ isa Vector) ? abs.(buffers.s_val) : norm(buffers.s_val, Inf)
                δ_old = copy(δ)
                δ = clamp.(max.(δ .* params.τ1, s_ref .* params.τ2), params.δmin, params.δmax)
                params.debugverbose && println("🔍 [DEBUG-OPT] Standard Shrink: δ changed from $δ_old to $δ")
            else
                verbose && println("🔄 Lagrangian test failure, applying backtracking")
                s_inf = norm(buffers.s_val, Inf)
                backtracking_mode = isodd(consecutive_rejects) ? :reshape : :shrink
                
                params.debugverbose && println("🔍 [DEBUG-OPT] Quad Backtracking (mode: $backtracking_mode). Consecutive rejects: $consecutive_rejects")
                δ = quadratic_backtracking_step!(
                    δ, L_y_lambda, L_zj_lambda, f_model, params, 
                    s_inf, step_norm_sq, :decrease; 
                    anisotropic=params.anisotropic_trust_region, 
                    s_vec=buffers.s_val, grad=buffers.∇f, mode=backtracking_mode
                )
                if params.anisotropic_trust_region && consecutive_rejects >= 2
                    was_corrected = enforce_aspect_ratio!(δ, 0.15)
                    was_corrected && params.verbose_out && verbose && println("🎈 [Rescue] Re-inflando dimensões.")
                end
            end
            
            # Vai para o Passo 2 (Iteração j+1)
            j += 1
            continue
        end

        params.debugverbose && println("✅ [DEBUG-OPT] ACCEPTED by Lagrangian check. (Beat threshold by $(lagrangian_threshold - L_zj_lambda))")
        # Se passou na Lagrangiana, zera o contador de rejeições para o Backtracking
        consecutive_rejects = 0 

        # =========================================================
        # PASSO 5: Atualização do parâmetro de penalidade θ
        # =========================================================
        L_x_lambda_k = F_xk + dot(λ_k, h_xk)
        
        Phi_y_thetak = compute_merit_functions(buffers.λ, θ_k, h_y, F_y)
        Phi_xk_thetak = compute_merit_functions(λ_k, θ_k, h_xk, F_xk)

        # Heurística Não-Monótona para o Teste do Theta
        ref_merit_thetak = isempty(merit_history) ? Phi_xk_thetak : max(Phi_xk_thetak, maximum(merit_history))

        params.debugverbose && println("🔍 [DEBUG-OPT] --- Theta Update ---")
        params.debugverbose && println("🔍 [DEBUG-OPT] Φ(y^k, θ_k): $Phi_y_thetak")
        params.debugverbose && println("🔍 [DEBUG-OPT] Ref Φ(x^k, θ_k): $ref_merit_thetak, Drop req: $(params.αΦ * norm_h_xk)")

        if Phi_y_thetak <= ref_merit_thetak - params.αΦ * norm_h_xk
            θ = θ_k
            params.debugverbose && println("🔍 [DEBUG-OPT] Condition met. Keeping θ = $θ")
        else
            numerador = (1.0 - params.αΦ) * norm_h_xk - norm_h_y
            denominador = L_y_lambda - L_x_lambda_k + norm_h_xk - norm_h_y
            
            params.debugverbose && println("🔍 [DEBUG-OPT] Condition failed. Updating θ...")
            params.debugverbose && println("🔍 [DEBUG-OPT] Numerator: $numerador, Denominator: $denominador")
            
            if abs(denominador) > 1e-12
                θ = numerador / denominador
            else
                params.debugverbose && println("⚠️ [DEBUG-OPT] Denominator too small, falling back to θ_k.")
                θ = θ_k # Fallback numérico
            end
        end
        # params.debugverbose && println("   [Theta Update] θ updated to $θ (j=$j)")

        # =========================================================
        # PASSO 6: Teste da Função de Mérito
        # Φ(z^j, λ, θ) ≤ Φ(x^k, λ^k, θ) - α_Φ ||h(x^k)||
        # =========================================================
        Phi_zj_theta = compute_merit_functions(buffers.λ, θ, h_zj, F_zj)
        Phi_xk_theta = compute_merit_functions(λ_k, θ, h_xk, F_xk)

        # Heurística Não-Monótona para a Mérito Final
        ref_merit_theta = isempty(merit_history) ? Phi_xk_theta : max(Phi_xk_theta, maximum(merit_history))
        merit_threshold = ref_merit_theta - params.αΦ * norm_h_xk

        params.debugverbose && println("🔍 [DEBUG-OPT] --- Merit Function Check ---")
        params.debugverbose && println("🔍 [DEBUG-OPT] Φ(z^j, θ): $Phi_zj_theta")
        params.debugverbose && println("🔍 [DEBUG-OPT] Ref Φ(x^k, θ): $ref_merit_theta")
        params.debugverbose && println("🔍 [DEBUG-OPT] Threshold to beat: $merit_threshold")

        # Gravação de Log da tentativa para o Teste de Mérito
        push!(full_log, StepLog_TwoPhase(
            iterglobal, copy(buffers.x_old), copy(buffers.x_temp), 
            Phi_zj_theta > merit_threshold ? :rejected : :accepted, 
            :optimization, δ, L_zj_lambda, Phi_zj_theta
        ))

        if Phi_zj_theta > merit_threshold
            # -- FALHA NO PASSO 6 --
            gap_merit = Phi_zj_theta - merit_threshold
            params.debugverbose && println("🚫 [DEBUG-OPT] REJECTED by Merit Function! Φ(z^j) is $gap_merit ABOVE threshold.")
            
            buffers.x_temp .= buffers.x_old # Reverte o passo

            if params.use_slp_stopping && ((δ isa Vector) ? maximum(δ) : δ) <= params.δmin + 1e-9
                verbose && println("⚠️ δ too small, aborting optimization")
                return y, buffers.λ, θ, δ, false
            end

            δ_old = copy(δ)
            δ = clamp.(δ .* params.τ2, params.δmin, params.δmax)
            j += 1
            
            params.debugverbose && println("🔍 [DEBUG-OPT] Merit failure. Shrinking δ from $δ_old to $δ")
            verbose && println("🔄 Merit test failure, reducing δ to $δ (j=$j)")
            continue
        end

        # =========================================================
        # PASSO 7: Passo Aceito! Atualiza estado final e δ para o próx passo
        # =========================================================
        params.debugverbose && println("✅ [DEBUG-OPT] ACCEPTED by Merit function! (Beat threshold by $(merit_threshold - Phi_zj_theta))")
        verbose && println("✅ Step accepted in optimization (j=$j)")
        
        # O passo foi aceito, calculamos o gradiente para o log ou backtracking futuro
        ∇f_new_accepted = problem.∇f(buffers.x_temp)
        buffers.∇f_old .= ∇f_new_accepted
        
        # Atualização (Aumento) da região de confiança
        if !params.backtracking_quadratic
            δ_old = copy(δ)
            δ = min.(δ ./ params.τ3, params.δmax)
            params.debugverbose && println("🔍 [DEBUG-OPT] Trust Region expanded from $δ_old to $δ")
        else
            params.debugverbose && println("🔍 [DEBUG-OPT] Applying Quadratic Backtracking for Trust Region expansion.")
            s_inf = norm(buffers.s_val, Inf)
            δ = quadratic_backtracking_step!(
                δ, L_y_lambda, L_zj_lambda, f_model, params, 
                s_inf, step_norm_sq, :increase; 
                anisotropic=params.anisotropic_trust_region, 
                s_vec=buffers.s_val, grad=∇f_new_accepted, 
                grad_old=buffers.∇f_old, mode=:shrink
            )
        end

        break # Sai do loop j
    end

    # Retorna z^j (atualmente em x_temp) como x^{k+1}
    y .= buffers.x_temp
    success = j < params.max_iter_opt

    params.debugverbose && println("🏁 [DEBUG-OPT] Optimization phase finished. Success: $success. Total inner iters: $j")

    return y, buffers.λ, θ, δ, success
end

"""
Main two-phase optimization algorithm (Algorithm 4.3)
"""
function two_phase_optimization(
    problem::OptimizationProblem,
    x0::Vector{Float64},
    params::TwoPhaseParams;
    solver_choice::Symbol=:gurobi,
    use_quadratic::Bool=false,
    max_outer_iter::Int=params.max_outer_iter,
    tolerance::Float64=params.tol,
    history::Bool=true,
)

    verbose=params.verbose_out
    # println(params)
    logio = nothing

    try
        verbose && println("🚀 Starting two-phase optimization")

        if params.debugverbose
            println("DEBUG: === INITIALIZATION ===")
            println("DEBUG: Basic parameters:")
            println("  max_outer_iter = $max_outer_iter")
            println("  tolerance = $tolerance")
            println("  r_resto = $(params.r_resto)")
            println("  rfeas = $(params.rfeas)")
        end

        n = length(x0)
        m = length(problem.h(x0))

        # Initialize variables
        x = clamp.(copy(x0), problem.xl, problem.xu)
        λ = zeros(m)
        θ = params.θ0
        δr = params.anisotropic_trust_region ? fill(params.δ0_resto, n) : params.δ0_resto
        δo = params.anisotropic_trust_region ? fill(params.δ0_opt, n) : params.δ0_opt
        iter = 0

        # Contadores de convergência
        countG = 0
        countF = 0
        countS = 0

        # Create buffers
        buffers = OptimizedBuffers(n, m)

        # History tracking
        x_hist = history ? Vector{Vector{Float64}}(undef, max_outer_iter + 1) : nothing
        if history
            x_hist[1] = copy(x)
        end
        full_log = Vector{StepLog_TwoPhase}()

        # === Inicialização dos Buffers Não-Monótonos ===
        history_size = params.non_monotone_M
        lagrangian_history = Float64[]
        merit_history = Float64[]

        # Evaluate initial state
        f_current = problem.f(x)
        h_current = problem.h(x)
        norm_h_current = norm(h_current)
        buffers.∇f_old = problem.∇f(x)
        
        # Popula o estado inicial no histórico não-monótono
        push!(lagrangian_history, f_current + dot(λ, h_current))
        push!(merit_history, compute_merit_functions(λ, θ, h_current, f_current))

        # ====================================================================
        # Verificação de Otimalidade da Solução Inicial (KKT / AGP)
        # ====================================================================
        jac_h = problem.∇h(x)
        Lgrad = buffers.∇f_old + jac_h' * λ
        gradpbox = clamp.(x .- Lgrad, problem.xl, problem.xu) .- x
        
        gpnorm = norm(gradpbox, params.norm_gpnorm)
        if params.gpnorm_div_nelem
            gpnorm /= n
        end

        if params.debugverbose
            println("DEBUG: Initial state:")
            println("  x0 = $x0")
            println("  f(x0) = $f_current")
            println("  ||h(x0)|| = $norm_h_current")
            println("  gpnorm = $gpnorm")
        end

        if (gpnorm < params.tolG) && (norm_h_current < tolerance)
            log_message!("✅ SOLUÇÃO INICIAL JÁ É ÓTIMA!", logio; verbose=verbose)
            log_message!("   Condições KKT satisfeitas.", logio; verbose=verbose)
            if params.use_slp_stopping
                countG = params.maxcount
                countF = params.maxcount
            end
        end

        # Create models
        const_env = Gurobi.Env()
        model_restaura = create_optimized_model(solver_choice, :restauracao, n, m; env=const_env)
        model_opt = create_optimized_model(solver_choice, :otimizacao, n, m; env=const_env)

        # Table header
        log_message!(repeat("=", 120), logio; verbose=verbose)
        log_message!("                    TWO-PHASE OPTIMIZATION - ALGORITHM 4.3", logio; verbose=verbose)
        log_message!(repeat("=", 120), logio; verbose=verbose)

        if params.use_slp_stopping
            msg = @sprintf("%-4s %-12s %-10s %-8s %-8s %-10s %-10s %-18s", "k", "f(xᵏ)", "||h(xᵏ)||", "δr", "δo", "||Δx||", "||gpnorm||", "(G|F|S)")
        else
            msg = @sprintf("%-4s %-12s %-10s %-8s %-8s %-10s  %-10s", "k", "f(xᵏ)", "||h(xᵏ)||", "δr", "δo", "||gpnorm||", "||Δx||")
        end
        log_message!(msg, logio; verbose=verbose)
        log_message!(repeat("-", 120), logio; verbose=verbose)

        d_r_print = (δr isa Vector) ? maximum(δr) : δr
        d_o_print = (δo isa Vector) ? maximum(δo) : δo

        if params.use_slp_stopping
            msg = @sprintf("%-4d %-12.5e %-10.3e %-8.2e %-8.2e %-10s %-10.3e %-18s", 0, f_current, norm_h_current, d_r_print, d_o_print, "-", gpnorm, "0|0|0")
        else
            msg = @sprintf("%-4d %-12.5e %-10.3e %-8.2e %-8.2e %-10.3e %-10s", 0, f_current, norm_h_current, d_r_print, d_o_print, gpnorm, "-")
        end
        log_message!(msg, logio; verbose=verbose)

        # Main loop (Algorithm 4.3)
        while iter < max_outer_iter
            
            # --- Check Stopping Criteria ---
            if params.use_slp_stopping
                is_feasible = norm_h_current < tolerance
                
                slp_converged_kkt = (countG >= params.maxcount && countF >= params.maxcount && is_feasible)
                
                slp_converged_step = (countS >= params.maxcount && is_feasible) 

                if slp_converged_kkt || slp_converged_step
                    log_message!("-" ^ 120, logio; verbose=verbose)
                    if slp_converged_kkt
                        log_message!("✅ CONVERGÊNCIA: Condições KKT (gradiente, redução E restrição pequenos)", logio; verbose=verbose)
                    else
                        log_message!("✅ CONVERGÊNCIA: Passo suficientemente pequeno em ponto viável", logio; verbose=verbose)
                    end
                    break
                elseif (countG >= params.maxcount || countS >= params.maxcount) && !is_feasible
                    log_message!("⚠️ ALERTA: Algoritmo estagnou mas o ponto NÃO é viável (||h(x)|| = $norm_h_current)", logio; verbose=verbose)
                end
            else
                #SLP OFF, paramos se KKT
                if norm_h_current < tolerance && gpnorm < params.tolG
                    log_message!("-" ^ 120, logio; verbose=verbose)
                    log_message!("✅ CONVERGÊNCIA: Solução KKT exata encontrada", logio; verbose=verbose)
                    break
                end
            end

            x_old_outer = copy(x)
            norm_h_old = norm_h_current

            # ================================================================
            # STEP 1: RESTORATION PHASE
            # ================================================================
            if norm_h_current > tolerance
                y, δr, success_resto = restoration_phase(problem, x, δr, model_restaura; params=params, buffers=buffers, full_log=full_log, iterglobal=iter)

                norm_h_y = norm(problem.h(y))
                if norm_h_y > params.r_resto * norm_h_old
                    log_message!("❌ FAILURE: ||h(yᵏ)|| > r||h(xᵏ)|| - Insufficient restoration", logio; verbose=verbose)
                    if history
                        x_hist[iter + 2] = copy(y)
                    end
                    return y, λ, θ, iter, 3, history ? x_hist[1:(iter + 2)] : nothing, full_log
                end
            else
                y = copy(x)
                success_resto = true
            end

            # ================================================================
            # STEP 2: OPTIMIZATION PHASE
            # ================================================================
            x_new, λ_new, θ_new, δo, success_opt = optimization_phase(
                problem, 
                x_old_outer,      # x^k (Ponto antes da restauração)
                λ,                # λ^k (Multiplicadores da iteração anterior)
                norm_h_old,       # ||h(x^k)||
                y,                # y^k (Ponto após a restauração)
                δo, 
                model_opt, 
                params; 
                θ_k = θ,          # θ_k (Passado como kwarg)
                verbose = params.verbose_in, 
                buffers = buffers, 
                full_log = full_log, 
                iterglobal = iter,
                lagrangian_history = lagrangian_history,
                merit_history = merit_history
            )

            # if !success_opt
            #     δo = max.(params.τ1 .* δo, params.δmin)
            # end

            # ================================================================
            # STEP 3: UPDATE AND LOG
            # ================================================================
            x .= x_new
            λ .= λ_new
            θ = θ_new

            f_new = problem.f(x)
            h_new = problem.h(x)
            norm_h_new = norm(h_new)

            Δx = norm(x - x_old_outer, Inf)
            ΔF = abs(f_new - f_current)

            # --- Recalcular GP Norm (AGP) ---
            grad_f_new = problem.∇f(x)
            jac_h_new = problem.∇h(x)
            Lgrad = grad_f_new + jac_h_new' * λ
            gradpbox = clamp.(x .- Lgrad, problem.xl, problem.xu) .- x
            gpnorm = norm(gradpbox, params.norm_gpnorm)
            params.debugverbose && println("DEBUG: GPnorm recalculated: $gpnorm")
            if params.gpnorm_div_nelem
                gpnorm /= n
            end
            params.debugverbose && println("DEBUG: GPnorm after division (if applicable): $gpnorm")

            iter += 1

            if history
                x_hist[iter + 1] = copy(x)
            end

            # === Atualizando a Memória Não-Monótona ===
            lagrangian_new = f_new + dot(λ, h_new)
            merit_new = compute_merit_functions(λ, θ, h_new, f_new)
            
            push!(lagrangian_history, lagrangian_new)
            push!(merit_history, merit_new)

            if length(lagrangian_history) > history_size
                popfirst!(lagrangian_history)
            end
            if length(merit_history) > history_size
                popfirst!(merit_history)
            end

            # --- Atualizar Contadores SLP ---
            if params.use_slp_stopping
                if gpnorm <= params.tolG
                    countG += 1
                else
                    countG = 0
                end

                if ΔF <= params.tolF
                    countF += 1
                else
                    countF = 0
                end

                if Δx <= params.tolS
                    countS += 1 # mudar para 1 se quiser contar passo pequeno, ou manter 0 para só considerar passo pequeno se for consecutivo
                else
                    countS = 0
                end
            end

            # --- Construir status_symbol ---
            status_symbol = ""
            if params.use_slp_stopping
                if countG >= params.maxcount && countF >= params.maxcount
                    status_symbol = " ✅KKT"
                elseif countS >= params.maxcount
                    status_symbol = " ✅STEP"
                end
            end

            if status_symbol == ""
                if !success_resto
                    status_symbol = " ❌R"
                elseif !success_opt
                    status_symbol = " ⚠️O"
                elseif (δo isa Vector ? maximum(δo) : δo) <= params.δmin + 1e-12
                    status_symbol = " ⚠️δo"
                end
            end

            d_r_print = (δr isa Vector) ? maximum(δr) : δr
            d_o_print = (δo isa Vector) ? maximum(δo) : δo 

            # Log formatado espelhando a topologia
            if params.use_slp_stopping
                msg = @sprintf("%-4d %-12.5e %-10.3e %-8.2e %-8.2e %-10.3e %-10.3e %d|%d|%d%s", iter, f_new, norm_h_new, d_r_print, d_o_print, Δx, gpnorm, countG, countF, countS, status_symbol)
            else
                msg = @sprintf("%-4d %-12.5e %-10.3e %-8.2e %-8.2e %-10.3e %-10.3e %s", iter, f_new, norm_h_new, d_r_print, d_o_print, gpnorm, Δx, status_symbol)
            end
            log_message!(msg, logio; verbose=verbose)

            f_current = f_new
            h_current = h_new
            norm_h_current = norm_h_new
        end

        # ================================================================
        # FINALIZATION
        # ================================================================
        log_message!(repeat("=", 120), logio; verbose=verbose)

        opstop = if params.use_slp_stopping
            if countG >= params.maxcount && countF >= params.maxcount && norm_h_current < tolerance
                0
            elseif countS >= params.maxcount && norm_h_current < tolerance
                1
            elseif iter >= max_outer_iter
                2
            else
                3 # Parou por estagnação inviável ou outro erro
            end
        else
            if norm_h_current < tolerance && gpnorm < params.tolG
                0
            elseif iter >= max_outer_iter
                2
            else
                1
            end
        end

        termination_msg = ""
        if params.use_slp_stopping
            termination_msg = if opstop == 0
                "✅ CONVERGÊNCIA  - KKT CONDITIONS: Gradiente E redução pequenos por $(params.maxcount) iterações consecutivas"
            elseif opstop == 1
                "✅ CONVERGÊNCIA  - SMALL STEP: Passo pequeno por $(params.maxcount) iterações consecutivas"
            elseif opstop == 2
                "⏰ MÁXIMO DE ITERAÇÕES: Atingiu limite de $max_outer_iter iterações"
            else
                "❌ TERMINAÇÃO POR FALHA: Problemas na restauração ou otimização"
            end
        else
            termination_msg = if opstop == 0
                "✅ CONVERGENCE: Exact KKT solution found"
            elseif opstop == 2
                "⏰ MAXIMUM ITERATIONS: Reached limit of $max_outer_iter iterations"
            else
                "⚠️ TERMINATION: Alternative criterion"
            end
        end

        log_message!(termination_msg, logio; verbose=verbose)
        log_message!("Optimization finished after $iter iterations.", logio; verbose=verbose)
        log_message!("🎉 Final result:", logio; verbose=verbose)
        log_message!(@sprintf("   f(x*) = %.6e", f_current), logio; verbose=verbose)
        log_message!(@sprintf("   ||h(x*)|| = %.2e", norm_h_current), logio; verbose=verbose)
        d_r_print = (δr isa Vector) ? maximum(δr) : δr
        d_o_print = (δo isa Vector) ? maximum(δo) : δo
        log_message!(@sprintf("   δr (max) = %.2e, δo (max) = %.2e", d_r_print, d_o_print), logio; verbose=verbose)

        return x, λ, θ, iter, opstop, history ? x_hist[1:(iter + 1)] : nothing, full_log
              
    finally
        if logio !== nothing
            close(logio)
        end
    end
end

end