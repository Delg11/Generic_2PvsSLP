module Generic_module_Twophase

export create_optimized_model, restoration_phase, optimization_phase, two_phase_optimization


using JuMP, Gurobi, Printf, LinearAlgebra, RipQP,SparseArrays, QuadraticModels
using ..SharedTypes
using ..Generic_module_ComputeB
# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

# Logs messages to the console and/or an IO stream
function log_message!(msg, io; verbose=true)
    if verbose
        println(msg)
    end
    if io !== nothing
        println(io, msg)
        flush(io)  # Ensure immediate write to disk
    end
end

# Computes the merit function value combining objective and feasibility metrics
@inline function compute_merit_functions(λ::Vector{Float64}, θ::Float64, h::Vector{Float64}, F::Float64)
    return θ * (F + dot(h, λ)) + (1 - θ) * norm(h)
end

# ==============================================================================
# JUMP MODEL CREATION
# ==============================================================================

# Creates the JuMP model structure for either the restoration or optimization phase
function create_optimized_model(solver_choice::Symbol, phase::Symbol, n::Int, m::Int; env::Union{Gurobi.Env, Nothing}=nothing)
# Create base model
    model = if solver_choice == :gurobi
        # Usa a constante global em vez de criar um novo env
        Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), MOI.Silent() => true, "Method" => 1, "Presolve" => 0))

    elseif solver_choice == :highs
        Model(optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "threads" => Threads.nthreads(), "primal_feasibility_tolerance" => 1e-8, "dual_feasibility_tolerance" => 1e-8))
    else
        error("Unsupported solver: $solver_choice")
    end
    model.obj_dict = Dict{Symbol, Any}()
    model.obj_dict[:phase] = phase
    if phase == :restauracao
        # Algorithm 4.1: Restoration subproblem structure
        # min e'w + eps * e'|d|  
        # s.t. A(z^ℓ)d + E(z^ℓ)w = -h(z^ℓ), |d| ≤ w_aux, z ∈ Ω, w ≥ 0
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
        # Algorithm 4.2: Optimization subproblem structure
        # min ∇L(y^k,λ')'(x-y^k)  s.t. J_h(y^k)(x-y^k) = 0
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

# Solves the specific restoration subproblem at a given point z
function solve_restoration_subproblem!(model::JuMP.Model, problem::OptimizationProblem, z::Vector{Float64}, δ::T) where {T<:Union{Float64, Vector{Float64}}}
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
        d_val = get_δ(δ, i)
        d_val = isfinite(d_val) ? d_val : 0.1
        zi = isfinite(z[i]) ? z[i] : 0.0
        lx = isfinite(problem.xl[i]) ? (problem.xl[i] - zi) : -Inf
        ux = isfinite(problem.xu[i]) ? (problem.xu[i] - zi) : Inf
        lb = max(lx, -d_val)
        ub = min(ux, d_val)
        lb = isfinite(lb) ? lb : -d_val
        ub = isfinite(ub) ? ub : d_val
        if lb > ub; lb = ub; end
        set_lower_bound(d[i], lb)
        set_upper_bound(d[i], ub)
    end
    # Determine E(z^ℓ) and update constraints: A(z^ℓ)d + E(z^ℓ)w = -h(z^ℓ)
    # To keep w ≥ 0, we need: E(z^ℓ) = -sign(h(z^ℓ))
    for i in 1:m
        Ei = h_z[i] > 0 ? -1.0 : (h_z[i] < 0 ? 1.0 : 0.0)
        for j in 1:n
            set_normalized_coefficient(con[i], d[j], A[i, j])
        end
        set_normalized_coefficient(con[i], w[i], Ei)
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

# Algorithm 4.1: Restoration Phase
function restoration_phase(
    problem::OptimizationProblem, 
    z::Vector{Float64}, 
    δ::T, 
    model::JuMP.Model; 
    params, 
    verbose::Bool=params.verbose_in, 
    buffers::OptimizedBuffers, 
    full_log::Vector{StepLog_TwoPhase}=Vector{StepLog_TwoPhase}(), 
    iterglobal::Int,
    history::Bool=true,
    t_start_alg::Float64=0.0
) where {T<:Union{Float64, Vector{Float64}}}
    # =========================================================
    # Step 1: Target parameters calculation based on original point
    # =========================================================
    h_init = problem.h(z)
    norm_h_init = norm(h_init)
    
    c_init = 0.5 * norm_h_init^2 

    r = params.r_resto 
    c_target = 0.5 * (r^2) * (norm_h_init^2) 

    eps_c = params.rfeas * norm_h_init 
    
    # =========================================================
    # Step 2: Initialization
    # =========================================================
    ℓ = 0
    c_prev = c_init
    h_z = h_init
    c_z = c_init
    
    verbose && println("🔄 Starting restoration phase...")
    params.debugverbose && println("DEBUG: Initial state - δ=$δ, norm_h_init=$norm_h_init, c_target=$c_target, eps_c=$eps_c")
    # Use temporary buffer to avoid modifying z directly
    buffers.z_temp .= z
    buffers.z_old .= z
    # Outer Loop (Iterations ℓ)
    while ℓ < params.max_iter_resto
        if params.max_time > 0 && t_start_alg > 0 && (time() - t_start_alg) > params.max_time
            params.debugverbose && println("⏱️ [DEBUG-RESTO] Timeout reached in restoration loop.")
            return buffers.z_temp, δ, false, h_z
        end
        
        # || P_Ω(z - ∇c(z)) - z ||
        A_z = problem.∇h(buffers.z_temp)
        grad_c = A_z' * h_z
        z_minus_grad = buffers.z_temp .- grad_c
        projected_step = clamp.(z_minus_grad, problem.xl, problem.xu)
        proj_grad_norm = norm(projected_step .- buffers.z_temp)
        params.debugverbose && println("DEBUG: Iter $ℓ - c(z)=$c_z, proj_grad_norm=$proj_grad_norm")
        # =========================================================
        # Step 3: Stopping criteria check
        # =========================================================
        if c_z <= c_target || proj_grad_norm <= eps_c
            verbose && println("✅ Restoration stopped in $ℓ iterations.")
            if c_z <= c_target
                params.debugverbose && println("DEBUG: Target feasibility reduction achieved (c_z <= c_target).")
            else
                params.debugverbose && println("DEBUG: Stationarity reached (proj_grad_norm <= eps_c).")
            end
            
            z .= buffers.z_temp
            return buffers.z_temp, δ, true, h_z
        end
        # =========================================================
        # Step 4: Inner loop initialization j <- 0 and choose δ
        # =========================================================
        j = 0
        success_step = false
        # Inner Loop (Iterations j - Trust region search)
        while j < params.max_iter_resto
            
            # =========================================================
            # Step 5: Subproblem resolution to compute z^{ℓ,j}
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
                # Step 6: Descent condition on c(z)
                # c(z^{ℓ,j}) ≤ c(z^ℓ) - α_R * ||z^{ℓ,j} - z^ℓ||^2
                # =========================================================
                step_norm_sq = dot(buffers.z_temp .- buffers.z_old, buffers.z_temp .- buffers.z_old)
                reduction_threshold = params.αR * step_norm_sq
                is_accepted = c_new <= c_z - reduction_threshold
                # Logging
                if history
                    log_entry = StepLog_TwoPhase(
                        iterglobal, copy(buffers.z_old), copy(buffers.z_temp), 
                        is_accepted ? :accepted : :rejected, :restoration, δ, 0.0, 0.0
                    )
                    push!(full_log, log_entry)
                end
                if is_accepted
                    success_step = true
                    c_prev = c_z
                    h_z = h_new
                    c_z = c_new
                    break 
                else
                    # Rejection: State restoration and strict trust region reduction
                    buffers.z_temp .= buffers.z_old
                    δ = max.(params.τ1 .* δ, params.δmin)
                end
            else
                # Solver failure handling via rejection
                buffers.z_temp .= buffers.z_old
                δ = max.(params.τ1 .* δ, params.δmin)
            end
            j += 1
        end 
        # Inner loop limit check
        if !success_step || j >= params.max_iter_resto
            verbose && println("⚠️ Restoration failure at iteration $ℓ (Too many inner iterations)")
            z .= buffers.z_temp
            return z, δ, false, h_z
        end
        # =========================================================
        # Step 7: Adaptive trust region update
        # (Feito APÓS a aceitação do passo para preparar o próximo ℓ)
        # =========================================================
        if params.restoration_ratio_update && ℓ > 0 && c_prev > params.ratio_safeguard_tol && @isdefined(c_new)
            ratio = clamp(c_new / c_prev, params.ratio_min_factor, params.ratio_max_factor)
            δ = clamp.(ratio .* δ, params.δmin, params.δmax)
        end
        # =========================================================
        # Step 7 (Theory): z^{ℓ+1} = z^{ℓ,j}, ℓ <- ℓ+1, return to Step 3
        # =========================================================
        # z^{ℓ+1} is already in buffers.z_temp due to success in the inner loop
        ℓ += 1
    end 

    z .= buffers.z_temp
    return z, δ, ℓ < params.max_iter_resto, h_z
end

# Solves the tangent step optimization subproblem
# min ∇f'·s  s.t. ∇h'·s = -h, bounds
function solve_optimization_subproblem!(model::JuMP.Model, n::Int, δ::T, problem::OptimizationProblem, buffers::OptimizedBuffers, params::TwoPhaseParams, B::Union{AbstractMatrix, Nothing}) where {T<:Union{Float64, Vector{Float64}}}
    m = length(buffers.h_old)
    grad_L = buffers.∇f + transpose(buffers.∇h) * buffers.λ
    
    # # Define quadratic matrix
    local Hqp
    params.use_quadratic && (Hqp = B)
    # if params.use_quadratic
    #     # Verifica o menor autovalor
    #     eig_min = minimum(eigvals(Matrix(B)))
        
    #     if eig_min < 1e-8
    #         # Adiciona um shift para tornar a matriz estritamente positiva
    #         shift = abs(eig_min) + 1e-4
    #         Hqp = B + (shift * I)
    #     else
    #         Hqp = B
    #     end
    # end
    # Helper: get trust region radius for dimension i
    get_δ(i) = (δ isa Vector) ? δ[i] : δ

    # Routing to RipQP
    if params.use_quadratic && params.quadratic_solver == :ripqp
        for i in 1:n
            d_val = get_δ(i)
            lx = problem.xl[i] - buffers.x_temp[i]
            ux = problem.xu[i] - buffers.x_temp[i]
            buffers.lvar[i] = max(lx, -d_val)
            buffers.uvar[i] = min(ux, d_val)
        end

        qm = QuadraticModel(buffers.∇f, Hqp; 
                            A=buffers.∇h, 
                            lcon=zeros(m), 
                            ucon=zeros(m), 
                            lvar=buffers.lvar, 
                            uvar=buffers.uvar, 
                            c0=0.0, 
                            name="OptSubproblem")

        stats = ripqp(qm, display=false)

        if stats.status == :first_order || stats.status == :acceptable
            copy!(buffers.s_val, stats.solution)
            if m > 0
                buffers.λ .= stats.multipliers[1:m]
            end
            return true, stats.objective
        end
    end

    # JuMP Internal Solver routing
    s = model[:s]
    con = model[:con]

    @inbounds for i in 1:n
        d_val = get_δ(i) # Uses the helper to handle Scalar vs Vector
        d_val = isfinite(d_val) ? d_val : params.δmin
        xi = isfinite(buffers.x_temp[i]) ? buffers.x_temp[i] : 0.0
        lx = isfinite(problem.xl[i]) ? (problem.xl[i] - xi) : -Inf
        ux = isfinite(problem.xu[i]) ? (problem.xu[i] - xi) : Inf
        lb = max(lx, -d_val)
        ub = min(ux, d_val)
        lb = isfinite(lb) ? lb : -d_val
        ub = isfinite(ub) ? ub : d_val
        if lb > ub; lb = ub; end
        set_lower_bound(s[i], lb)
        set_upper_bound(s[i], ub)
    end

    # Safeguard against non-finite coefficients (NaN/Inf in gradient or Hessian)
    if any(!isfinite, buffers.∇f) || any(!isfinite, grad_L) || (params.use_quadratic && (Hqp !== nothing && any(!isfinite, Hqp)))
        return false, 0.0
    end

    if params.use_quadratic
        @objective(model, Min, 0.5 * dot(s, Hqp * s) + dot(buffers.∇f, s))
    else
        @inbounds for j in 1:n
            set_objective_coefficient(model, s[j], grad_L[j])
        end
    end

    @inbounds for i in 1:m
        set_normalized_rhs(con[i], 0.0)
        for j in 1:n
            set_normalized_coefficient(con[i], s[j], buffers.∇h[i, j])
        end
    end

    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        @inbounds for i in 1:n
            buffers.s_val[i] = value(s[i])
        end
        if has_duals(model)
            @inbounds for i in 1:m
                buffers.λ[i] = dual(con[i])
            end
        else
            @inbounds for i in 1:m
                buffers.λ[i] = 0.0
            end
        end
        return true, objective_value(model)
    end

    return false, 0.0
end

# Executes the complete Optimization Phase (Algorithm 4.2)
function optimization_phase(
    problem::OptimizationProblem,
    x_k::Vector{Float64},      
    λ_k::Vector{Float64},      
    norm_h_xk::Float64,        
    y::Vector{Float64},        
    δ::Union{Float64, Vector{Float64}},
    model_opt::JuMP.Model,
    params;
    B::Union{AbstractMatrix, Nothing}=nothing,
    θ_k::Float64=0.5,          
    verbose::Bool=params.verbose_in,
    buffers::OptimizedBuffers,
    full_log::Vector{StepLog_TwoPhase}=Vector{StepLog_TwoPhase}(),
    iterglobal::Int,
    lagrangian_history::Vector{Float64}=Float64[],
    merit_history::Vector{Float64}=Float64[],
    history::Bool=true,
    F_y::Float64=NaN,
    h_y::Union{Vector{Float64}, Nothing}=nothing,
    F_xk::Float64=NaN,
    h_xk::Union{Vector{Float64}, Nothing}=nothing,
    t_start_alg::Float64=0.0
)

    # =========================================================
    # Step 1: Initialize j <- 0, choose δ_j and λ'
    # =========================================================
    j = 0
    n = length(y)
    m = problem.ncon
    θ = θ_k
    consecutive_rejects = 0

    buffers.x_temp .= y
    F_y_val = isnan(F_y) ? problem.f(y) : F_y
    h_y_val = h_y === nothing ? problem.h(y) : h_y
    norm_h_y = norm(h_y_val)
    

    F_xk_val = isnan(F_xk) ? problem.f(x_k) : F_xk
    h_xk_val = h_xk === nothing ? problem.h(x_k) : h_xk
    params.debugverbose && println("🔍 [DEBUG-OPT] Initial state: F(y^k)=$F_y_val, ||h(y^k)||=$norm_h_y, δ=$δ")
    params.debugverbose && println("🔍 [DEBUG-OPT] History sizes - Lagrangian: $(length(lagrangian_history)), Merit: $(length(merit_history))")

    buffers.∇f = problem.∇f(buffers.x_temp)
    buffers.∇h = problem.∇h(buffers.x_temp)

    use_new_atr = params.anisotropic_trust_region && params.atr_strategy != :legacy_parabolic
    g_prev_opt = if use_new_atr
        -compute_gradient_projection(
            buffers.x_temp, buffers.∇f, buffers.∇h, buffers.λ, problem.xl, problem.xu;
            norm_type = params.norm_gpnorm,
            strategy = params.gpnorm_strategy,
            max_iter = params.gpnorm_max_iter,
            tol = params.gpnorm_tol,
            debug = false
        )
    else
        Float64[]
    end

    local F_ret = F_y_val
    local h_ret = h_y_val
    local grad_ret = copy(buffers.∇f)
    # Main optimization iteration loop
    while j < params.max_iter_opt
        if params.max_time > 0 && t_start_alg > 0 && (time() - t_start_alg) > params.max_time
            params.debugverbose && println("⏱️ [DEBUG-OPT] Timeout reached in optimization subproblem loop.")
            return y, buffers.λ, θ_k, δ, false, F_y_val, h_y_val, buffers.∇f
        end
        params.debugverbose && println("\n🔍 [DEBUG-OPT] === Iteration j=$j started ===")
        # =========================================================
        # Step 2: Find z^{ℓ,j} by solving the subproblem
        # Step 3: Choose λ 
        # =========================================================
        success, f_model = solve_optimization_subproblem!(model_opt, n, δ, problem, buffers, params, B) # 
        # !params.use_quadratic && (buffers.λ .= 0.0)
        if !success
            δ = max.(params.τ2 .* δ, params.δmin)
            j += 1
            params.debugverbose && println("🚫 [DEBUG-OPT] Subproblem failure! Could not find a valid step.")
            params.debugverbose && println("🔄 Subproblem failure, reducing δ to $δ (j=$j)")
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
        # Step 4: Lagrangian descent evaluation
        # L(z^j, λ) ≤ L(y^k, λ) - α_L ||z^j - y^k||^2
        # =========================================================
        L_y_lambda = F_y_val + dot(buffers.λ, h_y_val) 
        L_zj_lambda = F_zj + dot(buffers.λ, h_zj)
        # Non-Monotonic Lagrangian Heuristic
        ref_lagrangian = isempty(lagrangian_history) ? L_y_lambda : max(L_y_lambda, maximum(lagrangian_history))
        lagrangian_threshold = ref_lagrangian - params.αL * step_norm_sq
        params.debugverbose && println("🔍 [DEBUG-OPT] --- Lagrangian Check ---")
        params.debugverbose && println("🔍 [DEBUG-OPT] L(y^k, λ): $L_y_lambda")
        params.debugverbose && println("🔍 [DEBUG-OPT] L(z^j, λ): $L_zj_lambda")
        params.debugverbose && println("🔍 [DEBUG-OPT] Ref Lagrangian: $ref_lagrangian")
        params.debugverbose && println("🔍 [DEBUG-OPT] Threshold to beat: $lagrangian_threshold (Drop req: $(params.αL * step_norm_sq))")
        if L_zj_lambda > lagrangian_threshold
            # -- step 4 fail --
            gap = L_zj_lambda - lagrangian_threshold
            params.debugverbose && println("🚫 [DEBUG-OPT] REJECTED by Lagrangian! L(z^j) is $gap ABOVE the threshold.")

            if history
                push!(full_log, StepLog_TwoPhase(
                    iterglobal, copy(buffers.x_old), copy(buffers.x_temp), 
                    :rejected, :optimization, δ, L_zj_lambda, compute_merit_functions(buffers.λ, θ_k, h_zj, F_zj)
                ))
            end
            consecutive_rejects += 1
            buffers.x_temp .= buffers.x_old # Take back de old step
            
            if params.use_unif_stopping && ((δ isa Vector) ? maximum(δ) : δ) <= params.δmin
                verbose && println("⚠️ δ too small, aborting optimization")
                return y, buffers.λ, θ_k, δ, false, F_y_val, h_y_val, buffers.∇f
            end
            # Trust region reduction post-rejection
            if !params.parabolic_heuristic && !use_new_atr
                s_ref = (δ isa Vector) ? abs.(buffers.s_val) : norm(buffers.s_val, Inf)
                δ_old = copy(δ)
                δ = clamp.(max.(δ .* params.τ1, s_ref .* params.τ2), params.δmin, params.δmax)
                params.debugverbose && println("🔍 [DEBUG-OPT] Standard Shrink: δ changed from $δ_old to $δ")
            else
                verbose && println("🔄 Lagrangian test failure, applying parabolic heuristic")
                s_inf = norm(buffers.s_val, Inf)

                if params.anisotropic_trust_region
                    parabolic_heuristic_mode = isodd(consecutive_rejects) ? :reshape : :shrink
                else
                    parabolic_heuristic_mode = :shrink
                end
                params.debugverbose && println("🔍 [DEBUG-OPT] Parabolic Heuristic (mode: $parabolic_heuristic_mode). Consecutive rejects: $consecutive_rejects")
                δ = parabolic_heuristic_step!(
                    δ, L_y_lambda, L_zj_lambda, f_model, params, 
                    s_inf, step_norm_sq, :decrease; 
                    anisotropic=params.anisotropic_trust_region, 
                    s_vec=buffers.s_val, grad=buffers.∇f, mode=parabolic_heuristic_mode,
                    min_reduction_ratio=params.parabolic_min_reduction_ratio,
                    max_reduction_ratio=params.parabolic_max_reduction_ratio,
                    min_increase_ratio=params.parabolic_min_increase_ratio,
                    max_increase_ratio=params.parabolic_max_increase_ratio,
                    ared_val=(L_y_lambda - L_zj_lambda),
                    pred_val=(-f_model),
                    g_prev=g_prev_opt,
                    d_curv=buffers.d_curv,
                )
                if params.anisotropic_trust_region && !use_new_atr && consecutive_rejects >= 2
                    was_corrected = enforce_aspect_ratio!(δ, 0.15)
                    was_corrected && params.verbose_out && verbose && println("🎈 [Rescue] Re-inflating dimensions.")
                end
            end
            
            # Go to Step 2 (Iter j+1)
            j += 1
            continue
        end
        params.debugverbose && println("✅ [DEBUG-OPT] ACCEPTED by Lagrangian check. (Beat threshold by $(lagrangian_threshold - L_zj_lambda))")

        consecutive_rejects = 0 
        # =========================================================
        # Step 5: Penalty parameter θ updates
        # =========================================================
        L_x_lambda_k = F_xk_val + dot(λ_k, h_xk_val)
        
        Phi_y_thetak = compute_merit_functions(buffers.λ, θ_k, h_y_val, F_y_val)
        Phi_xk_thetak = compute_merit_functions(λ_k, θ_k, h_xk_val, F_xk_val)
        # # Non-Monotonic Heuristic for Theta's test
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
                θ = clamp(numerador / denominador, params.θmin, θ_k)
            else
                params.debugverbose && println("⚠️ [DEBUG-OPT] Denominator too small, falling back to θ_k.")
                θ = θ_k
            end
        end

        # =========================================================
        # Step 6: Merit function checks
        # Φ(z^j, λ, θ) ≤ Φ(x^k, λ^k, θ) - α_Φ ||h(x^k)||
        # =========================================================
        Phi_zj_theta = compute_merit_functions(buffers.λ, θ, h_zj, F_zj)
        Phi_xk_theta = compute_merit_functions(λ_k, θ, h_xk_val, F_xk_val)
        # Non-Monotonic Heuristic for Final Merit
        ref_merit_theta = isempty(merit_history) ? Phi_xk_theta : max(Phi_xk_theta, maximum(merit_history))
        merit_threshold = ref_merit_theta - params.αΦ * norm_h_xk
        params.debugverbose && println("🔍 [DEBUG-OPT] --- Merit Function Check ---")
        params.debugverbose && println("🔍 [DEBUG-OPT] Φ(z^j, θ): $Phi_zj_theta")
        params.debugverbose && println("🔍 [DEBUG-OPT] Ref Φ(x^k, θ): $ref_merit_theta")
        params.debugverbose && println("🔍 [DEBUG-OPT] Threshold to beat: $merit_threshold")

        if history
            push!(full_log, StepLog_TwoPhase(
                iterglobal, copy(buffers.x_old), copy(buffers.x_temp), 
                Phi_zj_theta > merit_threshold ? :rejected : :accepted, 
                :optimization, δ, L_zj_lambda, Phi_zj_theta
            ))
        end
        if Phi_zj_theta > merit_threshold
            # -- Step 6 FAILS --
            gap_merit = Phi_zj_theta - merit_threshold
            params.debugverbose && println("🚫 [DEBUG-OPT] REJECTED by Merit Function! Φ(z^j) is $gap_merit ABOVE threshold.")
            
            buffers.x_temp .= buffers.x_old 
            if params.use_unif_stopping && ((δ isa Vector) ? maximum(δ) : δ) <= params.δmin + 1e-9
                params.debugverbose && println("⚠️ δ too small, aborting optimization")
                return y, buffers.λ, θ, δ, false, F_ret, h_ret, grad_ret
            end
            δ_old = copy(δ)
            if use_new_atr
                s_inf = norm(buffers.s_val, Inf)
                pred_merit = max(1e-12, θ * max(-f_model, 0.0) + (1.0 - θ) * max(norm_h_xk - norm_h_y, 0.0))
                ared_merit = Phi_xk_theta - Phi_zj_theta
                δ = parabolic_heuristic_step!(
                    δ, Phi_xk_theta, Phi_zj_theta, -pred_merit, params,
                    s_inf, step_norm_sq, :decrease;
                    anisotropic=true,
                    s_vec=buffers.s_val, grad=buffers.∇f, mode=:shrink,
                    min_reduction_ratio=params.parabolic_min_reduction_ratio,
                    max_reduction_ratio=params.parabolic_max_reduction_ratio,
                    min_increase_ratio=params.parabolic_min_increase_ratio,
                    max_increase_ratio=params.parabolic_max_increase_ratio,
                    ared_val=ared_merit,
                    pred_val=pred_merit,
                    g_prev=g_prev_opt,
                    d_curv=buffers.d_curv,
                )
            else
                δ = clamp.(δ .* params.τ2, params.δmin, params.δmax)
            end
            j += 1
            
            params.debugverbose && println("🔍 [DEBUG-OPT] Merit failure. Shrinking δ from $δ_old to $δ")
            params.debugverbose && println("🔄 Merit test failure, reducing δ to $δ (j=$j)")
            continue
        end
        # =========================================================
        # Step 7: Step Accepted! Update final state and δ for the next step
        # =========================================================
        params.debugverbose && println("✅ [DEBUG-OPT] ACCEPTED by Merit function! (Beat threshold by $(merit_threshold - Phi_zj_theta))")
        params.debugverbose && println("✅ Step accepted in optimization (j=$j)")
        

        ∇f_new_accepted = problem.∇f(buffers.x_temp)
        # buffers.∇f_old .= ∇f_new_accepted
        F_ret = F_zj
        h_ret = h_zj
        grad_ret = ∇f_new_accepted
        
        # Trust region expansion
        if !params.parabolic_heuristic && !use_new_atr
            if params.strong_agreement_rule
                ared = L_y_lambda - L_zj_lambda
                pred = - f_model
                
                rho_unif = 0.5
                
                if ared >= rho_unif * pred
                    δ_old = copy(δ)
                    δ = min.(δ ./ params.τ3, params.δmax)
                    params.debugverbose && println("🔍 [DEBUG-OPT] Trust Region expanded from $δ_old to $δ")
                end
            else
                δ_old = copy(δ)
                δ = min.(δ ./ params.τ3, params.δmax)
                params.debugverbose && println("🔍 [DEBUG-OPT] Trust Region expanded from $δ_old to $δ")
            end
        else
            params.debugverbose && println("🔍 [DEBUG-OPT] Applying Quadratic Backtracking for Trust Region expansion.")
            s_inf = norm(buffers.s_val, Inf)
            if use_new_atr
                jac_new_acc = problem.∇h(buffers.x_temp)
                g_curr_opt = -compute_gradient_projection(
                    buffers.x_temp, ∇f_new_accepted, jac_new_acc, buffers.λ, problem.xl, problem.xu;
                    norm_type = params.norm_gpnorm,
                    strategy = params.gpnorm_strategy,
                    max_iter = params.gpnorm_max_iter,
                    tol = params.gpnorm_tol,
                    debug = false
                )
                jac_diff_l1 = zeros(n)
                if m > 0
                    @inbounds for col in 1:n
                        csum = 0.0
                        for row in 1:m
                            csum += abs(jac_new_acc[row, col] - buffers.∇h[row, col])
                        end
                        jac_diff_l1[col] = csum
                    end
                end
                δ = parabolic_heuristic_step!(
                    δ, L_y_lambda, L_zj_lambda, f_model, params, 
                    s_inf, step_norm_sq, :increase; 
                    anisotropic=true, 
                    s_vec=buffers.s_val, grad=∇f_new_accepted, 
                    grad_old=buffers.∇f_old, mode=:shrink,
                    min_reduction_ratio=params.parabolic_min_reduction_ratio,
                    max_reduction_ratio=params.parabolic_max_reduction_ratio,
                    min_increase_ratio=params.parabolic_min_increase_ratio,
                    max_increase_ratio=params.parabolic_max_increase_ratio,
                    ared_val=(L_y_lambda - L_zj_lambda),
                    pred_val=(-f_model),
                    g_prev=g_prev_opt,
                    g_curr=g_curr_opt,
                    jac_diff_col_l1=jac_diff_l1,
                    theta_merit=θ,
                    d_curv=buffers.d_curv,
                )
            else
                δ = parabolic_heuristic_step!(
                    δ, L_y_lambda, L_zj_lambda, f_model, params, 
                    s_inf, step_norm_sq, :increase; 
                    anisotropic=params.anisotropic_trust_region, 
                    s_vec=buffers.s_val, grad=∇f_new_accepted, 
                    grad_old=buffers.∇f_old, mode=:shrink
                )
            end
        end
        break 
    end

    y .= buffers.x_temp
    success = j < params.max_iter_opt
    params.debugverbose && println("🏁 [DEBUG-OPT] Optimization phase finished. Success: $success. Total inner iters: $j")
    return y, buffers.λ, θ, δ, success, F_ret, h_ret, grad_ret
end

# Main wrapper algorithm orchestrating restoration and optimization (Algorithm 4.3)
function two_phase_optimization(
    problem::OptimizationProblem,
    x0::Vector{Float64},
    params::TwoPhaseParams;
    max_outer_iter::Int=params.max_outer_iter,
    tolerance::Float64=params.tol,
    history::Bool=true,
)
    solver_choice=params.general_solver
    verbose=params.verbose_out

    logio = nothing
    try
        params.debugverbose && println("🚀 Starting two-phase optimization")
        if params.debugverbose
            println("DEBUG: === INITIALIZATION ===")
            println("DEBUG: Basic parameters:")
            println("  max_outer_iter = $max_outer_iter")
            println("  tolerance = $tolerance")
            println("  r_resto = $(params.r_resto)")
            println("  rfeas = $(params.rfeas)")
        end
        n = length(x0)
        m = problem.ncon

        x = clamp.(copy(x0), problem.xl, problem.xu)
        λ = zeros(m)
        θ = params.θ0
        δr = params.anisotropic_trust_region ? fill(params.δ0_resto, n) : params.δ0_resto
        δo = params.anisotropic_trust_region ? fill(params.δ0_opt, n) : params.δ0_opt
        iter = 0
        t_start_alg = time()
        s_opt_outer = zeros(n)
        y_opt_outer = zeros(n)
        B   = params.use_quadratic ? spdiagm(0 => ones(n)) : nothing

        countG = 0
        countF = 0
        countS = 0
        # Create buffers
        buffers = OptimizedBuffers(n, m)
        Lgrad_current = zeros(n)
        # History tracking
        x_hist = history ? Vector{Vector{Float64}}(undef, max_outer_iter + 1) : nothing
        if history
            x_hist[1] = copy(x)
        end
        full_log = history ? Vector{StepLog_TwoPhase}() : StepLog_TwoPhase[]
        # === Initialization of Non-Monotonic Buffers ===
        history_size = params.non_monotone_M
        lagrangian_history = Float64[]
        merit_history = Float64[]

        f_current = problem.f(x)
        h_current = problem.h(x)
        norm_h_current = norm(h_current)
        buffers.∇f_old = problem.∇f(x)
        
        # Initial state population in non-monotonic history
        push!(lagrangian_history, f_current + dot(λ, h_current))
        push!(merit_history, compute_merit_functions(λ, θ, h_current, f_current))
        # ====================================================================
        # Initial solution optimality check
        # ====================================================================
        jac_h = problem.∇h(x)
        gradpbox = compute_gradient_projection(
            x, buffers.∇f_old, jac_h, λ, problem.xl, problem.xu;
            norm_type = params.norm_gpnorm,
            strategy = params.gpnorm_strategy,
            max_iter = params.gpnorm_max_iter,
            tol = params.gpnorm_tol,
            debug = params.gpnorm_debug
        )
        
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
            if verbose || logio !== nothing
                log_message!("✅ INITIAL SOLUTION IS OPTIMAL!", logio; verbose=verbose)
                log_message!("   KKT conditions are satisfied.", logio; verbose=verbose)
            end
            if params.use_unif_stopping
                countG = params.maxcount
                countF = params.maxcount
            end
        end
        # Create models


        model_restaura = create_optimized_model(solver_choice, :restauracao, n, m; env=GRB_ENV)
        model_opt = create_optimized_model(solver_choice, :otimizacao, n, m; env=GRB_ENV)
        # Table header
        if verbose || logio !== nothing
            log_message!(repeat("=", 120), logio; verbose=verbose)
            log_message!("                    TWO-PHASE OPTIMIZATION - ALGORITHM 4.3", logio; verbose=verbose)
            log_message!(repeat("=", 120), logio; verbose=verbose)
            if params.use_unif_stopping
                msg = @sprintf("%-4s %-12s %-10s %-8s %-8s %-9s %-10s %-10s %-18s", "k", "f(xᵏ)", "||h(xᵏ)||", "δr", "δo", "||λ||", "||Δx||", "||gpnorm||", "(G|F|S)")
            else
                msg = @sprintf("%-4s %-12s %-10s %-8s %-8s %-9s %-10s  %-10s", "k", "f(xᵏ)", "||h(xᵏ)||", "δr", "δo", "||λ||", "||gpnorm||", "||Δx||")
            end
            log_message!(msg, logio; verbose=verbose)
            log_message!(repeat("-", 120), logio; verbose=verbose)
            d_r_print = (δr isa Vector) ? maximum(δr) : δr
            d_o_print = (δo isa Vector) ? maximum(δo) : δo
            lambda_print_init = norm(λ)
            if params.use_unif_stopping
                msg = @sprintf("%-4d %-12.5e %-10.3e %-8.2e %-8.2e %-9.2e %-10s %-10.3e %-18s", 0, f_current, norm_h_current, d_r_print, d_o_print, lambda_print_init, "-", gpnorm, "0|0|0")
            else
                msg = @sprintf("%-4d %-12.5e %-10.3e %-8.2e %-8.2e %-9.2e %-10.3e %-10s", 0, f_current, norm_h_current, d_r_print, d_o_print, lambda_print_init, gpnorm, "-")
            end
            log_message!(msg, logio; verbose=verbose)
        end
        # Main loop (Algorithm 4.3)
        while iter < max_outer_iter
            if params.max_time > 0 && (time() - t_start_alg) > params.max_time
                if verbose || logio !== nothing
                    log_message!("⏱️ TIMEOUT: Execution exceeded $(params.max_time)s limit", logio; verbose=verbose)
                end
                return x, λ, θ, iter, 5, history ? x_hist[1:(iter + 1)] : nothing, full_log
            end
            if any(!isfinite, x)
                if verbose || logio !== nothing
                    log_message!("❌ BREAKDOWN: Non-finite value (NaN/Inf) detected in x vector", logio; verbose=verbose)
                end
                return x, λ, θ, iter, 4, history ? x_hist[1:(iter + 1)] : nothing, full_log
            end
            
            # --- Convergence checking ---
            if params.use_unif_stopping
                is_feasible = norm_h_current < tolerance
                
                unif_converged_kkt = (countG >= params.maxcount && countF >= params.maxcount && is_feasible)

                unif_converged_step = (countS >= params.maxcount && is_feasible) 
                if unif_converged_kkt || unif_converged_step
                    if verbose || logio !== nothing
                        log_message!("-" ^ 120, logio; verbose=verbose)
                        if unif_converged_kkt
                            log_message!("✅ CONVERGENCE - KKT CONDITIONS: Small gradient, reduction, and constraint violation", logio; verbose=verbose)
                        else
                            log_message!("✅ CONVERGENCE - SMALL STEP: Sufficiently small step at a feasible point", logio; verbose=verbose)
                        end
                    end
                    break
                elseif (countG >= params.maxcount || countS >= params.maxcount) && !is_feasible
                    if verbose || logio !== nothing
                        log_message!("⚠️ WARNING: Algorithm stagnated at an infeasible point (||h(x)|| = $norm_h_current)", logio; verbose=verbose)
                    end
                end
            else
                #UNIF OFF, stop if KKT
                if norm_h_current < tolerance && gpnorm < params.tolG
                    if verbose || logio !== nothing
                        log_message!("-" ^ 120, logio; verbose=verbose)
                        log_message!("✅ CONVERGENCE: Exact KKT solution found", logio; verbose=verbose)
                    end
                    break
                end
            end
            x_old_outer = copy(x)
            norm_h_old = norm_h_current
            if params.use_quadratic && iter > 0
                if norm(s_opt_outer) > 1e-12
                    B = Generic_module_ComputeB.compute_B(params.B_update_strategy, n, s_opt_outer, y_opt_outer, params.σ, problem, x, λ, B)
                end
            end
            buffers.Lgrad_old = copy(Lgrad_current)
            # ================================================================
            # Step 1: RESTORATION PHASE
            # ================================================================
            if norm_h_current > tolerance
                y, δr, success_resto, h_resto = restoration_phase(problem, x, δr, model_restaura; params=params, buffers=buffers, full_log=full_log, iterglobal=iter, history=history, t_start_alg=t_start_alg)
                if params.max_time > 0 && (time() - t_start_alg) > params.max_time
                    if verbose || logio !== nothing
                        log_message!("⏱️ TIMEOUT: Execution exceeded $(params.max_time)s limit", logio; verbose=verbose)
                    end
                    return y, λ, θ, iter, 5, history ? x_hist[1:(iter + 1)] : nothing, full_log
                end
                norm_h_y = norm(h_resto)
                if norm_h_y > params.r_resto * norm_h_old
                    if verbose || logio !== nothing
                        log_message!("❌ FAILURE: ||h(yᵏ)|| > r||h(xᵏ)|| - Insufficient restoration", logio; verbose=verbose)
                    end
                    if history
                        x_hist[iter + 2] = copy(y)
                    end
                    return y, λ, θ, iter, 3, history ? x_hist[1:(iter + 2)] : nothing, full_log
                end
                F_y_pass = NaN
                h_y_pass = h_resto
            else
                y = copy(x)
                success_resto = true
                F_y_pass = f_current
                h_y_pass = h_current
            end
            # ================================================================
            # Step 2: OPTIMIZATION PHASE
            # ================================================================

            x_new, λ_new, θ_new, δo, success_opt, f_opt_new, h_opt_new, grad_f_opt_new = optimization_phase(
                problem, 
                x_old_outer,      
                λ,                
                norm_h_old,       
                y,                
                δo, 
                model_opt, 
                params; 
                B = B,        
                θ_k = θ,          
                verbose = params.verbose_in, 
                buffers = buffers, 
                full_log = full_log, 
                iterglobal = iter,
                lagrangian_history = lagrangian_history,
                merit_history = merit_history,
                history = history,
                F_y = F_y_pass,
                h_y = h_y_pass,
                F_xk = f_current,
                h_xk = h_current,
                t_start_alg = t_start_alg
            )
            if params.max_time > 0 && (time() - t_start_alg) > params.max_time
                if verbose || logio !== nothing
                    log_message!("⏱️ TIMEOUT: Execution exceeded $(params.max_time)s limit", logio; verbose=verbose)
                end
                return x_new, λ_new, θ_new, iter, 5, history ? x_hist[1:(iter + 1)] : nothing, full_log
            end
            # ================================================================
            # Step 3: UPDATE AND LOG
            # ================================================================
            x .= x_new
            λ .= λ_new
            θ = θ_new
            
            f_new = f_opt_new
            h_new = h_opt_new
            norm_h_new = norm(h_new)
            Δx = norm(x - x_old_outer, Inf)
            ΔF = abs(f_new - f_current)
            
            # --- Recalculate Gradients at new point ---
            grad_f_new = grad_f_opt_new
            jac_h_new = problem.∇h(x)
            Lgrad = grad_f_new + jac_h_new' * λ

            # ================================================================
            # Captura do par secante (s, y) PURO da fase de otimização
            # ================================================================
            if success_opt && norm(buffers.s_val) > 1e-12
                s_opt_outer .= buffers.s_val
                
                # Lagrangiano no ponto inicial da otimização (y)
                # buffers.∇f e buffers.∇h contêm os gradientes em y calculados em optimization_phase
                Lgrad_y = buffers.∇f + buffers.∇h' * λ
                
                y_opt_outer .= Lgrad .- Lgrad_y
            else
                # Passo de otimização nulo/rejeitado: preserva a curvatura já acumulada.
                s_opt_outer .= 0.0
            end

            # --- Recalculate GP Norm (AGP) ---
            gradpbox = compute_gradient_projection(
                x, grad_f_new, jac_h_new, λ, problem.xl, problem.xu;
                norm_type = params.norm_gpnorm,
                strategy = params.gpnorm_strategy,
                max_iter = params.gpnorm_max_iter,
                tol = params.gpnorm_tol,
                debug = params.gpnorm_debug
            )
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
            # === Updating Non-Monotonic Memory ===
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
            # --- Update UNIF Counters ---
            if params.use_unif_stopping
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
                    countS += 1
                else
                    countS = 0
                end
            end

            status_symbol = ""
            if params.use_unif_stopping
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
            if verbose || logio !== nothing
                d_r_print = (δr isa Vector) ? maximum(δr) : δr
                d_o_print = (δo isa Vector) ? maximum(δo) : δo 
                lambda_print = norm(λ)

                if params.use_unif_stopping
                    msg = @sprintf("%-4d %-12.5e %-10.3e %-8.2e %-8.2e %-9.2e %-10.3e %-10.3e %d|%d|%d%s", iter, f_new, norm_h_new, d_r_print, d_o_print, lambda_print, Δx, gpnorm, countG, countF, countS, status_symbol)
                else
                    msg = @sprintf("%-4d %-12.5e %-10.3e %-8.2e %-8.2e %-9.2e %-10.3e %-10.3e %s", iter, f_new, norm_h_new, d_r_print, d_o_print, lambda_print, gpnorm, Δx, status_symbol)
                end
                log_message!(msg, logio; verbose=verbose)
            end
            f_current = f_new
            h_current = h_new
            norm_h_current = norm_h_new
        end
        # ================================================================
        # FINALIZATION
        # ================================================================
        if verbose || logio !== nothing
            log_message!(repeat("=", 120), logio; verbose=verbose)
        end
        opstop = if params.use_unif_stopping
            if countG >= params.maxcount && countF >= params.maxcount && norm_h_current < tolerance
                0
            elseif countS >= params.maxcount && norm_h_current < tolerance
                1
            elseif iter >= max_outer_iter
                2
            else
                3
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
        if params.use_unif_stopping
            termination_msg = if opstop == 0
                "✅ CONVERGENCE - KKT CONDITIONS: Small gradient and reduction sustained for $(params.maxcount) consecutive iterations"
            elseif opstop == 1
                "✅ CONVERGENCE - SMALL STEP: Sustained small step for $(params.maxcount) consecutive iterations"
            elseif opstop == 2
                "⏰ MAXIMUM ITERATIONS: Reached limit of $max_outer_iter iterations"
            else
                "❌ TERMINATION BY FAILURE: Issues encountered during restoration or optimization phases"
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
        if verbose || logio !== nothing
            log_message!(termination_msg, logio; verbose=verbose)
            log_message!("Optimization finished after $iter iterations.", logio; verbose=verbose)
            log_message!("🎉 Final result:", logio; verbose=verbose)
            log_message!(@sprintf("   f(x*) = %.6e", f_current), logio; verbose=verbose)
            log_message!(@sprintf("   ||h(x*)|| = %.2e", norm_h_current), logio; verbose=verbose)
            d_r_print = (δr isa Vector) ? maximum(δr) : δr
            d_o_print = (δo isa Vector) ? maximum(δo) : δo
            log_message!(@sprintf("   δr (max) = %.2e, δo (max) = %.2e", d_r_print, d_o_print), logio; verbose=verbose)
        end
        return x, λ, θ, iter, opstop, history ? x_hist[1:(iter + 1)] : nothing, full_log
              
    finally
        if logio !== nothing
            close(logio)
        end
    end
end
end