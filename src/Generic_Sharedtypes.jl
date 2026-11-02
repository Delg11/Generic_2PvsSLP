module SharedTypes

export StepLog, StepLog_UNIF, StepLog_TwoPhase, OptimizationProblem, OptimizedBuffers, TwoPhaseParams, UNIFParams, create_rosenbrock_problem0, create_rosenbrock_problem1, build_optimization_problem, GRB_ENV, SMALL_PROBLEMS,
       enforce_aspect_ratio!, parabolic_heuristic_step!, compute_gradient_projection, get_δ, max_δ

using ADNLPModels, NLPModels, Gurobi, JuMP, LinearAlgebra, Printf, SparseArrays

# Global environment for Gurobi to prevent license checkout overhead on every iteration
const GRB_ENV = Gurobi.Env(output_flag=0)

# Trust region helper functions (scalar or vector)
@inline get_δ(δ::Float64, i::Int) = δ
@inline get_δ(δ::Vector{Float64}, i::Int) = @inbounds δ[i]
@inline get_δ(δ::Union{Float64, Vector{Float64}}, i::Int) = (δ isa Vector) ? δ[i] : δ

@inline max_δ(δ::Float64) = δ
@inline max_δ(δ::Vector{Float64}) = maximum(δ)
@inline max_δ(δ::Union{Float64, Vector{Float64}}) = (δ isa Vector) ? maximum(δ) : δ

# ==============================================================================
# LOGGING STRUCTURES
# ==============================================================================

"""
Abstract base type for step iteration logs.
"""
abstract type AbstractStepLog end

"""
Logging structure specific to the SLP (Sequential Linear Programming) UNIF algorithm.
Records the states and metrics for a single optimization step.
"""
Base.@kwdef struct StepLog_UNIF <: AbstractStepLog
    iter::Int
    x_from::Vector{Float64}
    x_to::Vector{Float64}
    status::Symbol
    phase::Symbol
    delta::Union{Float64, Vector{Float64}}
    # UNIF-specific metrics
    ared::Float64 = 0.0
    pred::Float64 = 0.0
end

"""
Logging structure specific to the Two-Phase algorithm.
Records the states and metrics for a single optimization step, including merit function evaluations.
"""
Base.@kwdef struct StepLog_TwoPhase <: AbstractStepLog
    iter::Int
    x_from::Vector{Float64}
    x_to::Vector{Float64}
    status::Symbol
    phase::Symbol
    delta::Union{Float64, Vector{Float64}}
    # TwoPhase-specific metrics
    Lagrange::Float64
    Merit::Float64
end

# ==============================================================================
# PROBLEM & BUFFER STRUCTURES
# ==============================================================================

"""
Core mathematical formulation of the nonlinear optimization problem.
Encapsulates all necessary evaluation functions (objective, constraints, derivatives) 
and problem dimensions.
"""
struct OptimizationProblem{F, G, H, J, Hc, C, GC, HC, P, PG, L, M, HL}
    name::String           # Problem identifier
    nvar::Int              # Number of decision variables
    ncon::Int              # Number of constraints
    f::F                   # Objective function: f(x)
    ∇f::G                  # Gradient of the objective: ∇f(x)
    h::H                   # Equality constraints function: h(x)
    ∇h::J                  # Jacobian of constraints: ∇h(x)
    ∇²h::Hc                # Hessians of individual constraints: ∇²h_i(x)
    c::C                   # Constraint penalty metric (e.g., squared norm)
    ∇c::GC                 # Gradient of the penalty metric
    ∇²c::HC                # Hessian of the penalty metric
    project::P             # Projection operator for box constraints
    prox_grad_c::PG        # Proximal gradient mapping for constraint penalty
    Lagrangian::L          # Lagrangian function: L(x, λ)
    merit::M               # Merit function combining objective and constraint violation
    ∇²L::HL                # Hessian of the Lagrangian: ∇²L(x, λ)
    x0::Vector{Float64}    # Initial primal point
    y0::Vector{Float64}    # Initial dual multipliers
    xl::Vector{Float64}    # Variable lower bounds
    xu::Vector{Float64}    # Variable upper bounds
end

"""
Pre-allocated memory arrays used during optimization loops to avoid runtime memory allocation overhead.
"""
mutable struct OptimizedBuffers
    s_val::Vector{Float64}; ∇f::Vector{Float64}; ∇f_old::Vector{Float64}
    F_new::Float64; F_old::Float64
    x_old::Vector{Float64}; x_temp::Vector{Float64}
    z_temp::Vector{Float64}; z_old::Vector{Float64}
    h_old::Vector{Float64}; ∇h::Matrix{Float64}; λ::Vector{Float64}
    lvar::Vector{Float64}; uvar::Vector{Float64}
    Lgrad_old::Vector{Float64}
    d_curv::Vector{Float64}

    function OptimizedBuffers(n::Int, m::Int)
        new(zeros(n), zeros(n), zeros(n), 0.0, 0.0, zeros(n), zeros(n), 
            zeros(n), zeros(n), zeros(m), zeros(m, n), zeros(m), zeros(n), zeros(n),
            zeros(n), ones(n))
    end
end

# ==============================================================================
# ALGORITHM CONFIGURATION PARAMETERS
# ==============================================================================

"""
Configuration parameters governing the behavior of the Two-Phase optimization algorithm.
Includes tolerances, solver choices, trust region update strategies, and iteration limits.
"""
Base.@kwdef mutable struct TwoPhaseParams

    # General Solver Configuration
    general_solver::Symbol = :gurobi
    
    # Restoration Phase Settings
    r_resto::Float64 = 0.9          # Required feasibility improvement ratio
    rfeas::Float64 = 1e-12          # Absolute feasibility tolerance
    βc::Float64 = 0.5               # Backtracking coefficient for constraints
    αR::Float64 = 1e-4              # Armijo condition parameter for restoration step
    τ1::Float64 = 0.1               # Trust region contraction factor (severe)
    τ2::Float64 = 0.25              # Trust region contraction factor (moderate)
    τ3::Float64 = 0.5               # Trust region contraction factor (mild)
    θ0::Float64 = 0.9               # Initial penalty parameter weighting
    θmin::Float64 = 1e-6            # Minimum allowed penalty parameter
    tol::Float64 = 1e-4             # General stopping tolerance

    # Optimization Phase Settings
    αΦ::Float64 = (1 - 0.9)/2       # Armijo condition parameter for merit function
    αL::Float64 = 1e-6              # Minimum acceptable objective decrease

    # Trust Region Boundaries
    δ0_opt::Float64 = 0.1           # Initial trust region radius for optimization
    δ0_resto::Float64 = 0.5         # Initial trust region radius for restoration
    δmin::Float64 = 1e-6            # Minimum allowed trust region radius
    δmax::Float64 = 1.0             # Maximum allowed trust region radius

    # Iteration Limits and Stagnation Controls
    max_iter_resto::Int = 50        # Maximum iterations within a single restoration phase
    max_iter_opt::Int = 50          # Maximum iterations within a single optimization phase
    max_outer_iter::Int = 500       # Maximum overall algorithm iterations
    max_time::Float64 = 60.0        # Maximum wall-clock time in seconds per variant (<=0 for unlimited)
    stag_tol::Float64 = 1e-5        # Step size threshold to trigger stagnation counter
    max_stag::Int = 3               # Maximum consecutive stagnated iterations allowed
    non_monotone_M::Int = 5         # History length for non-monotone acceptance criteria

    # Trust Region Update Strategies
    strong_agreement_rule::Bool = true # flag to enable/disable ared/pred ratio-based trust region expansion
    parabolic_heuristic::Bool = true
    anisotropic_trust_region::Bool = true
    atr_strategy::Symbol = :legacy_parabolic # Options: :legacy_parabolic, :grad_sign, :decomposed_cut, :secant_box
    atr_max_cond::Float64 = 100.0            # Maximum anisotropy condition number max(δ)/min(δ)
    secant_ema_beta::Float64 = 0.5           # EMA smoothing factor for diagonal secant curvature
    restoration_ratio_update::Bool = true
    ratio_safeguard_tol::Float64 = 1e-12
    ratio_max_factor::Float64 = 2.0
    ratio_min_factor::Float64 = 0.1
    parabolic_min_reduction_ratio::Float64 = 0.1
    parabolic_max_reduction_ratio::Float64 = 0.5
    parabolic_min_increase_ratio::Float64 = 1.0
    parabolic_max_increase_ratio::Float64 = 2.0

    # Sequential Quadratic Programming (SQP) Configuration
    use_quadratic::Bool = false     # Flag to enable/disable quadratic model usage
    quadratic_solver::Symbol = :gurobi # Selected solver for QP subproblems
    σ::Float64 = 0.1                # Matrix regularization term (H + σI)
    B_update_strategy::Symbol = :identity # Strategy for Hessian approximation (:identity, :spectral, :exact)

    # Standard UNIF Stopping Criteria (KKT and step-based)
    maxcount::Int = 3
    tolG::Float64 = 1e-3
    tolF::Float64 = 5e-2
    tolS::Float64 = 1e-4
    use_unif_stopping::Bool = true
    
    # Verbosity and Debugging Options
    verbose_in::Bool = false
    verbose_out::Bool = true
    scale::Bool = false
    debugverbose::Bool = false
    log_to_file::Bool = false

    # GpNorm
    gpnorm_div_nelem::Bool = true
    norm_gpnorm::Union{Int, Float64} = Inf
    gpnorm_strategy::Symbol = :dykstra
    gpnorm_debug::Bool = false
    gpnorm_max_iter::Int = 500
    gpnorm_tol::Float64 = 1e-8


end

"""
Configuration parameters specifically for the standard UNIF (Unified Iterative Framework) algorithm.
"""
Base.@kwdef mutable struct UNIFParams
    maxiter::Int = 500              # Maximum total iterations
    max_time::Float64 = 60.0        # Maximum wall-clock time in seconds per variant (<=0 for unlimited)
    tolG::Float64 = 1e-3            # Tolerance for projected gradient
    tolF::Float64 = 5e-2            # Tolerance for objective function decrease
    tolS::Float64 = 1e-3            # Tolerance for step size magnitude
    maxcount::Int = 3               # Required consecutive successful checks for termination

    # Trust Region Update Parameters
    eta::Float64 = 0.1              # Threshold ratio for accepting a step
    rho::Float64 = 0.5              # Threshold ratio for expanding the trust region
    delta0::Float64 = 0.1           # Initial trust region radius
    alphaR::Float64 = 0.25          # Multiplier for shrinking the trust region radius
    alphaA::Float64 = 2.0           # Multiplier for expanding the trust region radius

    verbose::Bool = true
    debugverbose::Bool = false
    output_flag::Int = 0            # Controls Gurobi's internal solver output (0 = silent)

    # parabolic heuristic and Anisotropy Settings
    parabolic_heuristic::Bool = true
    anisotropic_trust_region::Bool = true
    atr_strategy::Symbol = :legacy_parabolic # Options: :legacy_parabolic, :grad_sign, :decomposed_cut, :secant_box
    atr_max_cond::Float64 = 100.0            # Maximum anisotropy condition number max(δ)/min(δ)
    secant_ema_beta::Float64 = 0.5           # EMA smoothing factor for diagonal secant curvature
    parabolic_min_reduction_ratio::Float64 = 0.1
    parabolic_max_reduction_ratio::Float64 = 0.5
    parabolic_min_increase_ratio::Float64 = 1.0
    parabolic_max_increase_ratio::Float64 = 2.0
    
    # Internal Safeguards for Trust Region Adjustment
    δmin::Float64 = 1e-6
    δmax::Float64 = 1
    ratio_safeguard_tol::Float64 = 1e-8
    τ1::Float64 = 0.25              # Severe reduction multiplier
    τ2::Float64 = 0.5               # Moderate reduction multiplier
    strong_agreement_rule::Bool = true

    # Sequential Quadratic Programming (SQP) Configuration
    use_quadratic::Bool = false
    quadratic_solver::Symbol = :gurobi
    σ::Float64 = 0.1                  # Matrix regularization term (H + σI)
    B_update_strategy::Symbol = :identity 
    
    verbose_out::Bool = false       # Specific verbosity flag for operations


    # GpNorm
    # gpnorm_div_nelem::Bool = true
    norm_gpnorm::Union{Int, Float64} = Inf
    gpnorm_strategy::Symbol = :dykstra
    gpnorm_debug::Bool = false
    gpnorm_max_iter::Int = 500
    gpnorm_tol::Float64 = 1e-8
end

# ==============================================================================
# PROBLEM GENERATORS
# ==============================================================================

"""
Generates a 2D unconstrained Rosenbrock problem formatted as an ADNLPModel.
Objective: min f(x) = (1-x₁)² + 100(x₂-x₁²)²
"""
function create_rosenbrock_problem0()
    objective(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

    x0 = [10.0, 10.0]
    xl = fill(-10.0, 2)
    xu = fill(10.0, 2)

    # Passing only objective and bounds natively constructs an unconstrained model
    return ADNLPModel(objective, x0, xl, xu; name="Rosenbrock_Problem0_Unconstrained")
end

"""
Generates a 2D Rosenbrock problem subject to a circular equality constraint.
Objective: min f(x) = (1-x₁)² + 100(x₂-x₁²)²
Subject to: x₁² + x₂² - 6 = 0
"""
function create_rosenbrock_problem1()
    objective(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

    constraints(x) = [
        x[1]^2 + x[2]^2 - 6,
    ]

    x0 = [10.0, 10.0]
    xl = fill(-10.0, 2)
    xu = fill(10.0, 2)
    lcon = [0.0]
    ucon = [0.0]

    return ADNLPModel(objective, x0, xl, xu, constraints, lcon, ucon; name="Rosenbrock_Problem1_Circular")
end

"""
Constructs the unified OptimizationProblem structure from a generic NLPModel interface.
Automatically handles maximization vs minimization and constructs penalty and Lagrangian functions.
"""
function build_optimization_problem(nlp::AbstractNLPModel)
    name = nlp.meta.name
    nvar = nlp.meta.nvar
    ncon = nlp.meta.ncon
    x0 = nlp.meta.x0
    y0 = nlp.meta.y0
    xl = nlp.meta.lvar
    xu = nlp.meta.uvar

    # Adjust objective sign based on optimization direction
    sign_obj = nlp.meta.minimize ? 1.0 : -1.0

    f(x) = sign_obj * obj(nlp, x)
    ∇f(x) = sign_obj * grad(nlp, x)
    
    h(x) = cons(nlp, x)
    ∇h(x) = jac(nlp, x)
    
    ∇²h(x) = [hess_constraint(nlp, x, i) for i in 1:ncon]

    # Constraint penalty functions
    c(x) = 0.5 * sum(hx^2 for hx in h(x))
    ∇c(x) = ∇h(x)' * h(x)

    function ∇²c(x)
        J = ∇h(x)
        Hh_list = ∇²h(x)
        H = J' * J
        for (i, hi) in enumerate(h(x))
            H += hi * Hh_list[i]
        end
        return H
    end

    project(x) = clamp.(x, xl, xu)
    prox_grad_c(x) = project(x - ∇c(x)) - x
    Lagrangian(x, λ) = f(x) + dot(h(x), λ)
    
    merit(x, λ, θ) = θ * Lagrangian(x, λ) + (1 - θ) * norm(h(x))

    # Incorporate the objective weight modifier directly into the Lagrangian Hessian evaluation
    ∇²L(x, λ) = hess(nlp, x, λ; obj_weight=sign_obj)

    return OptimizationProblem(
        name, nvar, ncon, 
        f, ∇f, h, ∇h, ∇²h, c, ∇c, ∇²c, project, prox_grad_c, Lagrangian, merit, ∇²L, 
        x0, y0, xl, xu
    )
end

# ==============================================================================
# FAST BENCHMARK TEST SET (SMALL_PROBLEMS)
# ==============================================================================
"""
Vetor de problemas pequenos e representativos do CUTEst para testes e validações rápidas.
Contém problemas com restrições (m = 1) e sem restrições/cotas (m = 0).
"""
const SMALL_PROBLEMS = [
    # Com restrições (m = 1)
    "HS6",       # Hock-Schittkowski 6
    "TRY-B",     # Curvatura pronunciada
    "HS28",      # Objetivo quadrático, restrição linear
    "HS26",      # Cúbico com restrição quadrática
    "HS35MOD",   # Quadrático padrão
    "EXTRASIM",  # Sanidade de viabilidade
    "TAME",      # Simples e rápido

    # Sem restrições ou apenas com cotas (m = 0)
    "ROSENBR",   # Vale da Banana
    "HS3",       # Quadrático com cotas
    "HS4",       # Cúbico com cotas
    "HS45",      # n=5 com cotas
    "SISSER",    # Vale estreito
]

# ==============================================================================
# SHARED OPTIMIZATION UTILITIES (Aspect Ratio, Parabolic Heuristic, GP Norm)
# ==============================================================================

"""
Ensures the trust region maintains a healthy minimum aspect ratio.
If a dimension falls below the threshold relative to the maximum, it is inflated 
to prevent stagnation in diagonal valleys.
"""
function enforce_aspect_ratio!(δ::Vector{Float64}, min_ratio::Float64=1e-3)
    max_delta = maximum(δ)
    threshold = max_delta * min_ratio
    changed = false
    for i in 1:length(δ)
        if δ[i] < threshold
            δ[i] = threshold 
            changed = true
        end
    end
    return changed
end

@inline function _merit_parabolic_terms(
    f_current::Float64,
    f_new::Float64,
    slope::Float64,
    ared_val::Float64,
    pred_val::Float64,
)
    if isfinite(ared_val) && isfinite(pred_val) && pred_val > 1e-15
        return pred_val, (pred_val - ared_val)
    else
        return max(-slope, 0.0), (f_new - f_current - slope)
    end
end

function _atr_grad_sign_step!(
    δ_current::Vector{Float64},
    s_vec::Vector{Float64},
    g_p::Vector{Float64},
    g_c::Vector{Float64},
    params,
    action::Symbol,
    max_increase_ratio::Float64,
)
    n = length(δ_current)
    δ_new = similar(δ_current)

    if action == :increase
        has_grads = length(g_p) == n && length(g_c) == n
        @inbounds for i in 1:n
            mult_i = 1.0
            if has_grads
                prod_i = g_p[i] * g_c[i]
                u_i = abs(s_vec[i]) / max(δ_current[i], 1e-30)
                if prod_i < -1e-16
                    mult_i = 0.5
                elseif prod_i > 1e-16 && u_i >= 0.5
                    mult_i = min(1.5, max_increase_ratio)
                end
            end
            δ_new[i] = clamp(δ_current[i] * mult_i, params.δmin, params.δmax)
        end
        enforce_aspect_ratio!(δ_new, 1.0 / params.atr_max_cond)
    else
        u_max = 0.0
        @inbounds for i in 1:n
            u_i = abs(s_vec[i]) / max(δ_current[i], 1e-30)
            if u_i > u_max
                u_max = u_i
            end
        end
        if u_max <= 1e-15
            @inbounds for i in 1:n
                δ_new[i] = clamp(0.5 * δ_current[i], params.δmin, params.δmax)
            end
        else
            @inbounds for i in 1:n
                si_abs = abs(s_vec[i])
                u_hat = (si_abs / max(δ_current[i], 1e-30)) / u_max
                d_cut = si_abs > 0.0 ? min(0.5 * δ_current[i], 0.75 * si_abs) : 0.5 * δ_current[i]
                δ_new[i] = clamp((1.0 - u_hat) * δ_current[i] + u_hat * d_cut, params.δmin, params.δmax)
            end
        end
    end
    return δ_new
end

function _atr_decomposed_cut_step!(
    δ_current::Vector{Float64},
    f_current::Float64,
    f_new::Float64,
    slope::Float64,
    ared_val::Float64,
    pred_val::Float64,
    s_vec::Vector{Float64},
    g_p::Vector{Float64},
    g_c::Vector{Float64},
    params,
    action::Symbol,
    min_reduction_ratio::Float64,
    max_reduction_ratio::Float64,
    min_increase_ratio::Float64,
    max_increase_ratio::Float64,
)
    n = length(δ_current)
    δ_new = similar(δ_current)

    if action == :decrease
        pred, R = _merit_parabolic_terms(f_current, f_new, slope, ared_val, pred_val)
        τ_parab = (R > params.ratio_safeguard_tol && pred > 0.0) ?
                  clamp(pred / (2.0 * R), min_reduction_ratio, max_reduction_ratio) :
                  max_reduction_ratio

        u_max_sq = 0.0
        @inbounds for i in 1:n
            u_sq = (s_vec[i] / max(δ_current[i], 1e-30))^2
            if u_sq > u_max_sq
                u_max_sq = u_sq
            end
        end

        if u_max_sq <= 1e-30
            @inbounds for i in 1:n
                δ_new[i] = clamp(τ_parab * δ_current[i], params.δmin, params.δmax)
            end
        else
            @inbounds for i in 1:n
                si_abs = abs(s_vec[i])
                ω_i = ((si_abs / max(δ_current[i], 1e-30))^2) / u_max_sq
                d_cut = si_abs > 0.0 ? min(τ_parab * δ_current[i], 0.75 * si_abs) : τ_parab * δ_current[i]
                δ_new[i] = clamp((1.0 - ω_i) * δ_current[i] + ω_i * d_cut, params.δmin, params.δmax)
            end
        end
    else
        has_grads = length(g_p) == n && length(g_c) == n
        ρ_glob = (isfinite(ared_val) && isfinite(pred_val) && pred_val > 1e-15) ? (ared_val / pred_val) : 1.0

        @inbounds for i in 1:n
            u_i = clamp(abs(s_vec[i]) / max(δ_current[i], 1e-30), 0.0, 1.0)
            mult_i = 1.0
            if has_grads
                prod_i = g_p[i] * g_c[i]
                b_i = g_p[i] * s_vec[i]
                if prod_i < -1e-16
                    α_star = abs(g_p[i]) / (abs(g_p[i]) + abs(g_c[i]))
                    mult_i = 1.0 - u_i * (1.0 - clamp(α_star, 0.5, 0.95))
                elseif b_i < -1e-16
                    c_i = (g_c[i] - g_p[i]) * s_vec[i]
                    raw_inc = (c_i > 1e-16) ? clamp(-b_i / c_i, min_increase_ratio, max_increase_ratio) : max_increase_ratio
                    mult_i = 1.0 + u_i * (raw_inc - 1.0)
                elseif u_i >= 0.8 && ρ_glob >= 0.5
                    mult_i = 1.0 + u_i * (min(1.5, max_increase_ratio) - 1.0)
                end
            end
            δ_new[i] = clamp(δ_current[i] * mult_i, params.δmin, params.δmax)
        end
        enforce_aspect_ratio!(δ_new, 1.0 / params.atr_max_cond)
    end
    return δ_new
end

function _atr_secant_box_step!(
    δ_current::Vector{Float64},
    f_current::Float64,
    f_new::Float64,
    slope::Float64,
    ared_val::Float64,
    pred_val::Float64,
    s_vec::Vector{Float64},
    g_p::Vector{Float64},
    g_c::Vector{Float64},
    jac_diff_col_l1::Vector{Float64},
    theta_merit::Float64,
    d_curv::Vector{Float64},
    params,
    action::Symbol,
    min_reduction_ratio::Float64,
    max_reduction_ratio::Float64,
    min_increase_ratio::Float64,
    max_increase_ratio::Float64,
)
    n = length(δ_current)
    δ_new = similar(δ_current)

    log_Δ = 0.0
    @inbounds for i in 1:n
        log_Δ += log(max(δ_current[i], 1e-30))
    end
    Δ_k = exp(log_Δ / n)

    if action == :decrease
        pred, R = _merit_parabolic_terms(f_current, f_new, slope, ared_val, pred_val)
        τ_parab = (R > params.ratio_safeguard_tol && pred > 0.0) ?
                  clamp(pred / (2.0 * R), min_reduction_ratio, max_reduction_ratio) :
                  max_reduction_ratio

        S_W = 0.0
        @inbounds for i in 1:n
            w_i = δ_current[i] / max(Δ_k, 1e-30)
            sw_i = abs(s_vec[i]) / max(w_i, 1e-12)
            if sw_i > S_W
                S_W = sw_i
            end
        end
        Δ_new = S_W > 0.0 ? min(τ_parab * Δ_k, 0.75 * S_W) : τ_parab * Δ_k
        @inbounds for i in 1:n
            w_i = δ_current[i] / max(Δ_k, 1e-30)
            δ_new[i] = clamp(Δ_new * w_i, params.δmin, params.δmax)
        end
    else
        u_max = 0.0
        @inbounds for i in 1:n
            u_i = abs(s_vec[i]) / max(δ_current[i], 1e-30)
            if u_i > u_max
                u_max = u_i
            end
        end

        if u_max < 0.5
            α_Δ = 1.0
        else
            pred, R = _merit_parabolic_terms(f_current, f_new, slope, ared_val, pred_val)
            α_Δ = (R <= params.ratio_safeguard_tol || pred <= 0.0) ?
                  max_increase_ratio :
                  clamp(pred / (2.0 * R), min_increase_ratio, max_increase_ratio)
        end
        Δ_new = clamp(α_Δ * Δ_k, params.δmin, params.δmax)

        if length(d_curv) == n && length(g_p) == n && length(g_c) == n
            β = params.secant_ema_beta
            s_thresh = 1e-10 * max(1.0, Δ_k)
            has_jac_diff = length(jac_diff_col_l1) == n
            θ_m = clamp(theta_merit, 0.0, 1.0)

            @inbounds for i in 1:n
                si_abs = abs(s_vec[i])
                if si_abs > s_thresh
                    dy_i = θ_m * abs(g_c[i] - g_p[i]) + (has_jac_diff ? (1.0 - θ_m) * jac_diff_col_l1[i] : 0.0)
                    if dy_i > 1e-16
                        d_hat = clamp(dy_i / si_abs, 1e-4, 1e6)
                        d_curv[i] = β * d_curv[i] + (1.0 - β) * d_hat
                    end
                end
            end

            half_log_κ = 0.5 * log(max(params.atr_max_cond, 1.01))
            log_w = Vector{Float64}(undef, n)
            sum_lw = 0.0
            @inbounds for i in 1:n
                lw = -0.5 * log(max(d_curv[i], 1e-12))
                log_w[i] = lw
                sum_lw += lw
            end
            mean_lw = sum_lw / n

            sum_lw2 = 0.0
            @inbounds for i in 1:n
                lw_c = clamp(log_w[i] - mean_lw, -half_log_κ, half_log_κ)
                log_w[i] = lw_c
                sum_lw2 += lw_c
            end
            mean_lw2 = sum_lw2 / n

            @inbounds for i in 1:n
                w_i = exp(log_w[i] - mean_lw2)
                δ_new[i] = clamp(Δ_new * w_i, params.δmin, params.δmax)
            end
        else
            @inbounds for i in 1:n
                w_i = δ_current[i] / max(Δ_k, 1e-30)
                δ_new[i] = clamp(Δ_new * w_i, params.δmin, params.δmax)
            end
        end
    end
    return δ_new
end

"""
Calculates trust region radius using quadratic interpolation models (Parabolic Heuristic)
or alternative anisotropic trust-region strategies (`params.atr_strategy`).
"""
function parabolic_heuristic_step!(
    δ_current::T,
    f_current::Float64,
    f_new::Float64,
    slope::Float64,
    params,
    s_norm::Float64,
    step_norm_sq::Float64,
    action::Symbol=:decrease;

    anisotropic::Bool=false,
    s_vec::Vector{Float64}=Float64[],
    grad::Vector{Float64}=Float64[],
    grad_old::Vector{Float64}=Float64[],
    mode::Symbol=:shrink, 

    min_reduction_ratio::Float64=0.1,
    max_reduction_ratio::Float64=0.5,
    min_increase_ratio::Float64=1.0,
    max_increase_ratio::Float64=2.0,

    ared_val::Float64=NaN,
    pred_val::Float64=NaN,
    g_prev::Vector{Float64}=Float64[],
    g_curr::Vector{Float64}=Float64[],
    jac_diff_col_l1::Vector{Float64}=Float64[],
    theta_merit::Float64=1.0,
    d_curv::Vector{Float64}=Float64[],
) where {T<:Union{Float64, Vector{Float64}}}
    # Isotropic behavior execution block
    if !anisotropic || !(δ_current isa Vector)
        δ_scalar = max_δ(δ_current)
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
    end

    strat = hasproperty(params, :atr_strategy) ? params.atr_strategy : :legacy_parabolic
    if strat != :legacy_parabolic
        g_p = !isempty(g_prev) ? g_prev : (!isempty(grad_old) ? grad_old : grad)
        g_c = !isempty(g_curr) ? g_curr : grad
        if strat == :grad_sign
            return _atr_grad_sign_step!(δ_current, s_vec, g_p, g_c, params, action, max_increase_ratio)
        elseif strat == :decomposed_cut
            return _atr_decomposed_cut_step!(
                δ_current, f_current, f_new, slope, ared_val, pred_val,
                s_vec, g_p, g_c, params, action,
                min_reduction_ratio, max_reduction_ratio, min_increase_ratio, max_increase_ratio
            )
        elseif strat == :secant_box
            return _atr_secant_box_step!(
                δ_current, f_current, f_new, slope, ared_val, pred_val,
                s_vec, g_p, g_c, jac_diff_col_l1, theta_merit, d_curv, params, action,
                min_reduction_ratio, max_reduction_ratio, min_increase_ratio, max_increase_ratio
            )
        else
            error("Unsupported atr_strategy: $strat. Valid options: :legacy_parabolic, :grad_sign, :decomposed_cut, :secant_box")
        end
    end

    # Legacy Parabolic Anisotropic calculation setup
    n = length(δ_current)
    δ_new = copy(δ_current)
    multipliers = zeros(n)
    # Phase A: Computation of directional multipliers
    for i in 1:n
        b_i = grad[i] * s_vec[i]
        a_i = f_new - f_current - b_i
        if a_i < params.ratio_safeguard_tol
            multipliers[i] = (action == :increase) ? max_increase_ratio : 1.0
        else
            multipliers[i] = -b_i / (2.0 * a_i)
        end
    end
    # Phase B: Strategy application
    has_history = !isempty(grad_old) && length(grad_old) == n
    if action == :decrease
        if mode == :reshape
            val_max, idx_max = findmax(multipliers)
            for i in 1:n
                if i == idx_max
                    multipliers[i] = 1.0 
                else
                    multipliers[i] = clamp(multipliers[i], min_reduction_ratio, max_reduction_ratio)
                end
            end
            if params.verbose_out
                @printf("   [Reshape] Scale altered: Dim %d maintained. Others reduced.\n", idx_max)
            end
        elseif mode == :shrink
            b_global = slope
            a_global = f_new - f_current - slope
            if a_global < params.ratio_safeguard_tol
                uniform_ratio = max_reduction_ratio
            else
                theo = -b_global / (2.0 * a_global)
                uniform_ratio = clamp(theo, min_reduction_ratio, max_reduction_ratio)
            end
            fill!(multipliers, uniform_ratio)
            if params.verbose_out
                @printf("   [Uniform Shrink] Scale maintained: Global multiplier at %.2f\n", uniform_ratio)
            end
        end
    else # :increase
        for i in 1:n
            multipliers[i] = clamp(multipliers[i], min_increase_ratio, max_increase_ratio)
        end
    end
    # Phase C: State update application
    δ_new .= clamp.(δ_current .* multipliers, params.δmin, params.δmax)
    if params.verbose_out
        vec_str = string("[", join([@sprintf("%.2f", x) for x in multipliers], ", "), "]")
        @printf("   [Aniso %s] Multipliers: %s\n", string(action), vec_str)
    end
    return δ_new
end

"""
Computes projected gradient norm (AGP / GP-norm) with box and equality constraints.
"""
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
            p_old = similar(p_dyk)
            q = zeros(n)
            r = zeros(n)
            y_val = zeros(n)
            
            local fact_J
            local y_plus_r, res, lambda_proj, jac_t_lambda
            if m > 0
                J_Jt = jac_h * jac_h'
                fact_J = try
                    lu(J_Jt + 1e-10 * I)
                catch
                    try
                        lu(Matrix(J_Jt) + 1e-6 * I)
                    catch
                        nothing
                    end
                end
                if isnothing(fact_J)
                    # Fallback suave para projeção em caixa quando J*J' for singular/deficiente em posto (ex: m > n)
                    copyto!(out_buffer, clamp.(v, xl, xu) .- x)
                    return
                end
                y_plus_r = zeros(n)
                res = zeros(m)
                lambda_proj = zeros(m)
                jac_t_lambda = zeros(n)
            end
            
            @inbounds for k in 1:max_iter
                copyto!(p_old, p_dyk)
                
                # Projeção 1: Limites da caixa (Box)
                for i in 1:n
                    val = p_dyk[i] + q[i]
                    y_val_i = clamp(val, xl[i], xu[i])
                    y_val[i] = y_val_i
                    q[i] = val - y_val_i
                end
                
                # Projeção 2: Hiperplanos
                if m > 0
                    for i in 1:n
                        y_plus_r[i] = y_val[i] + r[i]
                    end
                    mul!(res, jac_h, y_plus_r)
                    for i in 1:m
                        res[i] -= b_target[i]
                    end
                    ldiv!(lambda_proj, fact_J, res)
                    mul!(jac_t_lambda, jac_h', lambda_proj)
                    
                    for i in 1:n
                        p_dyk_i = y_plus_r[i] - jac_t_lambda[i]
                        p_dyk[i] = p_dyk_i
                        r[i] = y_plus_r[i] - p_dyk_i
                    end
                else
                    copyto!(p_dyk, y_val)
                end
                
                max_diff = 0.0
                for i in 1:n
                    diff_i = abs(p_dyk[i] - p_old[i])
                    if diff_i > max_diff
                        max_diff = diff_i
                    end
                end
                if max_diff < tol
                    break
                end
            end
            
            @inbounds for i in 1:n
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

end