module SharedTypes

export StepLog, StepLog_SLP, StepLog_TwoPhase, OptimizationProblem, OptimizedBuffers, TwoPhaseParams, SLPParams, create_rosenbrock_problem0, create_rosenbrock_problem1, build_optimization_problem, GRB_ENV

using ADNLPModels, NLPModels, Gurobi

# Global environment for Gurobi to prevent license checkout overhead on every iteration
const GRB_ENV = Gurobi.Env(output_flag=0)

# ==============================================================================
# LOGGING STRUCTURES
# ==============================================================================

"""
Abstract base type for step iteration logs.
"""
abstract type AbstractStepLog end

"""
Logging structure specific to the SLP (Sequential Linear Programming) algorithm.
Records the states and metrics for a single optimization step.
"""
Base.@kwdef struct StepLog_SLP <: AbstractStepLog
    iter::Int
    x_from::Vector{Float64}
    x_to::Vector{Float64}
    status::Symbol
    phase::Symbol
    delta::Union{Float64, Vector{Float64}}
    # SLP-specific metrics
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

    function OptimizedBuffers(n::Int, m::Int)
        new(zeros(n), zeros(n), zeros(n), 0.0, 0.0, zeros(n), zeros(n), 
            zeros(n), zeros(n), zeros(m), zeros(m, n), zeros(m), zeros(n), zeros(n))
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
    max_outer_iter::Int = 250       # Maximum overall algorithm iterations
    stag_tol::Float64 = 1e-5        # Step size threshold to trigger stagnation counter
    max_stag::Int = 3               # Maximum consecutive stagnated iterations allowed
    non_monotone_M::Int = 5         # History length for non-monotone acceptance criteria

    # Trust Region Update Strategies
    aredpred_ratio::Bool = true
    backtracking_quadratic::Bool = true
    anisotropic_trust_region::Bool = true
    use_ratio_update::Bool = true
    ratio_safeguard_tol::Float64 = 1e-12
    ratio_max_factor::Float64 = 2.0
    ratio_min_factor::Float64 = 0.1
    parabolic_min_reduction_ratio::Float64 = 0.1
    parabolic_max_reduction_ratio::Float64 = 0.5
    parabolic_min_increase_ratio::Float64 = 1.0
    parabolic_max_increase_ratio::Float64 = 2.0

    # Sequential Quadratic Programming (SQP) Configuration
    use_quadratic::Bool = false     # Flag to enable/disable quadratic model usage
    quadratic_solver::Symbol = :ripqp # Selected solver for QP subproblems
    σ::Float64 = 0.1                # Matrix regularization term (H + σI)
    B_update_strategy::Symbol = :identity # Strategy for Hessian approximation (:identity, :spectral, :exact)

    # Standard SLP Stopping Criteria (KKT and step-based)
    maxcount::Int = 3
    tolG::Float64 = 1e-3
    tolF::Float64 = 5e-2
    tolS::Float64 = 1e-4
    use_slp_stopping::Bool = true
    
    # Verbosity and Debugging Options
    verbose_in::Bool = false
    verbose_out::Bool = true
    scale::Bool = false
    norm_gpnorm::Union{Int, Float64} = 2
    gpnorm_div_nelem::Bool = true
    debugverbose::Bool = false
    log_to_file::Bool = false
end

"""
Configuration parameters specifically for the standard SLP (Sequential Linear Programming) algorithm.
"""
Base.@kwdef mutable struct SLPParams
    maxiter::Int = 500              # Maximum total iterations
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

    # Backtracking and Anisotropy Settings
    backtracking_quadratic::Bool = true
    anisotropic_trust_region::Bool = true
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
    aredpred_ratio::Bool = true

    # Sequential Quadratic Programming (SQP) Configuration
    use_quadratic::Bool = false
    quadratic_solver::Symbol = :ripqp
    σ::Float64 = 0.1                  # Matrix regularization term (H + σI)
    B_update_strategy::Symbol = :identity 
    
    verbose_out::Bool = false       # Specific verbosity flag for backtracking operations
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

end