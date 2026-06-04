module SharedTypes

export StepLog, StepLog_SLP, StepLog_TwoPhase, OptimizationProblem, OptimizedBuffers, TwoPhaseParams, SLPParams, create_rosenbrock_problem0, create_rosenbrock_problem1, build_optimization_problem, GRB_ENV

using ADNLPModels, NLPModels, Gurobi

const GRB_ENV = Gurobi.Env(output_flag=0)

# const GRB_ENV = Gurobi.Env(OutputFlag=0)
# 1. Definir um tipo pai (abstrato)
abstract type AbstractStepLog end

# 2. Criar as variantes específicas
Base.@kwdef struct StepLog_SLP <: AbstractStepLog
    iter::Int
    x_from::Vector{Float64}
    x_to::Vector{Float64}
    status::Symbol
    phase::Symbol
    delta::Union{Float64, Vector{Float64}}
    # Específicos do SLP
    ared::Float64 = 0.0
    pred::Float64 = 0.0
end

Base.@kwdef struct StepLog_TwoPhase <: AbstractStepLog
    iter::Int
    x_from::Vector{Float64}
    x_to::Vector{Float64}
    status::Symbol
    phase::Symbol
    delta::Union{Float64, Vector{Float64}}
    # Específicos do TwoPhase
    Lagrange::Float64
    Merit::Float64
end

"""
Estrutura principal do problema otimizada e unificada
"""
struct OptimizationProblem{F, G, H, J, Hc, C, GC, HC, P, PG, L, M, HL}
    name::String           # Nome do problema
    nvar::Int              # Número de variáveis
    ncon::Int              # Número de restrições
    f::F                   # Objective function
    ∇f::G                  # Gradient
    h::H                   # Constraints
    ∇h::J                  # Jacobian
    ∇²h::Hc                # Hessian of constraints
    c::C                   # Penalty function
    ∇c::GC                 # Penalty gradient
    ∇²c::HC                # Penalty Hessian
    project::P             # Projection operator
    prox_grad_c::PG        # Proximal gradient
    Lagrangian::L          # Lagrangian function
    merit::M               # Merit function
    ∇²L::HL                # Hessian of Lagrangian
    x0::Vector{Float64}    # Chute inicial primal
    y0::Vector{Float64}    # Chute inicial dual (multiplicadores)
    xl::Vector{Float64}    # Lower bounds
    xu::Vector{Float64}    # Upper bounds
end



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
# 2. PARÂMETROS DOS ALGORITMOS
# ==============================================================================

Base.@kwdef mutable struct TwoPhaseParams
    # Restoration phase parameters
    r_resto::Float64 = 0.9
    rfeas::Float64 = 1e-12
    βc::Float64 = 0.5
    αR::Float64 = 1e-4
    τ1::Float64 = 0.1
    τ2::Float64 = 0.25
    τ3::Float64 = 0.5
    θ0::Float64 = 0.9
    tol::Float64 = 1e-4

    # Optimization phase parameters
    αΦ::Float64 = (1 - 0.9)/2
    αL::Float64 = 1e-6

    # Trust region parameters
    δ0_opt::Float64 = 0.1
    δ0_resto::Float64 = 0.5
    δmin::Float64 = 1e-6
    δmax::Float64 = 1.0

    # Iteration limits
    max_iter_resto::Int = 50
    max_iter_opt::Int = 50
    max_outer_iter::Int = 250
    stag_tol::Float64 = 1e-5
    max_stag::Int = 3
    non_monotone_M::Int = 5

    # Trust region update strategy
    use_ratio_update::Bool = true
    ratio_safeguard_tol::Float64 = 1e-12
    ratio_max_factor::Float64 = 2.0
    ratio_min_factor::Float64 = 0.1

    # SLP stopping criteria
    maxcount::Int = 3
    tolG::Float64 = 1e-3
    tolF::Float64 = 5e-2
    tolS::Float64 = 1e-4
    use_slp_stopping::Bool = true
    # Control and debug options
    verbose_in::Bool = false
    verbose_out::Bool = true
    scale::Bool = false
    norm_gpnorm::Union{Int, Float64} = 2
    gpnorm_div_nelem::Bool = true
    backtracking_quadratic::Bool = true
    anisotropic_trust_region::Bool = true
    debugverbose::Bool = false
    log_to_file::Bool = false
end


Base.@kwdef mutable struct SLPParams
    maxiter::Int = 500
    tolG::Float64 = 1e-3      # Tolerância gradiente projetado
    tolF::Float64 = 5e-2      # Tolerância redução função
    tolS::Float64 = 1e-3      # Tolerância passo
    maxcount::Int = 3         # Contador de consistência para parada

    # Parâmetros Trust Region (baseado no StrSLP)
    eta::Float64 = 0.1        # Limite aceitação passo
    rho::Float64 = 0.5        # Limite expansão passo
    delta0::Float64 = 0.1     # Raio inicial (Ajustado para 1.0 para problemas gerais)
    alphaR::Float64 = 0.25     # Fator redução delta
    alphaA::Float64 = 2.0     # Fator aumento delta

    verbose::Bool = true
    debugverbose::Bool = false
    output_flag::Int = 0      # Gurobi output flag

    # --- NOVOS PARÂMETROS PARA BACKTRACKING E ANISOTROPIA ---
    backtracking_quadratic::Bool = true
    anisotropic_trust_region::Bool = true
    
    # Parâmetros internos usados pelo quadratic_backtracking_step!
    δmin::Float64 = 1e-6
    δmax::Float64 = 1
    ratio_safeguard_tol::Float64 = 1e-8
    τ1::Float64 = 0.25
    τ2::Float64 = 0.5
    
    verbose_out::Bool = false # Verbose específico do backtracking
    # --------------------------------------------------------
end

# ==============================================================================
# PROBLEM DEFINITIONS
# ==============================================================================

function create_rosenbrock_problem0()
    # A função objetivo permanece a mesma
    objective(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

    # Ponto inicial (mantendo igual ao problema 1)
    x0 = [10.0, 10.0]
    # x0 = [-1.2, 1.0]

    # Limites das variáveis (Box constraints)
    # Mesmo sendo "sem restrições" funcionais, geralmente mantemos
    # os limites do domínio definidos no problema original (-10 a 10).
    xl = fill(-10.0, 2)
    xu = fill(10.0, 2)

    # Retorna o modelo sem passar constraints, lcon ou ucon.
    # O ADNLPModel entende automaticamente que é um problema irrestrito (apenas com bounds).
    return ADNLPModel(objective, x0, xl, xu; name="Rosenbrock_Problem0_Unconstrained")
end
"""
Creates Rosenbrock 2D problem with circular constraint
min f(x) = (1-x₁)² + 100(x₂-x₁²)²
s.t. h₁(x) = x₁² + x₂² - 6 = 0
"""
function create_rosenbrock_problem1()
    objective(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

    constraints(x) = [
        x[1]^2 + x[2]^2 - 6,   # Circular constraint
    ]

    # Ponto inicial (mantendo igual ao problema 1)
    x0 = [10.0, 10.0]

    xl = fill(-10.0, 2)
    xu = fill(10.0, 2)
    lcon = [0.0]
    ucon = [0.0]

    return ADNLPModel(objective, x0, xl, xu, constraints, lcon, ucon; name="Rosenbrock_Problem1_Circular")
end

"""
Builds optimization problem structure from NLP model
"""
function build_optimization_problem(nlp::AbstractNLPModel)
    name = nlp.meta.name
    nvar = nlp.meta.nvar
    ncon = nlp.meta.ncon
    x0 = nlp.meta.x0
    y0 = nlp.meta.y0
    xl = nlp.meta.lvar
    xu = nlp.meta.uvar

    # Verifica se é minimização. Se não for, inverte o sinal da função objetivo
    sign_obj = nlp.meta.minimize ? 1.0 : -1.0

    f(x) = sign_obj * obj(nlp, x)
    ∇f(x) = sign_obj * grad(nlp, x)
    
    h(x) = cons(nlp, x)
    ∇h(x) = jac(nlp, x)
    
    # Usando ncon em vez de chamar length(h(x))
    ∇²h(x) = [hess_constraint(nlp, x, i) for i in 1:ncon]

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

    # A Hessiana da Lagrangiana deve considerar a inversão do objetivo.
    # Usamos o keyword argument obj_weight do NLPModels.
    ∇²L(x, λ) = hess(nlp, x, λ; obj_weight=sign_obj)

    return OptimizationProblem(
        name, nvar, ncon, 
        f, ∇f, h, ∇h, ∇²h, c, ∇c, ∇²c, project, prox_grad_c, Lagrangian, merit, ∇²L, 
        x0, y0, xl, xu
    )
end



end