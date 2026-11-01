module Generic_module_ComputeB

export compute_B

using SparseArrays
using LinearAlgebra
using ..SharedTypes

"""
    compute_B(strategy::Symbol, n::Int, s::Vector{Float64}, y_grad::Vector{Float64}, 
              σ_fallback::Float64, problem::OptimizationProblem, x_new::Vector{Float64}, 
              λ::Vector{Float64}, B_old::Union{AbstractMatrix{Float64}, Nothing} = nothing)

Calcula ou atualiza a aproximação da matriz Hessiana B para subproblemas quadráticos (SQP).
Implementa salvaguardas rigorosas de positividade, limites numéricos (clamps) e
amortecimento de Powell para estabilidade incondicional em otimizadores quadráticos (Gurobi/RipQP).
"""
function compute_B(strategy::Symbol, n::Int, s::Vector{Float64}, y_grad::Vector{Float64}, σ_fallback::Float64, problem::OptimizationProblem, x_new::Vector{Float64}, λ::Vector{Float64}, B_old::Union{AbstractMatrix{Float64}, Nothing} = nothing)
    # Avaliação segura e eficiente do gradiente do Lagrangeano: ∇L(x, λ) = ∇f(x) + ∇h(x)' * λ
    get_grad_L(x, mul_λ) = (problem.ncon > 0 && length(mul_λ) > 0) ? (problem.∇f(x) + problem.∇h(x)' * mul_λ) : problem.∇f(x)
    x_prev = x_new - s

    if strategy == :exact
        B_exact = problem.∇²L(x_new, λ)
        # Salvaguarda de semi-definida positiva (PSD) para problemas não-convexos:
        # Se B for indefinida, Gurobi e RipQP podem falhar ou rejeitar a matriz.
        if n <= 500
            # Para dimensões tratáveis, verifica o menor autovalor
            B_mat = Matrix{Float64}(B_exact)
            λ_min = minimum(real.(eigvals(Symmetric(B_mat))))
            if λ_min < 1e-5
                shift = abs(λ_min) + 1e-4
                return issparse(B_exact) ? (B_exact + shift * I) : (B_mat + shift * I)
            end
        else
            # Para n grande, salvaguarda diagonal mínima
            diag_min = minimum(diag(B_exact))
            if diag_min < 1e-5
                shift = abs(diag_min) + 1e-4
                return B_exact + shift * I
            end
        end
        return B_exact

    elseif strategy == :diag_exact
        B_exact = problem.∇²L(x_new, λ)
        d = Vector{Float64}(diag(B_exact))
        
        # Salvaguardas rigorosas contra não-convexidade e instabilidades numéricas
        @inbounds for i in 1:n
            val = d[i]
            if !isfinite(val) || val < 1e-5
                d[i] = 1e-5
            elseif val > 1e8
                d[i] = 1e8
            end
        end
        return spdiagm(0 => d)

    elseif strategy == :identity
        σ = σ_fallback > 0.0 ? σ_fallback : 1.0
        return σ * spdiagm(0 => ones(n))
        
    elseif strategy == :spectral
        if norm(s) < 1e-8 || norm(y_grad) < 1e-8
            return σ_fallback * spdiagm(0 => ones(n))
        end
        sTy = dot(s, y_grad)
        yTy = dot(y_grad, y_grad)
        if sTy <= 1e-12 || !isfinite(sTy) || !isfinite(yTy)
            return σ_fallback * spdiagm(0 => ones(n))
        end
        alpha = yTy / sTy
        alpha = clamp(alpha, 1e-4, 1e4)
        return alpha * spdiagm(0 => ones(n))
        
    elseif strategy == :reciprocal
        grad = get_grad_L(x_new, λ)
        b = zeros(n)
        for i in 1:n
            # Proteção contra divisão por zero
            if abs(x_new[i]) < 1e-10
                d2f_R = 0.0 
            else
                d2f_R = -2.0 / x_new[i] * grad[i]
            end
            
            # Regra de salvaguarda de positividade (Eq. 5.91)
            if d2f_R >= 0.0
                b[i] = d2f_R
            else
                b[i] = d2f_R + 1.1 * abs(d2f_R) # 0.1 * |d2f_R|
            end
            
            # Salvaguarda dupla: evita singularidade (limite inferior) e 
            # evita valores infinitos ou excessivos para otimizadores quadráticos (limite superior)
            if !isfinite(b[i]) || b[i] > 1e8
                b[i] = 1e8
            elseif b[i] < 1e-5
                b[i] = 1e-5
            end
        end
        return spdiagm(0 => b)
        
    elseif strategy == :exponential
        grad = get_grad_L(x_new, λ)
        # Evita re-avaliar ∇f(x_prev) e ∇h(x_prev) redundantes se y_grad já foi fornecido
        grad_old = (norm(y_grad) > 1e-14) ? (grad - y_grad) : get_grad_L(x_prev, λ)
        
        b = zeros(n)
        for i in 1:n
            x_curr = x_new[i]
            x_prev_i = x_prev[i]
            g_curr = grad[i]
            g_prev = grad_old[i]
            
            # Critério protegido para expoente a_i adaptativo
            # Protege contra singularidade quando x_curr ≈ x_prev (log(ratio_x) ≈ 0)
            dx_rel = abs(s[i]) / max(abs(x_curr), 1.0)
            if dx_rel < 1e-5 || (g_prev * g_curr) <= 1e-6
                ai = -1.0  # Cai de volta na aproximação recíproca clássica
            else
                ratio_g = g_prev / g_curr
                ratio_x = x_prev_i / x_curr
                if ratio_g > 0.0 && ratio_x > 0.0
                    denom = log(ratio_x)
                    if abs(denom) > 1e-4
                        ai = 1.0 + log(ratio_g) / denom
                        ai = clamp(ai, -10.0, 10.0) # Salvaguarda contra expoentes hiperbólicos anômalos
                    else
                        ai = -1.0
                    end
                else
                    ai = -1.0
                end
            end
            
            # Segunda derivada da aproximação exponencial com proteção
            if abs(x_curr) < 1e-10
                d2f_E = 0.0
            else
                d2f_E = (ai - 1.0) / x_curr * g_curr
            end
            
            # Regra de salvaguarda de positividade (Eq. 5.100)
            if d2f_E >= 0.0
                b[i] = d2f_E
            else
                b[i] = d2f_E + 1.1 * abs(d2f_E)
            end
            
            # Salvaguarda dupla completa (inferior e superior)
            if !isfinite(b[i]) || b[i] > 1e8
                b[i] = 1e8
            elseif b[i] < 1e-5
                b[i] = 1e-5
            end
        end
        return spdiagm(0 => b)
        
    elseif strategy == :quasi_newton
        # Avaliação lazy do gradiente do Lagrangeano caso algum passo s[i] seja minúsculo
        local grad_lazy = nothing
        b = zeros(n)
        for i in 1:n
            vi = s[i]       # variação de passo s_i
            wi = y_grad[i]  # variação de gradiente y_i
            
            if abs(vi) < 1e-10
                # Fallback protegido contra divisão por zero
                if grad_lazy === nothing
                    grad_lazy = get_grad_L(x_new, λ)
                end
                if abs(x_new[i]) < 1e-10
                    d2f_R = 0.0
                else
                    d2f_R = -2.0 / x_new[i] * grad_lazy[i]
                end
                
                b[i] = d2f_R >= 0.0 ? d2f_R : d2f_R + 1.1 * abs(d2f_R)
            else
                # Equação secante diagonal clássica (Eq. 5.105)
                qi_Qii = wi / vi
                
                # Regra de salvaguarda de positividade (Eq. 5.106)
                if qi_Qii >= 0.0
                    b[i] = qi_Qii
                else
                    b[i] = qi_Qii + 1.1 * abs(qi_Qii)
                end
            end
            
            # Salvaguarda dupla completa
            if !isfinite(b[i]) || b[i] > 1e8
                b[i] = 1e8
            elseif b[i] < 1e-5
                b[i] = 1e-5
            end
        end
        return spdiagm(0 => b)

    elseif strategy == :bfgs
        # Inicialização da matriz caso B_old seja nula ou dimensão incompatível
        B_mat = if isnothing(B_old) || size(B_old, 1) != n
            Matrix{Float64}(σ_fallback * I, n, n)
        else
            Matrix{Float64}(B_old)
        end
        
        Bs = B_mat * s
        sTBs = dot(s, Bs)
        yTs = dot(y_grad, s)
        
        # Proteção contra passo nulo ou degeneração de sTBs
        if sTBs <= 1e-12 || norm(s) < 1e-10 || !isfinite(sTBs)
            return B_mat
        end
        
        # Powell (1978) Damped BFGS Update para SQP com restrições:
        # Garante curvatura estritamente positiva (s' * r >= 0.2 * s' * B * s)
        # mesmo quando a curvatura do Lagrangeano é negativa (y' * s < 0).
        if isfinite(yTs) && yTs >= 0.2 * sTBs
            θ = 1.0
            r = y_grad
        else
            # Amortecimento em direção a B * s
            θ = max(0.0, min(1.0, (0.8 * sTBs) / (sTBs - (isfinite(yTs) ? yTs : 0.0))))
            r = θ * y_grad + (1.0 - θ) * Bs
        end
        
        sTr = dot(s, r)
        if sTr > 1e-12 && isfinite(sTr)
            B_new = B_mat + (r * r') / sTr - (Bs * Bs') / sTBs
            # Simetrização para eliminar erros de arredondamento
            B_new = 0.5 * (B_new + B_new')
            return B_new
        else
            return B_mat
        end

    else
        error("Unknown B calculation strategy: $strategy")
    end
end
end
