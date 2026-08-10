module Generic_module_ComputeB

export compute_B

using SparseArrays
using LinearAlgebra
using ..SharedTypes
function compute_B(strategy::Symbol, n::Int, s::Vector{Float64}, y_grad::Vector{Float64}, σ_fallback::Float64, problem::OptimizationProblem, x_new::Vector{Float64}, λ::Vector{Float64}, B_old::Union{AbstractMatrix{Float64}, Nothing} = nothing)
    ∇L(x, λ) = problem.∇f(x) + problem.∇h(x)' * λ
    x_prev = x_new - s
    if strategy == :exact
        B_exact = problem.∇²L(x_new, λ)
        return B_exact
    elseif strategy == :diag_exact
        B_exact = problem.∇²L(x_new, λ)
        
        # Extrai o vetor com os elementos da diagonal principal
        d = diag(B_exact) 
        
        # Constrói e retorna a matriz diagonal esparsa
        return spdiagm(0 => d)
    elseif strategy == :identity
        return σ_fallback * spdiagm(0 => ones(n))
        
    elseif strategy == :spectral
        if norm(s) < 1e-8 || norm(y_grad) < 1e-8
            return σ_fallback * spdiagm(0 => ones(n))
        end
        sTy = dot(s, y_grad)
        yTy = dot(y_grad, y_grad)
        if sTy <= 1e-12
            return σ_fallback * spdiagm(0 => ones(n))
        end
        alpha = yTy / sTy
        alpha = clamp(alpha, 1e-4, 1e4)
        return alpha * spdiagm(0 => ones(n))
        
    elseif strategy == :reciprocal
        # Obtém o gradiente do Lagrangiano (ou da obj) no ponto corrente
        # Nota: Assume-se que problem possui o método ∇L(x, λ). 
        # Caso use apenas ∇f, altere para problem.∇f(x_new)
        grad = ∇L(x_new, λ)
        
        b = zeros(n)
        for i in 1:n
            # Segunda derivada implícita da aproximação por variáveis recíprocas (Eq. 5.90)
            d2f_R = -2.0 / x_new[i] * grad[i]
            
            # Regra de salvaguarda de positividade para garantir convexidade (Eq. 5.91)
            if d2f_R >= 0.0
                b[i] = d2f_R
            else
                b[i] = d2f_R + 1.1 * abs(d2f_R)
            end
            
            # Salvaguarda para evitar valores nulos ou negativos (matriz singular)
            if b[i] < 1e-5
                b[i] = 1e-5
            end
        end
        return spdiagm(0 => b)
        
    elseif strategy == :exponential
        grad = ∇L(x_new, λ)
        grad_old = ∇L(x_prev, λ) # Usa o vetor x_prev do escopo superior
        b = zeros(n)
        for i in 1:n
            x_curr = x_new[i]
            x_prev_i = x_prev[i] # Nova variável local, evita shadowing
            g_curr = grad[i]
            g_prev = grad_old[i]
            
            # Critério para escolha do expoente a_i adaptativo (Eq. 5.95 e salvaguardas)
            if abs(s[i]) < 1e-6 || g_prev * g_curr <= 1e-6 
                ai = -1.0  # Cai de volta na aproximação recíproca clássica
            else
                ratio_g = g_prev / g_curr
                ratio_x = x_prev_i / x_curr
                if ratio_g > 0.0 && ratio_x > 0.0
                    ai = 1.0 + log(ratio_g) / log(ratio_x)
                else
                    ai = -1.0
                end
            end
            
            # Segunda derivada da aproximação exponencial (Eq. 5.99)
            d2f_E = (ai - 1.0) / x_curr * g_curr
            
            # Regra de salvaguarda de positividade (Eq. 5.100)
            if d2f_E >= 0.0
                b[i] = d2f_E
            else
                b[i] = d2f_E + 1.1 * abs(d2f_E)
            end
            
            if b[i] < 1e-5
                b[i] = 1e-5
            end
        end
        return spdiagm(0 => b)
        
    elseif strategy == :quasi_newton
        grad = ∇L(x_new, λ)
        
        b = zeros(n)
        for i in 1:n
            vi = s[i]       # variação de passo s_i
            wi = y_grad[i]  # variação de gradiente y_i
            
            if abs(vi) < 1e-10
                # Caso o passo seja nulo/muito pequeno ou seja a primeira iteração,
                # inicializa-se com a aproximação recíproca clássica (pág. 116)
                d2f_R = -2.0 / x_new[i] * grad[i]
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
            
            # Limite inferior estrito positivo delta (Eq. 5.101) para garantir convexidade
            if b[i] < 1e-5
                b[i] = 1e-5
            end
        end
        return spdiagm(0 => b)
    elseif strategy == :bfgs
        # Inicialização da matriz na primeira iteração
        if isnothing(B_old)
            return σ_fallback * Matrix{Float64}(I, n, n)
        end
        # B_old=problem.∇²L(x_prev, λ)
        
        yTs = dot(y_grad, s)
        
        # Condição de curvatura: só atualiza se estritamente positiva
        if yTs > 1e-8
            Bs = B_old * s
            sTBs = dot(s, Bs)
            
            # Fórmula de atualização BFGS para a matriz B (Hessiana aproximada)
            B_new = B_old + (y_grad * y_grad') / yTs - (Bs * Bs') / sTBs
            return B_new
        else
            # Pula a atualização para manter a matriz definida positiva
            return B_old
        end
    else
        error("Unknown B calculation strategy: $strategy")
    end
end
end