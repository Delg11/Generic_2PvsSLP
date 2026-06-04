# ==============================================================================
# Software Proprietário • Todos os Direitos Reservados
# ==============================================================================
# 1. IMPORTS E INCLUDES
# ==============================================================================
using Pkg

# Ativa o ambiente associado ao diretório onde este arquivo está salvo
Pkg.activate(@__DIR__) 
# Lê o Manifest.toml e instala/pré-compila todas as dependências necessárias
Pkg.instantiate()

# Importante: Certifique-se de que o ArgParse está adicionado ao seu ambiente
# Pkg.add("ArgParse")
using ArgParse

using CUTEst
using LinearAlgebra
using Printf
using Statistics
using DataFrames
using XLSX
using Dates
using Plots
using CSV

# Inclui os módulos dos algoritmos
include("Generic_Sharedtypes.jl")
include("Generic_module_Twophase.jl")
include("Generic_module_slp.jl")   

using .SharedTypes
using .Generic_module_Twophase
using .Generic_module_slp

# ==============================================================================
# 2. CONFIGURAÇÃO DE ARGUMENTOS DE LINHA DE COMANDO
# ==============================================================================
function parse_commandline()
    s = ArgParseSettings(description="Benchmark de Otimização SLP e TwoPhase")

    @add_arg_table s begin
        "--mode", "-m"
            help = "Modo de execução: 'test' para lista fixa de problemas ou 'filter' para buscar no CUTEst."
            arg_type = String
            default = "test"
        "--problems", "-p"
            help = "Lista de problemas separados por vírgula (usado apenas no modo 'test')."
            arg_type = String
            default = "HS6,ROSENBR"
        "--max-var", "-v"
            help = "Número máximo de variáveis (-1 para sem limite)."
            arg_type = Int
            default = -1
        "--max-con", "-c"
            help = "Número máximo de restrições (-1 para sem limite)."
            arg_type = Int
            default = -1
    end

    return parse_args(s)
end

args = parse_commandline()

# ==============================================================================
# 3. FUNÇÕES DE AVALIAÇÃO DE MÉTRICAS E PLOTS (PÓS-PROCESSAMENTO)
# ==============================================================================
function evaluate_m1(it_v, it_b, f_v, f_b, h_v, h_b, eps)
    if h_v > h_b + eps || f_v > f_b + eps return :Loss end
    if it_v < it_b return :Win elseif it_v > it_b return :Loss else return :Tie end
end

function evaluate_m2(f_v, f_b, h_v, h_b, eps)
    if h_v > h_b + eps return :Loss end
    if f_v < f_b - eps return :Win elseif f_v > f_b + eps return :Loss else return :Tie end
end

function evaluate_m3(t_v, t_b, f_v, f_b, h_v, h_b, eps)
    if h_v > h_b + eps || f_v > f_b + eps return :Loss end
    if t_v < t_b return :Win elseif t_v > t_b return :Loss else return :Tie end
end

function plot_performance_profile_fbest(df, variants_to_compare, title_suffix; 
                                        metric=:Tempo_ms, max_tau=10.0, 
                                        eps_h=1e-3, eps_f=1e-3, 
                                        strict_intersection=true)
    df_work = filter(r -> r.Variante in variants_to_compare, df)
    if nrow(df_work) == 0 return nothing end
    
    df_work[!, :Viable] = [!ismissing(r.h_norm) && r.h_norm <= eps_h for r in eachrow(df_work)]
    
    gdf_prob = groupby(df_work, :Problema)
    problemas_validos = String[]
    
    for sub in gdf_prob
        if strict_intersection
            has_all_viable = true
            for v in variants_to_compare
                idx = findfirst(r -> r.Variante == v, eachrow(sub))
                if isnothing(idx) || !sub[idx, :Viable]
                    has_all_viable = false
                    break
                end
            end
            if has_all_viable push!(problemas_validos, sub.Problema[1]) end
        else
            if any(sub.Viable) push!(problemas_validos, sub.Problema[1]) end
        end
    end
    
    filter!(r -> r.Problema in problemas_validos, df_work)
    if nrow(df_work) == 0 return nothing end

    df_feasible = filter(r -> r.Viable, df_work)
    f_bests = combine(groupby(df_feasible, :Problema), :f_final => minimum => :f_best)
    df_work = leftjoin(df_work, f_bests, on=:Problema)
    
    df_work[!, :Solved] = [!ismissing(r.f_best) && r.Viable && !ismissing(r.f_final) && r.f_final <= r.f_best + eps_f for r in eachrow(df_work)]
    df_work[!, :Metric] = [r.Solved ? Float64(r[metric]) : Inf for r in eachrow(df_work)]
    
    gdf = groupby(df_work, :Problema)
    min_metrics = Dict(k.Problema => minimum(v.Metric) for (k, v) in pairs(gdf))
    
    df_work[!, :Ratio] = [min_metrics[r.Problema] == Inf ? Inf : r.Metric / min_metrics[r.Problema] for r in eachrow(df_work)]
    
    metric_label = metric == :Iteracoes ? "Iterations" : (metric == :Tempo_ms ? "Time (ms)" : string(metric))

    nome_legendas = Dict(
        "BASE_SLP"            => "SLP",
        "BASE_2P"             => "BASE_2P",
        "2P_M1_BQ0_ATR0_URU1" => "2P",
        "SLP_BQ1_ATR0"        => "SLP_Iso",
        "SLP_BQ1_ATR1"        => "SLP_Aniso",
        "2P_M1_BQ1_ATR0_URU1" => "2P_Iso",
        "2P_M1_BQ1_ATR1_URU1" => "2P_Aniso"
    )

    num_problems = length(keys(min_metrics))

    p = plot(title="Performance Profile\n$title_suffix ($num_problems problems)\n", 
             xlabel="Factor (τ) relative to best $metric_label", ylabel="Proportion of solved problems",
             legend=:bottomright, xlims=(1.0, max_tau), ylims=(0.0, 1.05), framestyle=:box)
    
    taus = range(1.0, stop=max_tau, length=500)
    
    for var in variants_to_compare
        df_var = filter(row -> row.Variante == var, df_work)
        rho = [sum(df_var.Ratio .<= t) / num_problems for t in taus]
        plot!(p, taus, rho, label=get(nome_legendas, var, var), linewidth=2)
    end
    
    return p
end

# ==============================================================================
# 4. CONFIGURAÇÃO GERAL E SELEÇÃO DE ALGORITMOS
# ==============================================================================
println("\n" * "=" ^ 80)
println("⚙️ CONFIGURANDO VARIANTES E AMBIENTE DE EXPORTAÇÃO")
println("=" ^ 80)

const RUN_TWOPHASE = true
const RUN_SLP      = true

common_max_iter = 500
common_delta0   = 0.1
common_tolG     = 1e-3
common_tolF     = 5e-2
common_tolS     = 1e-4
common_maxcount = 3
common_verbose  = false 

data_hora_atual = Dates.format(now(), "yyyy-mm-dd_HH-MM")
dir_resultados = "Resultados_Benchmark_$(data_hora_atual)"
mkpath(dir_resultados)
println("📁 Diretório de resultados criado: $dir_resultados")

function build_twophase_params(M, bq, atr, uru)
    return TwoPhaseParams(
        max_outer_iter = common_max_iter, δ0_opt = common_delta0, δ0_resto = common_delta0,
        tolG = common_tolG, tolF = common_tolF, tolS = common_tolS, maxcount = common_maxcount,
        verbose_out = common_verbose, use_slp_stopping = true, rfeas = 1e-12, δmin = 1e-12, 
        δmax = 1e16, r_resto = 0.9, τ1 = 0.1, τ2 = 0.25, αL = 1e-8, αR = 1e-8,
        θ0 = 0.90, αΦ = 1e-8, max_iter_resto = 500, verbose_in = common_verbose, debugverbose = false,
        non_monotone_M = M, backtracking_quadratic = bq, anisotropic_trust_region = atr, use_ratio_update = uru
    )
end

variantes_twophase = []
if RUN_TWOPHASE
    for M in [1], bq in [false, true], atr in [false, true], uru in [true]
        if !bq && atr continue end 
        is_base = (M == 1 && !bq && !atr && !uru)
        nome = is_base ? "BASE_2P" : "2P_M$(M)_BQ$(Int(bq))_ATR$(Int(atr))_URU$(Int(uru))"
        push!(variantes_twophase, (nome=nome, is_base=is_base, params=build_twophase_params(M, bq, atr, uru), M=M, BQ=bq, ATR=atr, URU=uru))
    end
end

function build_slp_params(bq, atr)
    return SLPParams(
        delta0 = common_delta0, tolG = common_tolG, tolF = common_tolF, tolS = common_tolS,
        maxiter = common_max_iter, maxcount = common_maxcount, verbose = common_verbose,
        verbose_out = common_verbose, debugverbose = false, backtracking_quadratic = bq, anisotropic_trust_region = atr
    )
end

variantes_slp = []
if RUN_SLP
    for bq in [false, true], atr in [false, true]
        if !bq && atr continue end 
        is_base = (!bq && !atr)
        nome = is_base ? "BASE_SLP" : "SLP_BQ$(Int(bq))_ATR$(Int(atr))"
        push!(variantes_slp, (nome=nome, is_base=is_base, params=build_slp_params(bq, atr), BQ=bq, ATR=atr))
    end
end

# DataFrames base em memória
df_twophase = DataFrame(Problema=String[], nvar=Int[], ncon=Int[], Variante=String[], 
                        M=Int[], BQ=Bool[], ATR=Bool[], URU=Bool[], Status=String[], 
                        Iteracoes=Int[], Tempo_ms=Float64[], f_final=Float64[], h_norm=Float64[], Is_Base=Bool[])

df_slp = DataFrame(Problema=String[], nvar=Int[], ncon=Int[], Variante=String[], 
                   BQ=Bool[], ATR=Bool[], Status=String[], Iteracoes=Int[], 
                   Tempo_ms=Float64[], f_final=Float64[], h_norm=Float64[], Is_Base=Bool[])

# ==============================================================================
# 5. SELEÇÃO DE PROBLEMAS E LOOP PRINCIPAL
# ==============================================================================
if args["mode"] == "test"
    problems = String.(split(args["problems"], ","))
    println("\nModo TESTE ativado. Analisando a lista de problemas fixos.")
elseif args["mode"] == "filter"
    println("\nModo FILTER ativado. Construindo query...")
    
    # Dicionário para armazenar apenas os filtros que o usuário realmente definiu
    filtros = Dict{Symbol, Int}()
    
    print("Filtros -> ")
    if args["max-var"] != -1
        filtros[:max_var] = args["max-var"]
        print("Max Var: $(args["max-var"]) | ")
    else
        print("Max Var: [Sem limite] | ")
    end
    
    if args["max-con"] != -1
        filtros[:max_con] = args["max-con"]
        print("Max Con: $(args["max-con"])")
    else
        print("Max Con: [Sem limite]")
    end
    println() # Quebra de linha
    
    # O ; desempacota o dicionário passando as chaves como argumentos nomeados
    problems = select_sif_problems(; filtros...)
else
    println("\n❌ ERRO: O modo especificado ('$(args["mode"])') é inválido. Use 'test' ou 'filter'.")
    exit(1)
end

println("Total de problemas selecionados: $(length(problems))")

if length(problems) == 0
    println("Nenhum problema encontrado para os parâmetros selecionados. Encerrando.")
    exit(0)
end

function get_status_string(status_code::Int)
    if status_code == 0 return "KKT_OK"
    elseif status_code == 1 return "SMALL_STEP"
    elseif status_code == 2 return "MAX_IT"
    elseif status_code == 3 return "INFEASIBLE_STATIONARY"
    elseif status_code == 4 return "STALLED_INFEASIBLE"
    else return "UNKNOWN_$(status_code)" end
end

for prob_name in problems
    local nlp = nothing
    try
        nlp = CUTEstModel{Float64}(prob_name)
        problem = SharedTypes.build_optimization_problem(nlp)
        x0 = clamp.(nlp.meta.x0, problem.xl, problem.xu)
        dim, ncon = nlp.meta.nvar, nlp.meta.ncon

        println("\n🚀 PROCESSANDO: $prob_name [Var: $dim | Con: $ncon]")

        if RUN_TWOPHASE
            for var in variantes_twophase
                print("  [Two-Phase] > $(var.nome)... ")
                tempo_exec = @elapsed begin
                    y, λ, θ, iter, status, hist, log = two_phase_optimization(problem, x0, var.params; solver_choice=:gurobi, use_quadratic=false, history=false)
                end
                push!(df_twophase, (prob_name, dim, ncon, var.nome, var.M, var.BQ, var.ATR, var.URU, get_status_string(status), iter, tempo_exec * 1000.0, problem.f(y), norm(problem.h(y)), var.is_base))
                println("Pronto. ($(round(tempo_exec * 1000, digits=1))ms)")
            end
        end

        if RUN_SLP
            for var in variantes_slp
                print("  [SLP] > $(var.nome)... ")
                tempo_exec = @elapsed begin
                    y, λ, θ, iter, status, hist, log = solve_slp_trust_region(problem, x0, var.params)
                end
                push!(df_slp, (prob_name, dim, ncon, var.nome, var.BQ, var.ATR, get_status_string(status), iter, tempo_exec * 1000.0, problem.f(y), norm(problem.h(y)), var.is_base))
                println("Pronto. ($(round(tempo_exec * 1000, digits=1))ms)")
            end
        end
    catch e
        println("❌ ERRO em $prob_name: $e")
    finally
        isnothing(nlp) || finalize(nlp)
        
        # Backup incremental para evitar perda de dados em caso de crash
        CSV.write(joinpath(dir_resultados, "backup_parcial_twophase.csv"), df_twophase)
        CSV.write(joinpath(dir_resultados, "backup_parcial_slp.csv"), df_slp)
    end
end

# ==============================================================================
# 6. ANÁLISE CONSOLIDADA (INTEGRAÇÃO DIRETA NA MEMÓRIA)
# ==============================================================================
println("\n" * "=" ^ 80)
println("📊 INICIANDO ANÁLISE CONSOLIDADA E ESTATÍSTICAS")
println("=" ^ 80)

df_all = vcat(df_slp, df_twophase, cols=:union)

ignored_variants = ["BASE_2P", "2P_M1_BQ1_ATR0_URU0", "2P_M1_BQ1_ATR1_URU0"]
filter!(row -> !(row.Variante in ignored_variants), df_all)

invalid_mask = isnan.(df_all.f_final) .| isinf.(df_all.f_final) .| isnan.(df_all.h_norm) .| isinf.(df_all.h_norm)
df_valid = df_all[.!invalid_mask, :]

df_Scenario_A = df_valid
df_Scenario_B = filter(r -> !ismissing(r.Iteracoes) && r.Iteracoes > 0 && r.Status != "MAX_IT", df_valid)
df_Scenario_C = filter(r -> !ismissing(r.Status) && r.Status == "KKT_OK", df_valid)

function calculate_statistics(df_scenario)
    if nrow(df_scenario) == 0 return DataFrame() end
    combine(groupby(df_scenario, :Variante),
        :Iteracoes => (x -> mean(skipmissing(x))) => :Mean_Iterations,
        :Iteracoes => (x -> median(skipmissing(x))) => :Median_Iterations,
        :Tempo_ms  => (x -> mean(skipmissing(x))) => :Mean_Time_ms,
        :f_final   => (x -> mean(skipmissing(x))) => :Mean_f_final,
        :f_final   => (x -> median(skipmissing(x))) => :Median_f_final,
        :h_norm    => (x -> mean(skipmissing(x))) => :Mean_h_norm,
        :h_norm    => (x -> median(skipmissing(x))) => :Median_h_norm,
        nrow => :Num_Problems
    )
end

stats_A = calculate_statistics(df_Scenario_A)
stats_B = calculate_statistics(df_Scenario_B)
stats_C = calculate_statistics(df_Scenario_C)

eps = 1e-3

# --- INTRA-METHOD ---
intra_pairs = [
    ("BASE_SLP", "SLP_BQ1_ATR0"), ("BASE_SLP", "SLP_BQ1_ATR1"),
    ("2P_M1_BQ0_ATR0_URU1", "2P_M1_BQ1_ATR0_URU1"), ("2P_M1_BQ0_ATR0_URU1", "2P_M1_BQ1_ATR1_URU1")
]
intra_results = []
for (base_name, var_name) in intra_pairs
    df_base = filter(r -> r.Variante == base_name, df_valid)
    df_var  = filter(r -> r.Variante == var_name, df_valid)
    df_join = innerjoin(df_var, df_base, on=:Problema, makeunique=true)
    
    m1_w, m1_t, m1_l, m2_w, m2_t, m2_l, m3_w, m3_t, m3_l = zeros(Int, 9)
    for r in eachrow(df_join)
        res_m1 = evaluate_m1(r.Iteracoes, r.Iteracoes_1, r.f_final, r.f_final_1, r.h_norm, r.h_norm_1, eps)
        res_m1 == :Win ? m1_w += 1 : (res_m1 == :Tie ? m1_t += 1 : m1_l += 1)
        res_m2 = evaluate_m2(r.f_final, r.f_final_1, r.h_norm, r.h_norm_1, eps)
        res_m2 == :Win ? m2_w += 1 : (res_m2 == :Tie ? m2_t += 1 : m2_l += 1)
        res_m3 = evaluate_m3(r.Tempo_ms, r.Tempo_ms_1, r.f_final, r.f_final_1, r.h_norm, r.h_norm_1, eps)
        res_m3 == :Win ? m3_w += 1 : (res_m3 == :Tie ? m3_t += 1 : m3_l += 1)
    end
    push!(intra_results, (Base=base_name, Variant=var_name, Total=nrow(df_join), M1_W=m1_w, M1_T=m1_t, M1_L=m1_l, M2_W=m2_w, M2_T=m2_t, M2_L=m2_l, M3_W=m3_w, M3_T=m3_t, M3_L=m3_l))
end
df_res_intra = DataFrame(intra_results)

# --- INTER-METHOD ---
m1_w, m1_t, m1_l, m2_w, m2_t, m2_l, m3_w, m3_t, m3_l, total_inter = zeros(Int, 10)
inter_details = []

for prob in unique(df_valid.Problema)
    global m1_w, m1_t, m1_l, m2_w, m2_t, m2_l, m3_w, m3_t, m3_l, total_inter
    
    df_prob = filter(r -> r.Problema == prob, df_valid)
    slps_valid = filter(r -> (startswith(r.Variante, "SLP") || r.Variante == "BASE_SLP") && r.h_norm <= eps, df_prob)
    tps_valid = filter(r -> startswith(r.Variante, "2P") && r.h_norm <= eps, df_prob)
    
    if nrow(slps_valid) > 0 && nrow(tps_valid) > 0
        best_slp = sort(slps_valid, [:f_final, :Iteracoes])[1, :]
        best_2p = sort(tps_valid, [:f_final, :Iteracoes])[1, :]
        total_inter += 1
        
        res_m1 = evaluate_m1(best_2p.Iteracoes, best_slp.Iteracoes, best_2p.f_final, best_slp.f_final, best_2p.h_norm, best_slp.h_norm, eps)
        res_m1 == :Win ? m1_w += 1 : (res_m1 == :Tie ? m1_t += 1 : m1_l += 1)
        res_m2 = evaluate_m2(best_2p.f_final, best_slp.f_final, best_2p.h_norm, best_slp.h_norm, eps)
        res_m2 == :Win ? m2_w += 1 : (res_m2 == :Tie ? m2_t += 1 : m2_l += 1)
        res_m3 = evaluate_m3(best_2p.Tempo_ms, best_slp.Tempo_ms, best_2p.f_final, best_slp.f_final, best_2p.h_norm, best_slp.h_norm, eps)
        res_m3 == :Win ? m3_w += 1 : (res_m3 == :Tie ? m3_t += 1 : m3_l += 1)

        push!(inter_details, (Problema = prob, Best_SLP = best_slp.Variante, SLP_f_final = best_slp.f_final, SLP_Iteracoes = best_slp.Iteracoes, Best_2P = best_2p.Variante, f_final_2P = best_2p.f_final, Iteracoes_2P = best_2p.Iteracoes, Resultado_M1_Iteracoes = string(res_m1), Resultado_M2_F_Final = string(res_m2), Resultado_M3_Tempo = string(res_m3)))
    end
end

df_res_inter = DataFrame(Comparison=["Best 2P vs Best SLP"], Total=[total_inter], M1_W=[m1_w], M1_T=[m1_t], M1_L=[m1_l], M2_W=[m2_w], M2_T=[m2_t], M2_L=[m2_l], M3_W=[m3_w], M3_T=[m3_t], M3_L=[m3_l])
df_inter_details = DataFrame(inter_details)

# --- PLOTS ---
plots_dir = joinpath(dir_resultados, "Plots_Profiles")
mkpath(plots_dir) 

all_6_variants = ["BASE_SLP", "SLP_BQ1_ATR0", "SLP_BQ1_ATR1", "2P_M1_BQ0_ATR0_URU1", "2P_M1_BQ1_ATR0_URU1", "2P_M1_BQ1_ATR1_URU1"]
duels = [
    ("SLP_vs_SLP_Iso", ["BASE_SLP", "SLP_BQ1_ATR0"]), ("SLP_vs_SLP_Aniso", ["BASE_SLP", "SLP_BQ1_ATR1"]),
    ("SLP_Iso_vs_SLP_Aniso", ["SLP_BQ1_ATR0", "SLP_BQ1_ATR1"]), ("SLP_vs_2P", ["BASE_SLP", "2P_M1_BQ0_ATR0_URU1"]),
    ("SLP_Iso_vs_2P_Iso", ["SLP_BQ1_ATR0", "2P_M1_BQ1_ATR0_URU1"]), ("SLP_Aniso_vs_2P_Aniso", ["SLP_BQ1_ATR1", "2P_M1_BQ1_ATR1_URU1"]),
    ("2P_vs_2P_Iso", ["2P_M1_BQ0_ATR0_URU1", "2P_M1_BQ1_ATR0_URU1"]), ("2P_vs_2P_Aniso", ["2P_M1_BQ0_ATR0_URU1", "2P_M1_BQ1_ATR1_URU1"]),
    ("2P_Iso_vs_2P_Aniso", ["2P_M1_BQ1_ATR0_URU1", "2P_M1_BQ1_ATR1_URU1"])
]

p_all_time = plot_performance_profile_fbest(df_valid, all_6_variants, "All Variants", metric=:Tempo_ms)
if !isnothing(p_all_time) savefig(p_all_time, joinpath(plots_dir, "01_All_Variants_Time.png")) end

for (name, vars) in duels
    p_t = plot_performance_profile_fbest(df_valid, vars, replace(name, "_" => " "), metric=:Tempo_ms)
    if !isnothing(p_t) savefig(p_t, joinpath(plots_dir, "Duel_$(name)_Time.png")) end
end

# --- EXPORTAÇÃO EXCEL ---
output_file = joinpath(dir_resultados, "Consolidated_Statistics.xlsx")
XLSX.openxlsx(output_file, mode="w") do xf
    XLSX.rename!(xf[1], "Scenario_A")
    if nrow(stats_A) > 0 XLSX.writetable!(xf[1], collect(eachcol(stats_A)), names(stats_A)) end
    
    XLSX.addsheet!(xf, "Scenario_B")
    if nrow(stats_B) > 0 XLSX.writetable!(xf[2], collect(eachcol(stats_B)), names(stats_B)) end
    
    XLSX.addsheet!(xf, "Scenario_C")
    if nrow(stats_C) > 0 XLSX.writetable!(xf[3], collect(eachcol(stats_C)), names(stats_C)) end
    
    XLSX.addsheet!(xf, "Intra_Comparison")
    if nrow(df_res_intra) > 0 XLSX.writetable!(xf[4], collect(eachcol(df_res_intra)), names(df_res_intra)) end
    
    XLSX.addsheet!(xf, "Inter_Comparison")
    if nrow(df_res_inter) > 0 XLSX.writetable!(xf[5], collect(eachcol(df_res_inter)), names(df_res_inter)) end
    
    XLSX.addsheet!(xf, "Inter_Details")
    if nrow(df_inter_details) > 0 XLSX.writetable!(xf[6], collect(eachcol(df_inter_details)), names(df_inter_details)) end
end
println("✅ Execução e Análise concluídas. Resultados e gráficos salvos em: $dir_resultados")