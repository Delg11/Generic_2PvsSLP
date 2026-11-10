# ==============================================================================
# RUNNER: ANÁLISE ESTATÍSTICA E PERFIS DE DESEMPENHO
# ==============================================================================

# 1. Escolha a análise desejada:
#    :sbpo_artigo  -> Artigo Completo SBPO 2027 (critério KKT, 6 grupos G1 a G6)
#    :sbpo_resumo  -> Resumo Expandido SBPO 2026 (n <= 10, m <= 1, 4 perfis)
#    :tese_cap3    -> Tese Capítulo 3 (critério f-profile por valor de objetivo viável)
#    :tese_cap4    -> Tese Capítulo 4 (extensão SQP / aproximações Hessianas)
target_analysis = :sbpo_artigo

# 2. Caminho para a pasta contendo os dados (Consolidated_*.csv ou benchmark_*.csv)
# Se não for informado, busca a subpasta mais recente em benchmarks/
function find_default_results_dir()
    benchmarks_dir = normpath(joinpath(@__DIR__, "..", "benchmarks"))
    entries = filter(d -> isdir(joinpath(benchmarks_dir, d)) && !startswith(d, "."), readdir(benchmarks_dir))
    if !isempty(entries)
        # Ordena para pegar a pasta mais recente
        sorted_entries = sort(entries; rev=true)
        return joinpath(benchmarks_dir, sorted_entries[1])
    end
    return normpath(joinpath(@__DIR__, "..", "data"))
end

results_dir = get(ENV, "BENCHMARK_RESULTS_DIR", find_default_results_dir())
println("📂 Diretório de resultados selecionado: $results_dir")

# 3. Disparo da análise selecionada
if target_analysis == :sbpo_artigo
    include(normpath(joinpath(@__DIR__, "stats", "stats_sbpo.jl")))
    using .StatsSBPO
    StatsSBPO.run_statistical_analysis(results_dir; target=:artigo)

elseif target_analysis == :sbpo_resumo
    include(normpath(joinpath(@__DIR__, "stats", "stats_sbpo.jl")))
    using .StatsSBPO
    StatsSBPO.run_statistical_analysis(results_dir; target=:resumo)

elseif target_analysis == :tese_cap3
    include(normpath(joinpath(@__DIR__, "stats", "stats_tese_cap3_fprofile.jl")))
    using .StatsCap3FProfile
    StatsCap3FProfile.run_statistical_analysis(results_dir)

elseif target_analysis == :tese_cap4
    include(normpath(joinpath(@__DIR__, "stats", "stats_tese_cap4_sqp.jl")))
    using .StatsCap4SQP
    StatsCap4SQP.run_statistical_analysis(results_dir)

else
    error("Opção desconhecida: $target_analysis. Escolha :sbpo_artigo, :sbpo_resumo, :tese_cap3 ou :tese_cap4.")
end
