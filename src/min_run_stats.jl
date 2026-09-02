# ==============================================================================
# RUNNER: STATISTICAL ANALYSIS
# ==============================================================================

# Carrega o arquivo contendo o módulo
include("SQP_Stats.jl")
using .Generic_module_Stats

# Caminho absoluto para a pasta de resultados
# results_dir = "E:\\Generic_2PvsSLP\\Caso SQP na Colecao CUTEST"
results_dir = "C:\\Github\\Generic_2PvsSLP\\Benchmark_Results_2026-08-31_22-24"
# Executa a rotina principal
Generic_module_Stats.run_statistical_analysis(results_dir)