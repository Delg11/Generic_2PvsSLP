#!/usr/bin/env bash
# ==============================================================================
# run_cutest_benchmark.sh - Runner de Alta Performance para CUTEst via SSH
# ==============================================================================
# Repositório: Generic_2PvsSLP
# Algoritmos:
#   - UNIF (Gomes & Senne, 2011)
#   - Two-Phase / Inexact Restoration (Birgin et al., 2026)
# ==============================================================================
# Uso:
#   ./run_cutest_benchmark.sh [modo] [num_workers]
#
# Modos disponíveis:
#   test   : Teste rápido de sanidade (10 problemas didáticos, SLP puro)
#   slp    : Bateria completa CUTEst (n <= 200, m <= 100, SLP puro, todas as variantes de SAR, PH, ATR, RRU)
#   sqp    : Bateria CUTEst com aproximações quase-Newton (SQP)
#   all    : Bateria completa massiva (SLP + todas variantes SQP + todas estratégias ATR)
#
# Exemplos:
#   ./run_cutest_benchmark.sh test 4
#   ./run_cutest_benchmark.sh slp 16
#   nohup ./run_cutest_benchmark.sh slp 24 > /dev/null 2>&1 &
# ==============================================================================

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

MODE="${1:-test}"
WORKERS="${2:-${WORKERS:-0}}"  # 0 = detecção automática baseada no número de cores

# Parâmetros de Filtro e Execução (customizáveis via variáveis de ambiente)
MAX_VAR="${MAX_VAR:-200}"
MAX_CON="${MAX_CON:-100}"
MAX_ITER="${MAX_ITER:-500}"
TIMEOUT="${TIMEOUT:-60.0}"
QP_SOLVER="${QP_SOLVER:-gurobi}"
JULIA_BIN="${JULIA_BIN:-julia}"

TIMESTAMP="$(date +'%Y-%m-%d_%H-%M-%S')"
LOG_FILE="${LOG_DIR}/benchmark_${MODE}_${TIMESTAMP}.log"

echo "================================================================================"
echo "🚀 DISPARANDO BENCHMARK CUTEst (Generic_2PvsSLP)"
echo "================================================================================"
echo "📅 Data/Hora:       $(date +'%Y-%m-%d %H:%M:%S')"
echo "📁 Diretório:       ${PROJECT_DIR}"
echo "⚙️  Modo:            ${MODE}"
echo "👥 Workers:         $([ "${WORKERS}" -eq 0 ] && echo "Automático (Sys.CPU_THREADS - 2)" || echo "${WORKERS}")"
echo "🔢 Dimensões:       n <= ${MAX_VAR}, m <= ${MAX_CON}, max_iter = ${MAX_ITER}"
echo "⏱️  Timeout:         ${TIMEOUT}s por variante"
echo "🧮 QP Solver:       ${QP_SOLVER}"
echo "📝 Log consolidado: ${LOG_FILE}"
echo "================================================================================"

# Checagem básica do ambiente Julia
if ! command -v "${JULIA_BIN}" &> /dev/null; then
    echo "❌ ERRO: O comando '${JULIA_BIN}' não foi encontrado no PATH."
    echo "   Certifique-se de carregar o módulo do Julia ou definir JULIA_BIN=/caminho/para/julia"
    exit 1
fi

JULIA_VERSION="$("${JULIA_BIN}" -v || true)"
echo "ℹ️  Julia Detectado: ${JULIA_VERSION}"

# ==============================================================================
# Preparação do Ambiente Julia (Instanciação e Pré-compilação)
# ==============================================================================
if [ "${SKIP_INSTANTIATE:-false}" = "false" ]; then
    echo ""
    echo "================================================================================"
    echo "📦 INSTANCIANDO PROJETO E VERIFICANDO DEPENDÊNCIAS VIA PKG"
    echo "================================================================================"
    "${JULIA_BIN}" --project="${PROJECT_DIR}" -e '
        using Pkg
        println("⏳ Ativando e instanciando o ambiente do projeto (Pkg.instantiate())...")
        Pkg.instantiate()
        println("✅ Dependências verificadas e prontas!")
    '
    echo "================================================================================"
    echo ""
fi

# Montagem dos argumentos conforme o modo
COMMON_ARGS=(
    --project="${PROJECT_DIR}"
    "${PROJECT_DIR}/src/Generic_main.jl"
    --workers "${WORKERS}"
    --max-iter "${MAX_ITER}"
    --timeout "${TIMEOUT}"
    --qp-solver "${QP_SOLVER}"
)

case "${MODE}" in
    test|quick)
        echo "🧪 Modo TESTE RÁPIDO (10 problemas didáticos, SLP com todas as 4 estratégias ATR)..."
        BENCHMARK_CMD=(
            "${JULIA_BIN}"
            "${COMMON_ARGS[@]}"
            --mode test
            --problems "HS6,TRY-B,HS26,ROSENBR,SISSER,BT1,HS7,HS8,HS9,MARATOS"
            --tp-sar all
            --tp-ph all
            --tp-atr all
            --tp-atr-strat all
            --tp-rru all
            --tp-sqp none
            --unif-sar all
            --unif-ph all
            --unif-atr all
            --unif-atr-strat all
            --unif-sqp none
        )
        ;;

    slp)
        echo "📊 Modo SLP COMPLETO (n <= ${MAX_VAR}, m <= ${MAX_CON}, todas as variantes SAR, PH, ATR e RRU)..."
        BENCHMARK_CMD=(
            "${JULIA_BIN}"
            "${COMMON_ARGS[@]}"
            --mode filter
            --max-var "${MAX_VAR}"
            --max-con "${MAX_CON}"
            --tp-sar all
            --tp-ph all
            --tp-atr all
            --tp-atr-strat all
            --tp-rru all
            --tp-sqp none
            --unif-sar all
            --unif-ph all
            --unif-atr all
            --unif-atr-strat all
            --unif-sqp none
        )
        ;;

    sqp)
        echo "📐 Modo SQP (n <= ${MAX_VAR}, m <= ${MAX_CON}, aproximações de Hessiana)..."
        BENCHMARK_CMD=(
            "${JULIA_BIN}"
            "${COMMON_ARGS[@]}"
            --mode filter
            --max-var "${MAX_VAR}"
            --max-con "${MAX_CON}"
            --tp-sar all
            --tp-ph all
            --tp-atr all
            --tp-atr-strat legacy_parabolic
            --tp-rru true
            --tp-sqp quadratic
            --unif-sar all
            --unif-ph all
            --unif-atr all
            --unif-atr-strat legacy_parabolic
            --unif-sqp quadratic
        )
        ;;

    all|full)
        echo "💥 Modo COMPLETO MASSIVO (SLP + todas as variantes SQP + todas as estratégias ATR)..."
        BENCHMARK_CMD=(
            "${JULIA_BIN}"
            "${COMMON_ARGS[@]}"
            --mode filter
            --max-var "${MAX_VAR}"
            --max-con "${MAX_CON}"
            --tp-sar all
            --tp-ph all
            --tp-atr all
            --tp-atr-strat all
            --tp-rru all
            --tp-sqp all
            --unif-sar all
            --unif-ph all
            --unif-atr all
            --unif-atr-strat all
            --unif-sqp all
        )
        ;;

    *)
        echo "❌ Modo inválido: '${MODE}'. Escolha entre: test, slp, sqp, all."
        exit 1
        ;;
esac

echo ""
echo "Comando a ser executado:"
echo "${BENCHMARK_CMD[*]}"
echo ""

# Execução com tee gravando log e exibindo no terminal
"${BENCHMARK_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE="${PIPESTATUS[0]}"

if [ "${EXIT_CODE}" -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ Benchmark finalizado com sucesso!"
    echo "📊 Gerando análises estatísticas e gráficos de resumo..."
    echo "================================================================================"
    
    # Procura a pasta de resultados mais recente criada pelo benchmark
    LATEST_RESULTS="$(find "${PROJECT_DIR}/benchmarks" -maxdepth 1 -type d -name "Benchmark_Results_*" | sort -r | head -n 1 || true)"
    
    if [ -n "${LATEST_RESULTS}" ] && [ -d "${LATEST_RESULTS}" ]; then
        echo "📁 Pasta de resultados identificada: ${LATEST_RESULTS}"
        export BENCHMARK_RESULTS_DIR="${LATEST_RESULTS}"
        
        # Roda o script de estatísticas configurado
        "${JULIA_BIN}" --project="${PROJECT_DIR}" "${PROJECT_DIR}/src/min_run_stats.jl" 2>&1 | tee -a "${LOG_FILE}" || true
    fi
    
    echo "🏁 Processo concluído! Logs completos salvos em: ${LOG_FILE}"
else
    echo "❌ O benchmark terminou com erro (código ${EXIT_CODE}). Verifique o log: ${LOG_FILE}"
    exit "${EXIT_CODE}"
fi
