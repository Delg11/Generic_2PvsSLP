# Generic_2PvsSLP: UNIF and Two-Phase Trust-Region Benchmarking

Este repositório contém o arcabouço computacional e estatístico desenvolvido para avaliar o desempenho de algoritmos de Programação Não-Linear (PNL), comparando abordagens Uniformes (UNIF / SLP Clássico) contra o método de Região de Confiança de Duas Fases (Two-Phase).

O projeto realiza uma avaliação modular sobre o conjunto de problemas CUTEst, testando progressivamente o impacto de passos parabólicos, anisotropia, atualização heurística de raios e aproximações de matrizes Hessianas (SQP).

📊 **[Acesse o Dashboard Interativo de Perfis de Desempenho (Dolan-Moré)](https://delg11.github.io/Generic_2PvsSLP/)**

## Estrutura do Repositório

* `src/`: Códigos-fonte dos algoritmos e scripts analíticos.
  * `Generic_main.jl`: Script principal de execução e configuração dos testes.
  * `Generic_Sharedtypes.jl`: Definições de estruturas de dados comuns.
  * `Generic_module_slp.jl`: Algoritmo UNIF (SLP e extensões SQP).
  * `Generic_module_Twophase.jl`: Algoritmo de Região de Confiança de Duas Fases.
  * `Generic_module_Stats.jl`: Módulo para geração de estatísticas e perfis de desempenho.
* `data/`: Base de dados de testes.
  * `benchmark_slp.csv`: Resultados brutos das execuções dos métodos uniformes.
  * `benchmark_twophase.csv`: Resultados brutos das execuções dos métodos de duas fases.
  * `Consolidated_Statistics.xlsx`: Cópia da planilha de resultados.
* `analysis/`: Resultados processados e gráficos consolidados.
  * `Summary_Statistics.csv`: Resumo descritivo geral das execuções.
  * `Consolidated_Statistics.xlsx`: Planilha automatizada com análises Inter e Intra-métodos.
  * `Plots/`: Gráficos `.png` de Perfis de Desempenho estruturados por grupos de comparação.
* Raiz (`/`):
  * `index.html`: Dashboard em React/Plotly.js para visualização interativa hospedado no GitHub Pages.
  * `instalar_dependencias.jl`: Script auxiliar para configuração de pacotes.
  * `Project.toml` / `Manifest.toml`: Controle de dependências e ambiente do Julia.

## Parâmetros de Execução (CLI)

O script `Generic_main.jl` suporta diversos argumentos via linha de comando para customizar a execução dos testes e realizar a habilitação de características.

| Argumento | Atalho | Padrão | Descrição |
| :--- | :---: | :--- | :--- |
| `--mode` | `-m` | `test` | Modo de execução: `test` (problemas específicos) ou `filter` (bateria CUTEst). |
| `--problems` | `-p` | `HS6` | Lista separada por vírgula dos problemas a executar no modo `test`. |
| `--max-var` | `-v` | `-1` | Número máximo de variáveis permitidas no modo `filter` (-1 para sem limite). |
| `--max-con` | `-c` | `-1` | Número máximo de restrições permitidas no modo `filter` (-1 para sem limite). |
| `--run-slp` | | `true` | Ativa/desativa a execução dos métodos UNIF/SLP (`true`/`false`). |
| `--run-twophase`| | `true` | Ativa/desativa a execução dos métodos de Duas Fases (`true`/`false`). |

**Mapeamento de Variantes (SLP e Two-Phase)**
Os argumentos abaixo aceitam `all` (testa ativado e desativado), `true` (apenas ativado) ou `false` (apenas desativado).

| Parâmetro | Flag SLP (`--slp-*`) | Flag Two-Phase (`--tp-*`) | Significado |
| :--- | :--- | :--- | :--- |
| **APR** | `--slp-apr` | `--tp-apr` | Razão de aceitação estrita (Ared/Pred Ratio). |
| **BQ** | `--slp-bq` | `--tp-bq` | Passo parabólico (Backtracking Quadratic). |
| **ATR** | `--slp-atr` | `--tp-atr` | Região de Confiança Anisotrópica. |
| **URU** | N/A | `--tp-uru` | Atualização geométrica da razão na restauração. |
| **SQP** | `--slp-sqp` | `--tp-sqp` | Estratégias Hessianas (`none`, `identity`, `spectral`, `exact` ou `all`). |

## Como Executar

### 1. Instalar Dependências
Na primeira utilização, execute o script de instalação para configurar o ambiente com as versões exatas definidas no `Manifest.toml`:

```bash
julia instalar_dependencias.jl

```

### 2. Rodar os Algoritmos

A execução dos problemas requer o ambiente Julia configurado e a interface CUTEst.jl instalada. A partir da raiz do repositório, utilize o comando abaixo. O parâmetro `--threads=auto` garante a utilização otimizada dos núcleos disponíveis.

```bash
julia --threads=auto --project=. src/Generic_main.jl -m filter -v -1 -c 1 --run-slp true --run-twophase true

```

### 3. Gerar Estatísticas e Gráficos

Para processar os dados tabulares e regerar planilhas e gráficos, utilize o módulo de estatísticas apontando para o diretório de arquivos brutos:

```julia
include("src/Generic_module_Stats.jl")
using .Generic_module_Stats

# O parâmetro deve indicar o diretório onde os arquivos CSV estão localizados.
Generic_module_Stats.run_statistical_analysis("data/")

```

