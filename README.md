# Generic_2PvsSLP: UNIF and Two-Phase Trust-Region Benchmarking

Este repositório contém o arcabouço computacional e estatístico desenvolvido para avaliar o desempenho de algoritmos de Programação Não-Linear (PNL), comparando abordagens Uniformes (UNIF / SLP Clássico e extensões SQP) contra o método de Região de Confiança de Duas Fases (Two-Phase Trust-Region).

O projeto realiza uma avaliação modular rigorosa sobre o conjunto de problemas da biblioteca CUTEst, investigando o impacto de passos parabólicos (PH), regiões de confiança anisotrópicas (ATR), regras de concordância estrita (SAR), atualizações de razão de restauração (RRU) e aproximações quase-Newton da Hessiana do Lagrangiano (SQP).

📊 **[Acesse o Dashboard Interativo de Perfis de Desempenho (Dolan-Moré)](https://delg11.github.io/Generic_2PvsSLP/)**

---

## Estrutura do Repositório

```text
Generic_2PvsSLP/
├── src/
│   ├── Generic_main.jl            # Ponto de entrada CLI e orquestrador principal de benchmarks
│   ├── Generic_Sharedtypes.jl     # Tipos comuns, parâmetros, modelos JuMP compartilhados e GP-norm
│   ├── Generic_module_Twophase.jl # Solucionador de Região de Confiança de Duas Fases
│   ├── Generic_module_unif.jl     # Solucionador UNIF (SLP e SQP com região de confiança)
│   ├── Generic_module_ComputeB.jl # Métodos de atualização quase-Newton (BFGS, SR1, espectral)
│   ├── worker_env.jl              # Gerenciador de workers distribuídos e JIT warmup compartilhado
│   ├── reproducibility.jl         # Módulo de proveniência e metadados científicos (Git, CPU, Solvers)
│   ├── run_fast_benchmark.jl      # Harness de regressão rápida e validação matemática contínua
│   ├── Generic_module_Stats.jl    # Pipeline de consolidação estatística e perfis de desempenho
│   └── stats/                     # Scripts estatísticos especializados para tese e artigos
│       ├── stats_sbpo.jl          # Análise descritiva para o congresso SBPO
│       ├── stats_tese_cap3_fprofile.jl # Perfis de função objetivo (Capítulo 3)
│       └── stats_tese_cap4_sqp.jl # Perfis de desempenho comparativo SQP (Capítulo 4)
├── benchmarks/
│   ├── fast_baseline.csv          # Baseline de referência para testes de conformidade matemática
│   └── Caso_SQP_CUTEST/          # Resultados consolidados e figuras de perfis de desempenho
├── data/                          # Dados brutos de execuções (.csv)
├── analysis/                      # Planilhas (.xlsx) e gráficos gerados (.png)
├── index.html                     # Dashboard interativo em React/Plotly.js (GitHub Pages)
├── instalar_dependencias.jl       # Script automatizado de configuração do ambiente
├── Project.toml / Manifest.toml   # Fixação exata de dependências do ecossistema Julia
└── README.md                      # Documentação completa do projeto
```

---

## Características Algorítmicas e Correções Teóricas

O repositório implementa rigorosamente a teoria clássica de métodos de região de confiança para PNL com restrições de igualdade e canalização em caixa:

1. **Two-Phase Trust Region:**
   - **Fase de Restauração:** Minimização de infeabilidade via subproblema linear/quadrático com relaxações L1.
   - **Fase de Otimização:** Passo tangencial com modelo quadrático ou linear restrito ao subespaço aproximado.
   - **Função de Mérito com Salvaguarda:** Parâmetro de penalidade $\theta$ estritamente limitado por $\theta = \text{clamp}(\text{numerador} / \text{denominador},\, \theta_{\min},\, \theta_k)$ com $\theta_{\min} = 10^{-6}$, impedindo valores negativos que inverteriam o sentido de otimização.

2. **UNIF (SLP & SQP Trust Region):**
   - **Vetor Secante da Hessiana do Lagrangiano:** Em problemas com restrições ($m > 0$), o vetor secante quasi-Newton inclui a variação das restrições ponderadas pelos multiplicadores:
     $$y_k = (\nabla f(x_{k+1}) - \nabla f(x_k)) + (J(x_{k+1}) - J(x_k))^T \lambda$$
     assegurando convergência assintótica superlinear em formulações SQP.
   - **Critério de Parada e Viabilidade Instantânea:** A viabilidade das restrições $\|h(x)\|_{\infty} \le 10^{-3}$ é avaliada dinamicamente no ponto recém-aceito, prevenindo diagnósticos falsos de infactibilidade.

3. **Mapeamento de Variantes (Dimensões Algorítmicas):**
   - **SAR (Strong Agreement Rule):** Expansão da região de confiança condicionada à razão $\rho = \text{ared} / \text{pred}$.
   - **PH (Parabolic Heuristic):** Ajuste fino do raio $\delta$ através de interpolação parabólica unidimensional ao longo da direção do passo.
   - **ATR (Anisotropic Trust Region):** Vetor de raios $\vec{\delta} \in \mathbb{R}^n$ permitindo dilatações e contrações independentes por coordenada com salvaguarda de aspecto (`enforce_aspect_ratio!`).
   - **RRU (Restoration Ratio Update):** Atualização dinâmica da taxa de contração exigida na fase de restauração.
   - **SQP (Estratégias de Curvatura):** Suporte a `identity`, `spectral`, `exact` e atualizações de posto 1 e 2 via `Generic_module_ComputeB.jl`.

---

## Reprodutibilidade Científica (Artigos e Tese)

Para atender aos mais altos padrões de reprodutibilidade em periódicos internacionais e tese de doutorado, cada execução registra automaticamente metadados completos de proveniência através do módulo [`src/reproducibility.jl`](file:///c:/Github/Generic_2PvsSLP/src/reproducibility.jl):

- **Controle de Versão (Git):** Commit SHA completo, branch ativa, e flag booleana de estado da árvore de trabalho (`git_dirty`).
- **Ambiente de Hardware:** Modelo do processador (CPU), quantidade de núcleos lógicos e memória física.
- **Ambiente de Software:** Versão do Julia, Sistema Operacional (OS), arquitetura e contagem de threads/workers.
- **Versões dos Solvers:** Versões exatas dos pacotes `Gurobi.jl`, `RipQP.jl`, `CUTEst.jl` e `JuMP.jl`.

Ao rodar via `Generic_main.jl`, os metadados são injetados diretamente nas colunas dos CSVs gerados e gravados em arquivos de manifesto `run_metadata.json` e `run_environment.txt` na pasta de benchmark correspondente.

---

## Paralelização e Gerenciamento de Workers

O framework possui balanceamento de carga paralelo de granulação fina (`src/worker_env.jl`):

- **Distribuição Granular:** As tarefas são despachadas por par `(problema, variante)` em vez de apenas por problema, garantindo ocupação máxima de todos os workers (100% de utilização de CPU mesmo em baterias pequenas de teste).
- **Aquecimento de JIT Transparente:** Cada worker inicializado compila previamente os caminhos críticos (resolvendo instâncias de aquecimento com `CUTEstModel("HS6")` e subproblemas JuMP). Isso elimina o tempo artificial de compilação JIT da primeira iteração dos benchmarks.
- **Sessão Interativa no REPL:** Em sessões longas de desenvolvimento ou via SSH, os workers podem ser mantidos vivos na memória do REPL para rodar múltiplos experimentos sem pagar o custo de inicialização repetidamente.

---

## Guia de Execução

### 1. Instalação de Dependências

Certifique-se de ter o Julia 1.10+ instalado com o CUTEst configurado no sistema:

```bash
julia instalar_dependencias.jl
```

### 2. Validação Rápida e Teste de Regressão Matemática

Antes de submeter grandes baterias de experimentos, execute o script de validação rápida (conjunto de 12 problemas com restrições e sem restrições, 48 execuções totais):

```bash
# Comparar a execução atual contra o baseline matemático de referência
julia --project=. src/run_fast_benchmark.jl --compare-baseline

# Atualizar o baseline com a versão atual (se mudanças matemáticas forem aprovadas)
julia --project=. src/run_fast_benchmark.jl --save-baseline
```

### 3. Execução Principal de Benchmarks (`Generic_main.jl`)

O script `Generic_main.jl` suporta execução sequencial ou paralela (especificando `-p <nworkers>`):

```bash
# Executar paralelamente com 4 workers no modo 'test' para os problemas HS6 e TRY-B
julia -p 4 --project=. src/Generic_main.jl -m test -p HS6,TRY-B

# Executar bateria completa filtrada do CUTEst (problemas com até 1 restrição)
julia -p 8 --project=. src/Generic_main.jl -m filter -v -1 -c 1 --run-unif true --run-twophase true
```

#### Parâmetros de Linha de Comando (CLI)

| Argumento | Atalho | Padrão | Descrição |
| :--- | :---: | :--- | :--- |
| `--mode` | `-m` | `test` | Modo de execução: `test` (problemas específicos) ou `filter` (bateria CUTEst). |
| `--problems` | `-p` | `HS6` | Lista separada por vírgula dos problemas a executar no modo `test`. |
| `--max-var` | `-v` | `-1` | Número máximo de variáveis permitidas no modo `filter` (-1 para sem limite). |
| `--max-con` | `-c` | `-1` | Número máximo de restrições permitidas no modo `filter` (-1 para sem limite). |
| `--run-unif` | | `true` | Ativa/desativa os solucionadores UNIF/SLP/SQP (`true`/`false`). |
| `--run-twophase`| | `true` | Ativa/desativa os solucionadores Two-Phase (`true`/`false`). |
| `--unif-sar` | | `all` | Regra de Concordância Estrita no UNIF (`true`, `false`, `all`). |
| `--unif-ph`  | | `all` | Heurística Parabólica no UNIF (`true`, `false`, `all`). |
| `--unif-atr` | | `all` | Região de Confiança Anisotrópica no UNIF (`true`, `false`, `all`). |
| `--unif-sqp` | | `none` | Estratégia de aproximação Hessiana no UNIF (`none`, `identity`, `spectral`, `exact`). |
| `--tp-sar`   | | `all` | Regra de Concordância Estrita no Two-Phase (`true`, `false`, `all`). |
| `--tp-ph`    | | `all` | Heurística Parabólica no Two-Phase (`true`, `false`, `all`). |
| `--tp-atr`   | | `all` | Região de Confiança Anisotrópica no Two-Phase (`true`, `false`, `all`). |
| `--tp-rru`   | | `true` | Atualização da Razão de Restauração no Two-Phase (`true`, `false`, `all`). |
| `--tp-sqp`   | | `none` | Estratégia de aproximação Hessiana no Two-Phase (`none`, `identity`, `spectral`, `exact`). |

---

### 4. Geração de Estatísticas e Perfis de Desempenho

Após a conclusão dos benchmarks, consolide os dados e gere as figuras de Perfis de Desempenho (Dolan-Moré):

```julia
include("src/Generic_module_Stats.jl")
using .Generic_module_Stats

# Processa os CSVs brutos e gera as planilhas e gráficos em analysis/
Generic_module_Stats.run_statistical_analysis("data/")
```

Para reproduzir os perfis específicos de capítulos de tese ou artigos, execute os scripts dedicados em `src/stats/`:
- `julia --project=. src/stats/stats_tese_cap3_fprofile.jl`
- `julia --project=. src/stats/stats_tese_cap4_sqp.jl`
- `julia --project=. src/stats/stats_sbpo.jl`
