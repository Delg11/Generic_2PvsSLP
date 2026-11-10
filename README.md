# Generic_2PvsSLP: UNIF and Two-Phase Trust-Region Benchmarking

Este repositório contém o arcabouço computacional e estatístico desenvolvido para avaliar o desempenho de algoritmos de Programação Não-Linear (PNL), comparando abordagens Uniformes (**UNIF** / SLP Clássico e extensões SQP) contra o método de Região de Confiança de Duas Fases (**Two-Phase Trust-Region** / Restauração Inexata).

O projeto realiza uma avaliação modular rigorosa sobre o conjunto de problemas da biblioteca CUTEst, investigando o impacto de passos parabólicos (**PH**), regiões de confiança anisotrópicas (**ATR**), regras de concordância estrita (**SAR**), atualizações de razão de restauração (**RRU**) e aproximações quase-Newton da Hessiana do Lagrangiano (**SQP**).

📊 **[Acesse o Dashboard Interativo de Perfis de Desempenho (Dolan-Moré)](https://delg11.github.io/Generic_2PvsSLP/)**

---

## Fundamentação Teórica e Algorítmica

Os solucionadores implementados baseiam-se em dois modelos teóricos fundamentais da literatura de otimização contínua:

1. **Abordagem Unificada (UNIF - SLP/SQP):**
   - **Referência:** Gomes, F. A. M., & Senne, T. A. (2011). *An SLP algorithm and its application to topology optimization*. Computational & Applied Mathematics, 30(1), 53–89. [DOI: 10.1590/S1807-03022011000100004](https://doi.org/10.1590/S1807-03022011000100004).
   - Resolve simultaneamente viabilidade e otimização por meio de uma função de mérito $L_1$ com balanço $\theta$, acionando fase de recuperação de viabilidade quando a linearização local for infactível.

2. **Abordagem de Duas Fases (Two-Phase / Inexact Restoration):**
   - **Referência:** Birgin, E. G., Bueno, L. F., Martini, T., Moreno, D., Senne, T. A., & Siqueira, T. (2026). *A general merit function-based global convergent framework for nonlinear optimization*. Computational Optimization and Applications. [DOI: 10.1007/s10589-026-00804-1](https://doi.org/10.1007/s10589-026-00804-1).
   - Desacopla a busca a cada iteração:
     - **Fase de Restauração:** Minimização de infeabilidade $\|h(x)\|$ até atingir o decréscimo proporcional exigido.
     - **Fase de Otimização:** Passo tangencial minimizando o modelo da função objetivo restrito ao subespaço tangente linearizado.

---

## Estrutura do Repositório

```text
Generic_2PvsSLP/
├── src/
│   ├── Generic_main.jl            # Ponto de entrada CLI e orquestrador principal de benchmarks
│   ├── Generic_Sharedtypes.jl     # Tipos comuns, parâmetros, modelos JuMP compartilhados e GP-norm
│   ├── Generic_module_Twophase.jl # Solucionador Two-Phase (Birgin et al., 2026)
│   ├── Generic_module_unif.jl     # Solucionador UNIF (Gomes & Senne, 2011)
│   ├── Generic_module_ComputeB.jl # Métodos de atualização quase-Newton (BFGS, SR1, espectral)
│   ├── worker_env.jl              # Gerenciador de workers distribuídos e JIT warmup compartilhado
│   ├── reproducibility.jl         # Módulo de proveniência e metadados científicos (Git, CPU, Solvers)
│   ├── run_fast_benchmark.jl      # Harness de regressão rápida e validação matemática contínua
│   ├── Generic_module_Stats.jl    # Pipeline de consolidação estatística e perfis de desempenho
│   └── stats/                     # Scripts estatísticos especializados para artigos e teses
│       ├── stats_sbpo.jl          # Análise descritiva para o congresso SBPO
│       ├── stats_tese_cap3_fprofile.jl # Perfis de função objetivo (Capítulo 3)
│       └── stats_tese_cap4_sqp.jl # Perfis de desempenho comparativo SQP (Capítulo 4)
├── scripts/
│   ├── run_cutest_comparison_extended.jl # Comparação estendida com problemas didáticos
│   ├── generate_cutest_baseline_trajectories.jl # Geração de trajetórias iterativas
│   ├── generate_ipopt_comparison.jl      # Comparação contra Ipopt (baseline)
│   └── generate_rosenbrock_videos.jl    # Animações de trajetórias de convergência
├── benchmarks/                    # Saídas de benchmarks (CSVs consolidados, manifestos)
├── analysis/                      # Planilhas (.xlsx) e gráficos gerados (.png)
├── data/                          # Dados brutos de execuções
├── run_cutest_benchmark.sh        # Script automatizado para execução no servidor via SSH
├── CITATION.bib                   # Citações acadêmicas em formato BibTeX
├── COMPUTATIONAL_SAFEGUARDS_AND_THEORY.md # Fundamentação teórica das salvaguardas numéricas
├── index.html                     # Dashboard interativo em React/Plotly.js (GitHub Pages)
├── instalar_dependencias.jl       # Script automatizado de configuração do ambiente
├── Project.toml / Manifest.toml   # Fixação exata de dependências do ecossistema Julia
└── README.md                      # Documentação completa do projeto
```

---

## Escopo Matemático e Restrições Estruturais dos Problemas

O repositório é projetado para classes de Programação Não-Linear com **canalização em caixa** e **restrições de igualdade pura**:
$$\min_{x \in \mathbb{R}^n} f(x) \quad \text{sujeito a} \quad h(x) = 0, \quad x_l \le x \le x_u$$

### 1. Requisito de Posto e Restrições de Igualdade ($m < n$ ou $m = 0$)
- **Problemas Válidos:**
  - Irrestritos ou com cotas em caixa ($m = 0$).
  - Restrições de igualdade pura com $m < n$ satisfazendo a Qualificação de Independência Linear das Restrições (LICQ), ou seja, $\text{posto}(J(x)) = m$.
- **Problemas Fora do Escopo:**
  - **Sistemas Sobre-Determinados ($m \ge n$):** Formulações como problemas de mínimos quadrados não-lineares da biblioteca SIF com centenas de equações de resíduos para poucas variáveis não possuem graus de liberdade para otimização sob igualdade pura e são automaticamente descartados pelo filtro.
  - **Restrições de Desigualdade Gerais:** Devem ser previamente reformuladas com variáveis de folga antes da aplicação direta destes algoritmos.

### 2. Condição Estrutural da Projeção de Dykstra (GP-Norm e Gradiente Tangente)
A função [`compute_gradient_projection`](file:///c:/Github/Generic_2PvsSLP/src/Generic_Sharedtypes.jl) utiliza o algoritmo iterativo de Dykstra para projetar simultaneamente nas cotas de caixa $[x_l, x_u]$ e na variedade tangente linearizada $\ker(J(x)) = \{ d \in \mathbb{R}^n : J(x) d = 0 \}$.

---

## Características Algorítmicas e Dimensões de Avaliação

1. **Two-Phase Trust Region:**
   - **Fase de Restauração:** Minimização de infeabilidade via subproblema linear/quadrático com relaxações $L_1$.
   - **Fase de Otimização:** Passo tangencial com modelo quadrático ou linear restrito ao subespaço aproximado.
   - **Função de Mérito com Salvaguarda:** Parâmetro de penalidade $\theta$ estritamente limitado por $\theta = \text{clamp}(\text{numerador} / \text{denominador},\, \theta_{\min},\, \theta_k)$ com $\theta_{\min} = 10^{-6}$.

2. **UNIF (SLP & SQP Trust Region):**
   - **Vetor Secante da Hessiana do Lagrangiano:** Em problemas com restrições ($m > 0$), o vetor secante quasi-Newton inclui a variação das restrições ponderadas pelos multiplicadores:
     $$y_k = (\nabla f(x_{k+1}) - \nabla f(x_k)) + (J(x_{k+1}) - J(x_k))^T \lambda$$
   - **Critério de Parada e Viabilidade Instantânea:** Avaliação dinâmica no ponto recém-aceito para $\|h(x)\|_{\infty} \le 10^{-3}$.

3. **Mapeamento de Variantes (Dimensões Algorítmicas):**
   - **SAR (Strong Agreement Rule):** Expansão da região de confiança condicionada à razão $\rho = \text{ared} / \text{pred}$.
   - **PH (Parabolic Heuristic):** Ajuste fino do raio $\delta$ através de interpolação parabólica unidimensional ao longo da direção do passo.
   - **ATR (Anisotropic Trust Region):** Vetor de raios $\vec{\delta} \in \mathbb{R}^n$ com estratégias:
     - `legacy_parabolic`: Curvatura por coordenada via passo parabólico (Gomes & Senne, 2011).
     - `grad_sign`: Baseada na concordância de sinais do gradiente.
     - `decomposed_cut`: Cortes decompostos por coordenada.
     - `secant_box`: Estimativa de curvatura por coordenadas via vetor secante.
   - **RRU (Restoration Ratio Update):** Atualização dinâmica da taxa de contração exigida na fase de restauração.
   - **SQP (Estratégias de Curvatura):** Suporte a `none` (SLP puro) e aproximações quadráticas: `exact`, `diag_exact`, `bfgs`, `spectral`, `identity`, `quasi_newton`, `reciprocal`, `exponential`.

---

## Guia de Execução

### 1. Instalação de Dependências

```bash
julia instalar_dependencias.jl
```

### 2. Validação Rápida e Teste de Regressão Matemática

Antes de submeter baterias longas, execute a verificação rápida:

```bash
# Executa nos problemas didáticos e compara com a referência
julia --project=. src/run_fast_benchmark.jl
```

### 3. Execução Remota via SSH (`run_cutest_benchmark.sh`)

Para executar em servidores Linux via SSH com tolerância a desconexões e paralelismo configurado:

```bash
chmod +x run_cutest_benchmark.sh

# Teste rápido de 10 problemas (HS6, ROSENBR, etc.):
./run_cutest_benchmark.sh test 4

# Bateria completa CUTEst com SLP puro (n <= 200, m <= 100):
./run_cutest_benchmark.sh slp 16

# Bateria completa massiva (SLP + SQP + todas as variantes ATR):
nohup ./run_cutest_benchmark.sh all 24 > /dev/null 2>&1 &
```

### 4. Execução Direta via CLI (`Generic_main.jl`)

```bash
# Executar paralelamente com 8 workers no modo filtro CUTEst:
julia --project=. -p 8 src/Generic_main.jl -m filter -v 200 -c 100 -i 500 \
    --tp-sar all --tp-ph all --tp-atr all --tp-atr-strat all --tp-rru all --tp-sqp none \
    --unif-sar all --unif-ph all --unif-atr all --unif-atr-strat all --unif-sqp none
```

#### Parâmetros de Linha de Comando (CLI)

| Argumento | Atalho | Padrão | Descrição |
| :--- | :---: | :---: | :--- |
| `--mode` | `-m` | `test` | Modo: `test` (problemas fixos), `small` (didáticos) ou `filter` (bateria CUTEst). |
| `--problems` | `-p` | `HS6` | Lista separada por vírgula dos problemas a executar no modo `test`. |
| `--workers` | `-w` | `1` | Número de workers (`1` sequencial, `0` auto-detecta CPU threads, `N > 1` paralelo). |
| `--max-var` | `-v` | `-1` | Número máximo de variáveis no modo `filter` (-1 para sem limite). |
| `--max-con` | `-c` | `-1` | Número máximo de restrições no modo `filter` (-1 para sem limite). |
| `--max-iter` | `-i` | `500` | Máximo de iterações externas por variante. |
| `--timeout` | `-t` | `60.0` | Tempo limite em segundos por variante/problema. |
| `--run-unif` | | `true` | Ativa/desativa os solucionadores UNIF/SLP/SQP (`true`/`false`). |
| `--run-twophase`| | `true` | Ativa/desativa os solucionadores Two-Phase (`true`/`false`). |
| `--unif-sar` | | `all` | Regra de Concordância Estrita no UNIF (`true`, `false`, `all`). |
| `--unif-ph`  | | `all` | Heurística Parabólica no UNIF (`true`, `false`, `all`). |
| `--unif-atr` | | `all` | Região de Confiança Anisotrópica no UNIF (`true`, `false`, `all`). |
| `--unif-atr-strat` | | `legacy_parabolic` | Estratégias ATR no UNIF (`legacy_parabolic`, `grad_sign`, `decomposed_cut`, `secant_box`, `all`). |
| `--unif-sqp` | | `none` | Estratégia de Hessiana no UNIF (`none`, `quadratic`, `all`, ou lista). |
| `--tp-sar`   | | `all` | Regra de Concordância Estrita no Two-Phase (`true`, `false`, `all`). |
| `--tp-ph`    | | `all` | Heurística Parabólica no Two-Phase (`true`, `false`, `all`). |
| `--tp-atr`   | | `all` | Região de Confiança Anisotrópica no Two-Phase (`true`, `false`, `all`). |
| `--tp-atr-strat` | | `legacy_parabolic` | Estratégias ATR no Two-Phase (`legacy_parabolic`, `grad_sign`, `decomposed_cut`, `secant_box`, `all`). |
| `--tp-rru`   | | `all` | Atualização da Razão de Restauração no Two-Phase (`true`, `false`, `all`). |
| `--tp-sqp`   | | `none` | Estratégia de Hessiana no Two-Phase (`none`, `quadratic`, `all`, ou lista). |
| `--qp-solver`| | `gurobi` | Solucionador para subproblemas quadráticos (`gurobi` ou `ripqp`). |

---

### 5. Geração de Estatísticas e Perfis de Desempenho

Após a conclusão dos benchmarks, os gráficos e tabelas são gerados automaticamente pelo script shell. Caso queira processar manualmente:

```bash
julia --project=. src/min_run_stats.jl
```

Ou diretamente pelos scripts dedicados em `src/stats/`:
- `julia --project=. src/stats/stats_sbpo.jl` (Perfis dos Grupos G1 a G6 e Resumo Expandido)
- `julia --project=. src/stats/stats_tese_cap3_fprofile.jl` (Perfis baseados em valor de função viável)
- `julia --project=. src/stats/stats_tese_cap4_sqp.jl` (Perfis comparativos das variantes SQP)

---

## Citação Acadêmica

Se este código ou seus dados forem úteis em sua pesquisa, cite as referências teóricas correspondentes (disponíveis em [`CITATION.bib`](file:///c:/Github/Generic_2PvsSLP/CITATION.bib)):

```bibtex
@article{senne:09,
  title={An SLP algorithm and its application to topology optimization},
  author={Gomes, Francisco A. M. and Senne, Thadeu A.},
  journal={Computational \& Applied Mathematics},
  volume={30},
  number={1},
  pages={53--89},
  year={2011},
  doi={10.1590/S1807-03022011000100004}
}

@article{birgin:24,
  title={A general merit function-based global convergent framework for nonlinear optimization},
  author={Birgin, Ernesto G. and Bueno, Lu{\'i}s Felipe and Martini, Tiara and Moreno, Dimary and Senne, Thadeu A. and Siqueira, Thiago},
  journal={Computational Optimization and Applications},
  year={2026},
  doi={10.1007/s10589-026-00804-1}
}
```
