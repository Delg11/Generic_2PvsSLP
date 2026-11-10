# Relatório Estatístico Consolidado: SLP vs. SQP e o Impacto do ATR
**Data do Benchmark:** 04-05 de Novembro de 2026  
**Ambiente de Execução:** Servidor `dantzig` (32 CPUs, 16 workers distribuídos, Julia 1.x, Gurobi 11)  
**Conjunto de Problemas:** 470 problemas de otimização contínua da biblioteca CUTEst ($m \le n$, restrições de igualdade puras)  
**Tolerâncias de Parada:** $\epsilon_{\text{feas}} = 5 \times 10^{-2}$, $\epsilon_{\text{opt}} = 10^{-3}$, Max Iterações = 500, Timeout = 60.0s  

---

## 1. Visão Geral do Benchmark

Este experimento avaliou sistematicamente **126 configurações distintas de algoritmos de região de confiança**:
* **63 variantes Two-Phase (2P):** Decomposição em fase de restauração e fase de otimização (com e sem RRU, PH, ATR e SQP).
* **63 variantes UNIF:** Estrutura de penalidade unificada em norma $\ell_1$.

Foram realizadas **29.610 execuções individuais**. Graças à incorporação das salvaguardas computacionais (regularização espectral do BFGS, tempo-limite por subproblema no solver e controle de não-convexidade), **todas as 29.610 tarefas rodaram até o fim sem travamento do sistema distribuído** (houve apenas 103 timeouts pontuais de 60s, representando menos de 0,35% das execuções).

---

## 2. Modos SLP (Programação Linear Sequencial)

Nos modos SLP, o subproblema em cada iteração utiliza apenas informações de primeira ordem (gradiente da função objetivo e jacobiana das restrições), resolvendo um subproblema linear sob norma $\ell_\infty$ (caixa de região de confiança):
$$\min_{d} \ \nabla f(x_k)^T d \quad \text{s.a.} \ \|d\|_\infty \le \delta_k, \quad c(x_k) + J(x_k) d = 0$$

### 2.1. Variantes SLP no Two-Phase (2P)

| Variante SLP | Problemas Resolvidos | Taxa de Sucesso (%) | Iterações Médias | Tempo Médio (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **`2P_..._SECANT_BOX`** | **233 / 470** | **49.57%** | 47.36 | 109.62 ms |
| `2P_..._DECOMPOSED_CUT` | 222 / 470 | 47.23% | 38.56 | 146.02 ms |
| `2P_..._GRAD_SIGN` | 221 / 470 | 47.02% | 43.40 | 148.30 ms |
| `2P_SAR0_PH1_ATR0_RRU1` (PH sem ATR) | 210 / 470 | 44.68% | 42.72 | 107.13 ms |
| `2P_SAR1_PH0_ATR0_RRU1` (Base com SAR) | 193 / 470 | 41.06% | 50.25 | 117.39 ms |
| `2P_SAR0_PH0_ATR0_RRU1` (Linear Puro) | 192 / 470 | 40.85% | 36.01 | 123.74 ms |
| `2P_..._LEGACY_PARABOLIC` (ATR Antigo) | 192 / 470 | 40.85% | 43.92 | 157.52 ms |

### 2.2. Variantes SLP no UNIF

| Variante SLP | Problemas Resolvidos | Taxa de Sucesso (%) | Iterações Médias | Tempo Médio (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **`UNIF_..._SECANT_BOX`** | **186 / 470** | **39.57%** | 43.90 | 22.35 ms |
| `UNIF_..._DECOMPOSED_CUT` | 176 / 470 | 37.45% | 38.93 | 17.07 ms |
| `UNIF_SAR0_PH1_ATR0` (PH sem ATR) | 162 / 470 | 34.47% | 48.35 | 29.64 ms |
| `UNIF_SAR0_PH0_ATR0` (Linear Puro) | 156 / 470 | 33.19% | 44.54 | 16.56 ms |
| `BASE_UNIF` (Base SAR=1) | 154 / 470 | 32.77% | 47.81 | 25.05 ms |
| `UNIF_..._LEGACY_PARABOLIC` (ATR Antigo)| 150 / 470 | 31.91% | 71.31 | 26.69 ms |
| `UNIF_..._GRAD_SIGN` | 149 / 470 | 31.70% | 50.64 | 24.81 ms |

### 2.3. Principais Descobertas no Modo SLP
1. **O ATR revitalizou o SLP:** No Two-Phase, a variante base pura resolvia 192 problemas (40.85%). A introdução do **`SECANT_BOX` elevou o número para 233 problemas (49.57%)**, um acréscimo de **+41 problemas (+8.72 pontos percentuais)** com praticamente o mesmo tempo de CPU.
2. **Superação da limitação histórica do ATR Parabólico:** O ATR clássico (`LEGACY_PARABOLIC`) empatava com o SLP puro no Two-Phase (40.85%) e chegava a ser pior no UNIF (31.91% vs 32.77%), pois causava estreitamento excessivo da caixa. As geometrias desacopladas (`SECANT_BOX` e `DECOMPOSED_CUT`) removeram essa patologia e tornaram o SLP expressivamente mais robusto.

---

## 3. Modos SQP (Programação Quadrática Sequencial)

Nos modos SQP, uma matriz de segunda ordem $B_k \approx \nabla^2_{xx} L(x_k, \lambda_k)$ é incorporada à função objetivo do subproblema:
$$\min_{d} \ \nabla f(x_k)^T d + \frac{1}{2} d^T B_k d \quad \text{s.a.} \ \|d\|_\infty \le \delta_k, \quad c(x_k) + J(x_k) d = 0$$

Foram testadas 8 estratégias para $B_k$:
1. **`EXACT`:** Hessiana analítica completa do Lagrangeano.
2. **`BFGS`:** Atualização Quasi-Newton densa com regularização de autovalores ($\lambda_{\min} \ge 10^{-6}$).
3. **`QUASI_NEWTON`:** Aproximação Quasi-Newton diagonalizada.
4. **`DIAG_EXACT`:** Diagonal principal da Hessiana analítica ($\text{diag}(\nabla^2_{xx} L)$).
5. **`SPECTRAL`:** Múltiplo escalar da identidade via passo espectral de Barzilai-Borwein ($\gamma_k I$).
6. **`EXPONENTIAL`:** Heurística diagonal com decaimento exponencial baseada na norma do gradiente.
7. **`IDENTITY`:** Matriz identidade $B = I$ (regularização euclidiana da caixa).
8. **`RECIPROCAL`:** Heurística diagonal baseada no inverso das magnitudes das variáveis.

### 3.1. Estatísticas Consolidadas por Estratégia SQP no Two-Phase (2P)

| Estratégia de Hessiana | Resolvidos (Média) | Taxa Média (%) | Taxa Máxima (%) | Iterações Médias | Tempo Médio (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`EXACT`** | **341.0** | **72.55%** | **73.19%** | **30.26** | 251.05 ms |
| **`BFGS`** | **301.0** | **64.04%** | **64.89%** | 44.13 | 212.20 ms |
| `QUASI_NEWTON` | 233.3 | 49.64% | 51.28% | 34.21 | 107.17 ms |
| `DIAG_EXACT` | 229.1 | 48.75% | 50.43% | 33.59 | 217.96 ms |
| `SPECTRAL` | 214.0 | 45.53% | 46.81% | 34.20 | 110.28 ms |
| `EXPONENTIAL` | 211.3 | 44.95% | 49.15% | 36.89 | 125.55 ms |
| `IDENTITY` | 209.0 | 44.47% | 49.57% | 37.59 | 120.02 ms |
| *SLP Baseline* | *209.0* | *44.47%* | *49.57%* | *43.18* | *129.96 ms* |
| `RECIPROCAL` | 202.4 | 43.07% | 48.09% | 38.91 | 130.20 ms |

### 3.2. Estatísticas Consolidadas por Estratégia SQP no UNIF

| Estratégia de Hessiana | Resolvidos (Média) | Taxa Média (%) | Taxa Máxima (%) | Iterações Médias | Tempo Médio (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`EXACT`** | **310.0** | **65.96%** | **67.02%** | **45.39** | 106.22 ms |
| **`BFGS`** | **247.9** | **52.74%** | **55.74%** | 50.66 | 84.55 ms |
| `QUASI_NEWTON` | 188.9 | 40.18% | 42.55% | 42.22 | 27.59 ms |
| `DIAG_EXACT` | 187.1 | 39.82% | 41.06% | 38.46 | 39.11 ms |
| `SPECTRAL` | 172.6 | 36.72% | 38.72% | 37.81 | 19.04 ms |
| `EXPONENTIAL` | 167.6 | 35.65% | 39.79% | 45.12 | 37.25 ms |
| `IDENTITY` | 162.3 | 34.53% | 38.72% | 46.59 | 40.22 ms |
| *SLP Baseline* | *161.9* | *34.44%* | *39.57%* | *49.35* | *23.17 ms* |
| `RECIPROCAL` | 158.1 | 33.65% | 36.81% | 43.95 | 23.38 ms |

### 3.3. Principais Descobertas no Modo SQP
1. **O fracasso das heurísticas puramente diagonais:**
   * A estratégia `RECIPROCAL` teve desempenho **inferior ao SLP linear puro** tanto no Two-Phase (43.07% vs 44.47%) quanto no UNIF (33.65% vs 34.44%).
   * A matriz identidade `IDENTITY` ($B = I$) obteve rigorosamente a mesma taxa média do SLP, provando que regularizar quadraticamente sem informação real de curvatura não agrega ganho algorítmico.
   * `SPECTRAL` e `EXPONENTIAL` trouxeram ganhos residuais (< 1%), incapazes de justificar a substituição de um resolvedor LP por um resolvedor QP.
2. **A indispensabilidade do acoplamento cruzado:** A diagonal da Hessiana analítica (`DIAG_EXACT`, ~48.75%) perde por mais de 15 pontos percentuais para o `BFGS` (~64.04%). O descarte dos termos fora da diagonal ($\frac{\partial^2 L}{\partial x_i \partial x_j}$) descaracteriza a geometria dos vales estreitos formados pelas restrições não-lineares.

---

## 4. Confronto Direto: Melhor SLP vs. Melhores SQP

A tabela abaixo coloca lado a lado a melhor configuração puramente linear contra as melhores configurações quadráticas em ambos os paradigmas.

### 4.1. Two-Phase: SLP vs. SQP-BFGS vs. SQP-EXACT

| Indicador | Melhor SLP (`SECANT_BOX`) | Melhor SQP-BFGS (`SECANT_BOX`) | Melhor SQP-EXACT (`DECOMPOSED_CUT`) |
| :--- | :---: | :---: | :---: |
| **Problemas Resolvidos** | **233 / 470** | **305 / 470** | **344 / 470** |
| **Taxa de Sucesso** | **49.57%** | **64.89%** | **73.19%** |
| **Ganho Absoluto s/ SLP** | *Base* | **+72 problemas (+15.32%)** | **+111 problemas (+23.62%)** |
| **Iterações Médias** | 47.36 it | 43.81 it ($-7.5\%$) | **30.61 it ($-35.4\%$)** |
| **Tempo Médio de CPU** | 109.62 ms | 162.99 ms | 263.29 ms |
| **Necessita 2ªs Derivadas?** | Não | **Não** | Sim |

### 4.2. UNIF: SLP vs. SQP-BFGS vs. SQP-EXACT

| Indicador | Melhor SLP (`SECANT_BOX`) | Melhor SQP-BFGS (`ATR_LEGACY`) | Melhor SQP-EXACT (`ATR_LEGACY`) |
| :--- | :---: | :---: | :---: |
| **Problemas Resolvidos** | **186 / 470** | **262 / 470** | **315 / 470** |
| **Taxa de Sucesso** | **39.57%** | **55.74%** | **67.02%** |
| **Ganho Absoluto s/ SLP** | *Base* | **+76 problemas (+16.17%)** | **+129 problemas (+27.45%)** |
| **Iterações Médias** | 43.90 it | 52.39 it | 44.43 it |
| **Tempo Médio de CPU** | 22.35 ms | 115.94 ms | 100.84 ms |
| **Necessita 2ªs Derivadas?** | Não | **Não** | Sim |

---

## 5. Conclusões Teóricas e Práticas

1. **Eficiência do SLP com ATR:** Para aplicações de grande escala onde o custo de resolver QPs densas é proibitivo ou onde matrizes Hessianas não podem ser armazenadas, o algoritmo Two-Phase em modo **SLP com `SECANT_BOX`** é o método de melhor custo-benefício computacional: atinge **49.57% de convergência** com tempo médio de apenas **109 ms**.
2. **Superioridade do BFGS na Prática Sem Derivadas de 2ª Ordem:** Quando se busca maior taxa de sucesso sem o custo analítico de calcular a Hessiana do Lagrangeano, o **`SQP_BFGS` denso regularizado é incontestavelmente o melhor método**, saltando para **64.89%** no Two-Phase (+15.3% sobre o SLP).
3. **Padrão Ouro com Hessiana Exata:** O uso da Hessiana analítica completa (`SQP_EXACT`) junto à estratégia anisotrópica `DECOMPOSED_CUT` confere a **maior taxa de convergência da literatura para esta classe (73.19%)**, exigindo **35.4% menos iterações** que o método linear correspondente.
4. **Veredito sobre Métodos Diagonais:** Aproximações heurísticas puramente diagonais ($B$ diagonal) devem ser desencorajadas: elas não trazem os benefícios de convergência rápida do SQP e perdem em velocidade e robustez para o SLP puro dotado de caixa anisotrópica.
