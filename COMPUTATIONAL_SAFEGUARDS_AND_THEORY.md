# Aspectos Computacionais, Aritmética Finita e Preservação das Garantias Teóricas em Métodos de Região de Confiança (SLP/SQP)

> **Nota Editorial:** Este documento foi estruturado em formato de seção metodológica de tese/artigo científico para fundamentar as salvaguardas numéricas e operacionais implementadas no arcabouço de avaliação experimental (*benchmarking*) dos algoritmos **UNIF** e **Two-Phase (Dois Estágios)** sob a coleção CUTEst.

---

## 1. Contextualização: Da Aritmética Exata à Aritmética de Ponto Flutuante (IEEE 754)

Na formulação teórica de algoritmos de Região de Confiança baseados em Programação Linear Sequencial (SLP) e Programação Quadrática Sequencial (SQP), assume-se implicitamente que as operações algébricas e a resolução dos subproblemas locais ocorrem em aritmética real exata ($\mathbb{R}$) ou que os subproblemas quadráticos possuem solução obtida em tempo polinomial previsível.

Entretanto, quando um algoritmo é submetido a testes exaustivos de larga escala em coleções heterogêneas como a **CUTEst** — que inclui problemas propositalmente degenerados, funções com posto deficiente na origem (e.g., `ARGLINB`, `ARGLINC`), vales altamente curvos de baixa dimensão (e.g., `BT7`–`BT13`, `ALLINITC`) e problemas de mínimos quadrados mal-condicionados (e.g., `BA-L1LS`, `DIAMON2DLS`, `DMN15103LS`) — emerge uma lacuna inevitável entre a propriedade algébrica teórica e sua realização em aritmética de ponto flutuante de dupla precisão (**IEEE 754 `Float64`**, onde a precisão de máquina é $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$).

A ocorrência de estagnações computacionais ou bloqueios nos solucionadores de subproblemas nesses casos **não decorre de falha na teoria do algoritmo principal**, mas sim da interação entre três fenômenos estritamente computacionais:

1. **Cancelamento catastrófico e perda numérica de definição positiva** em matrizes densas de curvatura (BFGS amortecido e Hessiana exata regularizada);
2. **Tolerâncias relativas de fatoração de Cholesky e acionamento automático de *Spatial Branch-and-Bound*** em solucionadores comerciais de Programação Quadrática (como o Gurobi);
3. **Concorrência de *threads* e gerenciamento de memória C/Fortran** em execuções massivamente paralelas.

Nas subseções a seguir, detalha-se a natureza matemática de cada um desses fenômenos e demonstra-se rigorosamente por que as salvaguardas implementadas **preservam de forma integral a teoria de convergência global** dos métodos analisados.

---

## 2. Condicionamento da Matriz de Curvatura $B_k$ e Aritmética Finita

Nos subproblemas quadráticos da fase de otimização (tanto na variante UNIF-SQP quanto no algoritmo de Dois Estágios), busca-se um passo $s_k \in \mathbb{R}^n$ que minimize o modelo quadrático local sujeito à linearização das restrições e à região de confiança (isotrópica ou anisotrópica):

$$\min_{s \in \mathbb{R}^n} \; m_k(s) = \nabla f(x_k)^T s + \frac{1}{2} s^T B_k s \quad \text{sujeito a} \quad J_h(x_k) s + h(x_k) = 0, \quad \|D_k s\|_{\infty} \le \delta_k, \quad \ell \le x_k + s \le u$$

onde $B_k \in \mathbb{R}^{n \times n}$ é uma aproximação simétrica da Hessiana do Lagrangeano $\nabla^2_{xx} \mathcal{L}(x_k, \lambda_k)$.

### 2.1. O Fenômeno da Perda de Definição Positiva no BFGS Amortecido de Powell

Em teoria (aritmética exata), a atualização BFGS com amortecimento de Powell (1978),

$$B_{k+1} = B_k - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} + \frac{r_k r_k^T}{s_k^T r_k}, \qquad r_k = \theta_k y_k + (1 - \theta_k) B_k s_k,$$

preserva a definição estritamente positiva ($B_{k+1} \succ 0$) sempre que $B_k \succ 0$, visto que o parâmetro $\theta_k \in (0, 1]$ assegura $s_k^T r_k \ge 0.2 \, s_k^T B_k s_k > 0$.

Contudo, em aritmética finita (`Float64`), considere um problema cujo gradiente varia abruptamente em uma direção específica enquanto permanece quase constante nas demais (como ocorre em problemas de posto 1 do tipo `ARGLINB`/`ARGLINC` ou em restrições fortemente não-lineares como `BT8` e `ALLINITC`). Após poucas iterações:

1. O termo corretivo de posto 1, $(r_k r_k^T)/(s_k^T r_k)$, introduz um autovalor dominante $\lambda_{\max}(B_k) \sim 10^{12}$ a $10^{15}$, enquanto o subespaço ortogonal mantém autovalores da ordem da inicialização ($\lambda_{\min}(B_k) \sim 10^{-1}$).
2. O número de condição espectral atinge $\kappa(B_k) = \lambda_{\max}/\lambda_{\min} \gtrsim 10^{14}$.
3. Na iteração seguinte, a subtração matricial de posto 1, $B_k - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k)$, é avaliada com erro de arredondamento absoluto proporcional a $\epsilon_{\text{mach}} \cdot \|B_k\|_2 \approx 2.22 \times 10^{-16} \times 10^{14} \approx 2.2 \times 10^{-2}$.
4. Como a perturbação numérica ($\sim 10^{-2}$) supera os autovalores menores de $B_k$ ou a margem teórica de positividade de Schur, a matriz computada $\tilde{B}_{k+1}$ adquire **autovalores numericamente negativos** ($\lambda_{\min}(\tilde{B}_{k+1}) < 0$), violando em ponto flutuante uma propriedade que seria inviolável na matemática contínua.

### 2.2. Sensibilidade Escalar na Regularização da Hessiana Exata

Fenômeno análogo ocorre na estratégia de Hessiana exata (`:exact`) quando se aplica um deslocamento espectral puramente aditivo do tipo:

$$\tilde{B}_k = \nabla^2_{xx}\mathcal{L}(x_k, \lambda_k) + \left(|\lambda_{\min}| + \tau_{\text{abs}}\right) I, \qquad \tau_{\text{abs}} = 10^{-4}.$$

Se a Hessiana exata possuir autovalores negativos de grande magnitude (e.g., $\lambda_{\min} = -10^{10}$) e autovalores positivos igualmente elevados ($\lambda_{\max} = 10^{10}$, como observado em `BA-L1LS` e `BA-L1SPLS`), um deslocamento absoluto $\tau_{\text{abs}} = 10^{-4}$ produz uma razão espectral $\lambda_{\min}(\tilde{B}_k) / \lambda_{\max}(\tilde{B}_k) \approx 10^{-14}$. Para solucionadores quadráticos baseados em fatoração de Cholesky ou método de Barreira, matrizes cuja razão espectral cai abaixo da tolerância interna de pivoteamento ($\sim 10^{-9}$ a $10^{-11}$) são classificadas como **indefinidas (não-PSD)** ou numericamente singulares.

---

## 3. Interação com Solucionadores Comerciais (O Efeito *Spatial Branch-and-Bound*)

O impacto mais severo da perda numérica de definição positiva descrita na Seção 2 manifesta-se na camada do solucionador de subproblemas (especificamente o **Gurobi**).

Nas versões modernas do Gurobi ($\ge 9.5$), o parâmetro padrão `NonConvex = -1` (equivalente ao modo ativo `NonConvex = 2`) instrui o otimizador a **não abortar** quando a matriz quadrática $B_k$ falha no teste interno de semidefinição positiva. Em vez disso, o solucionador reinterpreta silenciosamente o subproblema local de região de confiança como um **Problema de Programação Quadrática Não-Convexa Global (Continuous MIQP)** e inicia uma árvore de busca espacial (*Spatial Branch-and-Bound* baseada em envelopes de McCormick e cortes RLT).

Do ponto de vista computacional, essa transição automática gera duas consequências críticas:

1. **Explosão de complexidade por subproblema:** Um subproblema local que deveria ser resolvido em milissegundos (ou rejeitado imediatamente para provocar a contração do raio da região de confiança $\delta_k$) passa a ser tratado como um problema NP-difícil de otimização global contínua, podendo permanecer horas explorando milhões de nós de ramificação dentro de uma única chamada de biblioteca C (`GRBoptimize`), mesmo para dimensões tão pequenas quanto $n = 4$ ou $n = 5$.
2. **Inoperância de *timeouts* na linguagem hospedeira:** Como a verificação de tempo limite do algoritmo (`time() - t_start > max_time`) ocorre no laço iterativo em Julia, uma chamada bloqueada dentro da rotina compilada em C sem um limite de tempo próprio (`TimeLimit`) no otimizador impede o retorno do controle ao algoritmo mestre, congelando permanentemente o processo trabalhador (*worker*).

---

## 4. Salvaguardas Implementadas e Preservação da Teoria de Convergência

Para eliminar completamente as patologias computacionais acima sem alterar a estrutura algorítmica dos métodos UNIF e Two-Phase, foram introduzidas salvaguardas numéricas e arquiteturais estritamente alinhadas à teoria clássica de Região de Confiança (Conn, Gould & Toint, 2000; Nocedal & Wright, 2006).

### 4.1. Regularização Espectral Relativa e Limitante de Gershgorin

Toda matriz não-diagonal $B_k$ (oriunda de `:exact` ou `:bfgs`) passa por um operador de regularização simétrica $\mathcal{R}(B_k)$ antes de ser enviada ao subproblema:

1. **Normalização de Escala Superior:** Se o raio espectral (para $n \le 500$) ou o limitante superior de Gershgorin $\max_i (|B_{ii}| + \sum_{j \neq i} |B_{ij}|)$ (para $n > 500$, calculado em $O(\text{nnz})$) exceder $M_{\max} = 10^8$, a matriz é reescalonada por $M_{\max} / \lambda_{\max}$, mantendo todas as entradas no mesmo intervalo seguro $[10^{-5}, 10^8]$ já adotado pelas aproximações diagonais (`:diag_exact`, `:reciprocal`, `:exponential`, `:quasi_newton`).
2. **Deslocamento de Positividade Relativo e Absoluto:** Garante-se que o menor autovalor (ou o limitante inferior do Teorema dos Círculos de Gershgorin, $\min_i (B_{ii} - \sum_{j \neq i} |B_{ij}|)$) satisfaça:
   $$\lambda_{\min}(\mathcal{R}(B_k)) \ge \max\left(\tau_{\text{abs}}, \; \tau_{\text{rel}} \cdot \max(0, \lambda_{\max})\right), \qquad \tau_{\text{abs}} = 10^{-4}, \quad \tau_{\text{rel}} = 10^{-6}.$$

### 4.2. Por que isso NÃO afeta (e de fato garante) a Teoria?

Na teoria de convergência global de métodos de Região de Confiança e SQP (ver *Conn, Gould & Toint, Trust-Region Methods, SIAM, 2000, Teoremas 6.3.4 e 15.4.1*; *Nocedal & Wright, 2006, Cap. 4 e 18*), **a convergência global de primeira ordem não exige que $B_k$ seja a Hessiana exata nem a matriz BFGS não-perturbada**.

A única hipótese teórica exigida sobre a sequência de matrizes $\{B_k\}_{k=0}^{\infty}$ para que o método mantenha suas propriedades de convergência global para pontos estacionários de Karush-Kuhn-Tucker (KKT) é a **Condição de Limitação Uniforme (Uniform Boundedness Condition)**:

$$\exists \, \beta_{\min}, \beta_{\max} \in (0, +\infty) \quad \text{tais que} \quad \|B_k\| \le \beta_{\max} \quad (\text{e, no caso convexo, } \beta_{\min} I \preceq B_k \preceq \beta_{\max} I), \quad \forall k \ge 0.$$

Portanto:
- **Sem a salvaguarda computacional:** O erro de ponto flutuante permite que $\|B_k\| \to \infty$ ou que $\lambda_{\min}(B_k)$ se torne negativo de forma descontrolada, **violando a hipótese teórica de limitação uniforme** e travando o solucionador quadrático.
- **Com a salvaguarda implementada:** A condição teórica $\beta_{\min} I \preceq B_k \preceq \beta_{\max} I$ (com $\beta_{\min} = 10^{-4}$ e $\beta_{\max} = 10^8$) passa a ser **rigorosamente satisfeita em todas as iterações em ponto flutuante**, reconciliando a implementação computacional com a hipótese exata dos teoremas de convergência.
- Além disso, quando $B_k$ já está bem condicionada (como ocorre próximo à solução ótima em problemas regulares onde a condição de curvatura de segunda ordem se verifica), o operador $\mathcal{R}(B_k)$ atua como a identidade ($\mathcal{R}(B_k) = B_k$), **preservando integralmente a taxa de convergência local superlinear/quadrática**.

### 4.3. Restrição de Convexidade Estrita (`NonConvex = 0`) e Limite de Tempo por Subproblema

No nível do solucionador de subproblemas (Gurobi), foram impostas três diretrizes operacionais:

1. **Desativação do *Spatial Branch-and-Bound* (`NonConvex = 0`):** O subproblema de região de confiança é, por definição metodológica, um modelo local convexo/linearizado. Caso ocorra qualquer instabilidade numérica residual, o solucionador deve reportar falha imediata em tempo $O(1)$, permitindo que o próprio algoritmo mestre execute sua regra teórica de recuo (contração do raio da região de confiança $\delta_k \leftarrow \tau \delta_k$ ou transição para a fase de restauração).
2. **Limite de Tempo Intrínseco por Subproblema (`TimeLimit`):** Cada chamada individual ao solucionador quadrático/linear recebe um teto estrito de tempo (limitado a $\min(5\text{s}, t_{\text{restante}})$), garantindo que o relógio global do algoritmo nunca seja bloqueado por chamadas externas em C.
3. **Isolamento de Threads (`Threads = 1` no Gurobi e no BLAS) e Coleta de Lixo Explícita (`GC.gc()`):** Em ambientes distribuídos com $W$ processos trabalhadores paralelos, fixar 1 *thread* por processo evita a sobre-alocação combinatória de $W \times T$ *threads* de sistema operacional e elimina o esgotamento de memória RAM (*Out-Of-Memory*) decorrente de estruturas alocadas na *heap* C/Fortran a cada problema avaliado.

---

## 5. Síntese para Citação na Tese / Artigo

Caso deseje incluir um parágrafo conciso na seção de **Configuração Experimental / Detalhes de Implementação** do seu texto acadêmico, pode utilizar diretamente a redação abaixo:

> *"Para assegurar a estabilidade numérica em aritmética de ponto flutuante (IEEE 754) ao longo de toda a coleção CUTEst, as aproximações matriciais densas e esparsas da Hessiana do Lagrangeano ($B_k$, nas variantes Exact e Damped-BFGS) são submetidas a uma salvaguarda de regularização espectral relativa que garante $\beta_{\min} I \preceq B_k \preceq \beta_{\max} I$, com $\beta_{\min} = \max(10^{-4}, 10^{-6}\lambda_{\max}(B_k))$ e $\beta_{\max} = 10^8$ (avaliada via decomposição espectral para $n \le 500$ e via Teorema dos Círculos de Gershgorin para $n > 500$). Do ponto de vista teórico, tal salvaguarda assegura o cumprimento estrito da hipótese clássica de limitação uniforme de $\|B_k\|$ exigida para a convergência global de métodos de Região de Confiança (Conn et al., 2000), ao mesmo tempo em que previne que erros de cancelamento numérico acionem rotinas de ramificação espacial não-convexa no solucionador de subproblemas quadráticos (Gurobi)."*

