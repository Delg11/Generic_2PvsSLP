================================================================================
METODOLOGIAS DE AVALIAÇÃO DOS PERFIS DE DESEMPENHO (DOLAN-MORÉ) - DIAGNÓSTICO
================================================================================

1. Pasta '1_Status_KKT_OK':
   - Critério: Convergência Formal KKT (Status == "KKT_OK" com viabilidade satisfeita).
   - O patamar final (τ -> ∞) reflete rigorosamente a taxa de sucesso no atendimento
     das condições de otimalidade de primeira ordem KKT.
   - Corresponde 1:1 à métrica 'Solved_KKT' do sumário estatístico.

2. Pasta '2_Melhor_Objetivo_f':
   - Critério: Qualidade Global da Solução (Melhor f(x*) viável encontrada).
   - Um solver pontua no problema se for viável (||h(x*)|| <= 1e-5) e encontrar
     o menor valor da função objetivo f(x*) entre todos os concorrentes viáveis daquele grupo
     (dentro de tolerância relativa/absoluta de 1e-5).
   - O patamar final (τ -> ∞) indica em quantos problemas o algoritmo encontrou a
     melhor solução viável do torneio.
================================================================================
