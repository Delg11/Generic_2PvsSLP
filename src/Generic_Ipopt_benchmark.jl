using CUTEst
using NLPModelsIpopt
using NLPModels
using LinearAlgebra

function run_ipopt_benchmark()
    println("Buscando problemas no CUTEst (max_var=Inf, max_con=1)...")
    problems = select_sif_problems(max_var=Inf, max_con=1)
    
    println("Total de problemas selecionados: ", length(problems))
    
    output_file = "ipopt_benchmark_fstar.csv"
    println("Salvando resultados em: $output_file")
    println("-"^60)
    
    open(output_file, "w") do file
        # Cabeçalho padrão CSV
        write(file, "Problem,f_star,h_norm,Status,Iter,Time_s\n")
        
        for prob_name in problems
            local nlp = nothing
            try
                nlp = CUTEstModel{Float64}(prob_name)
                
                dim = nlp.meta.nvar
                ncon = nlp.meta.ncon
                println("Processando: $prob_name [Var: $dim | Con: $ncon]")
                
                # Execução do solver
                tempo = @elapsed begin
                    stats = ipopt(nlp, print_level=0)
                end
                
                # Extração dos dados
                f_star = stats.objective
                status = string(stats.status)
                iter = stats.iter
                
                # Cálculo da violação das restrições
                x_star = stats.solution
                h_val = cons(nlp, x_star)
                h_norm = length(h_val) > 0 ? norm(h_val) : 0.0
                
                # Formatação CSV delimitada por vírgula
                linha = "$prob_name,$f_star,$h_norm,$status,$iter,$tempo"
                        
                # Escrita com flush para backup incremental em disco
                write(file, linha * "\n")
                flush(file)
                
            catch e
                println("Erro no problema $prob_name: $e")
                write(file, "$prob_name,ERRO,ERRO,ERRO,ERRO,ERRO\n")
                flush(file)
            finally
                isnothing(nlp) || finalize(nlp)
            end
        end
    end
    
    println("-"^60)
    println("Execução finalizada. Valores registrados em: $output_file")
end

run_ipopt_benchmark()