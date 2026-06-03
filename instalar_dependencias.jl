using Pkg

# Ativa o ambiente no diretório onde este arquivo está salvo
Pkg.activate(@__DIR__)

# Lista apenas pacotes externos
pacotes_externos = [
    "CUTEst",
    "DataFrames",
    "XLSX",
    "Plots",
    "StatsPlots",
    "CSV",
    "ADNLPModels",
    "NLPModels",
    "JuMP",
    "Gurobi"
]

println("Iniciando o download e registro dos pacotes no Project.toml...")
Pkg.add(pacotes_externos)

println("Baixando versões exatas e travando no Manifest.toml...")
Pkg.instantiate()

println("Pré-compilando dependências (isso pode levar alguns minutos)...")
Pkg.precompile()

println("✅ Ambiente configurado com sucesso! Project.toml e Manifest.toml foram atualizados.")