using CSV
using DataFrames

const SQP_MARKERS = ("_SQP_IDENTITY", "_SQP_SPECTRAL")
const BENCHMARK_FILES = ("benchmark_slp.csv", "benchmark_twophase.csv")

function remove_sqp_rows!(path)
	data = CSV.read(path, DataFrame)
	:Variant in propertynames(data) || error("CSV sem coluna Variant: $path")

	original_rows = nrow(data)
	filter!(data) do row
		!any(marker -> occursin(marker, string(row.Variant)), SQP_MARKERS)
	end

	temporary_path = path * ".tmp"
	CSV.write(temporary_path, data)
	mv(temporary_path, path; force=true)

	println("$(basename(path)): removidas $(original_rows - nrow(data)) linhas SQP")
end

for filename in BENCHMARK_FILES
	remove_sqp_rows!(joinpath(@__DIR__, filename))
end
