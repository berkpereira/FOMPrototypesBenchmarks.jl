# ————————————————————————————————————————————————
# A) List *only* the solver‐defining override keys here
# ————————————————————————————————————————————————
# 0) A little helper to build each override dict
function make_override(variant; acceleration=:none,
	accel_memory=nothing,
	krylov_tries_per_mem=1,
	anderson_interval=10,
	anderson_broyden_type=:QR2,
	anderson_mem_type=:restarted)

	d = Dict{String,Any}()
	d["variant"]      = variant
	d["acceleration"] = acceleration
	if accel_memory !== nothing
		d["accel-memory"] = accel_memory
	end
	# only relevant for Anderson
	if acceleration === :anderson
		d["anderson-interval"]     = anderson_interval
		d["anderson-broyden-type"] = anderson_broyden_type
		d["anderson-mem-type"]     = anderson_mem_type
	end
	# only relevant for Krylov
	if acceleration === :krylov
		d["krylov-tries-per-mem"] = krylov_tries_per_mem
	end
	return d
end

# bench_type in {:fom, :spmv}
function load_problem_list(set::String, bench_type ::Symbol)
	if bench_type == :fom
		fn = joinpath(dirname(@__DIR__), "problem_search_results_fom", "search_results_$set.txt")
	elseif bench_type == :spmv
		fn = joinpath(dirname(@__DIR__), "problem_search_results_spmv", "search_results_$set.txt")
	else
		@error "Unrecognised bench_type: $bench_type"
	end
	
	lines = readlines(fn)
    # drop empty or “#…” comment lines, strip whitespace
	return [
		strip(line) for line in lines
		if !isempty(strip(line)) && !startswith(strip(line), "#")
	]
end