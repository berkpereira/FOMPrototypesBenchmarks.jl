# ————————————————————————————————————————————————
# 2) Canonical method ID (only hyperkeys that *define* the method)
# ————————————————————————————————————————————————

using FOMPrototypes, JLD2, Random, Dates

function method_id(config::Dict{String,Any})
	ks = String[
		"variant",
		"res-norm",
		"acceleration",
		"rho",
		"theta",
	]
	if config["acceleration"] == :anderson
		append!(ks, [
			"accel-memory",
			"anderson-period",
			"anderson-broyden-type",
			"anderson-mem-type",
			"anderson-reg",
		])
	elseif config["acceleration"] == :krylov
		append!(ks, [
			"accel-memory",
			"krylov-operator",
		])
	end
	sort!(ks)
	parts = String[]
	for k in ks
		push!(parts, string(k, "=", config[k]))
	end
	return join(parts, "_")
end

# ————————————————————————————————————————————————
# 3) run_multiple: accept a *solver config* + problem identifiers
# ————————————————————————————————————————————————
function run_multiple(args::Dict{String,Any},
					  problem_set::String,
					  problem_name::String;
					  reps::Vector{Int},
					  resdir::String="results",
					  save_results::Bool=true)

	# 3a) decide which reps are missing
	mid = method_id(args)
	outdir = joinpath(resdir, problem_set, problem_name, mid)
	if save_results
		missing = Int[]
		for r in reps
			f = joinpath(outdir, "rep$(r).jld2")
			!isfile(f) && push!(missing, r)
		end
		isempty(missing) && return
		reps = missing
	end

	# 3b) fetch data & solve reference once
	prob     = FOMPrototypes.fetch_data(problem_set, problem_name)
	
	# optional) solve reference using external solver
	# x_ref, s_ref, y_ref, obj_ref = FOMPrototypes.solve_reference(prob, problem_set, problem_name, args)

	# 3c) run each missing replicate
	for rep in reps
		Random.seed!(rep)
		ws, results, to = FOMPrototypes.run_prototype(
			prob, problem_set, problem_name, args)
		if save_results
			mkpath(outdir)
			f = joinpath(outdir, "rep$(rep).jld2")
			ts = Dates.now() # timestamp
			@save f args problem_set problem_name rep to ws results ts
		end
	end
end

# ————————————————————————————————————————————————
# 4) run_warmup: just a tiny dummy solve for compilation
# ————————————————————————————————————————————————
function run_warmup(warmup_args::Dict{String,Any})
	# pick a trivial problem that actually exists

	# see about size of this problem!
	# small alternative is NYPA_Maragal_1_lasso
	run_multiple(warmup_args, "sslsq", "NYPA_Maragal_5_lasso";
				 reps=[1], save_results=false)
end