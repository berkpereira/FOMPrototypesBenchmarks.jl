# ————————————————————————————————————————————————
# A) List *only* the solver‐defining override keys here
# ————————————————————————————————————————————————
# 0) A little helper to build each override dict
function make_override(variant; acceleration=:none, accel_memory=nothing,
					anderson_period=2, anderson_broyden_type=:QR2,
					anderson_mem_type=:restarted)
	d = Dict{String,Any}()
	d["variant"]      = variant
	d["acceleration"] = acceleration
	if accel_memory !== nothing
		d["accel-memory"] = accel_memory
	end
	if acceleration === :anderson
		# only relevant for Anderson
		d["anderson-period"]       = anderson_period
		d["anderson-broyden-type"] = anderson_broyden_type
		d["anderson-mem-type"]     = anderson_mem_type
	end
	return d
end

function load_problem_list(set::String)
	fn = joinpath(dirname(@__DIR__), "problem_lists", "$set.txt")
	lines = readlines(fn)
    # drop empty or “#…” comment lines, strip whitespace
	return [
		strip(line) for line in lines
		if !isempty(strip(line)) && !startswith(strip(line), "#")
	]
end