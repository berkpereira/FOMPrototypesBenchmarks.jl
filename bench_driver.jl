#!/usr/bin/env julia

using Distributed, Logging, Random
using FOMPrototypesBenchmarks: method_id, DEFAULT_SOLVER_ARGS
import Dates

start_time = Dates.now()

const problems = [
	("sslsq","NYPA_Maragal_5_lasso"),
	("sslsq","HB_ash958_huber"),
]
const nreps = 3

# ————————————————————————————————————————————————
# A) List *only* the solver‐defining override keys here
# ————————————————————————————————————————————————
const overrides = [
	Dict(
        "variant"               => Symbol(1),
        "acceleration"          => :none,
		"max-iter"              => Inf,
		"rel-kkt-tol"           => 1e-12,
		"global-timeout"        => 15.0,
		"loop-timeout"          => 3.0,
        ),
	# Dict(
    #     "variant"               => :ADMM,
    #     "acceleration"          => :anderson,
    #     "accel-memory"          => 100,
    #     ),
	# Dict(
    #     "variant"               => :PDHG,
    #     "acceleration"          => :none,
    #     ),
]

# build full solver configurations by merging with default key-value pairs
const solver_configs = [merge(copy(DEFAULT_SOLVER_ARGS), o) for o in overrides]

# B) Precompute each combo’s string ID
const combo_ids = method_id.(solver_configs)

# C) Scan for missing work
work = Tuple{Dict,String,String,String,Vector{Int}}[]
for (cfg,id) in zip(solver_configs, combo_ids)
	for (ps,pn) in problems
		missing_reps = [r for r in 1:nreps if !isfile(joinpath("results", ps, pn, id, "rep$(r).jld2"))]
		!isempty(missing_reps) && push!(work, (cfg, id, ps, pn, missing_reps))
	end
end

if isempty(work)
    @info "✅ Nothing to do, exiting"
    elapsed_time = Dates.now() - start_time
    t = Dates.Time(0) + elapsed_time
    @info "Elapsed time: $(Dates.format(t, "HH:MM:SS.s"))"
    exit(0)
end

# D) Spawn SLURM-provided workers
# reserve a CPU for this driver process, hence -1
nworkers = parse(Int, ENV["SLURM_CPUS_PER_TASK"]) - 1
addprocs(nworkers; exeflags="--project=.")
@everywhere using FOMPrototypesBenchmarks: run_multiple, run_warmup

# E) Split into groups by #combos, round-robin
groups = begin
	assigns = mod1.(1:nworkers, length(solver_configs))
	ws      = workers()
	[ ws[assigns .== i] for i in 1:length(solver_configs) ]
end

# F) Warm-up & dispatch each combo in parallel
@sync for (i,cfg) in enumerate(solver_configs)
	id     = combo_ids[i]
	wgroup = groups[i]
	@async begin
		@info "Combo $i ($id) using workers $wgroup"

		# F1) warm up each worker to JIT-compile
		@sync for w in wgroup
			@async remotecall_wait(run_warmup, w, cfg)
		end

		# F2) only the tasks for this combo
		tasks = filter(t -> t[1] === cfg, work)

		# F2.5) randomised the order of tasks for better load balancing
		# even when number of tasks is small compared to number of workers
		shuffle!(tasks)

		# F3) dispatch via a CachingPool
		pool = CachingPool(wgroup)
		pmap(t -> begin
			(_, _, ps, pn, reps) = t
			run_multiple(cfg, ps, pn; reps=reps)
		end, pool, tasks)
	end
end

@info "🎉 All done!"
elapsed_time = Dates.now() - start_time
t = Dates.Time(0) + elapsed_time
@info "Elapsed time: $(Dates.format(t, "HH:MM:SS.s"))"