using Distributed, Logging, Random
using FOMPrototypesBenchmarks
import Dates

# pick problem sets and auto-generate the list
const problem_sets = [
    "sslsq",
    "maros",
    "mpc",
    "netlib_feasible",
]


# note this loads problems from search results (ie possibly filtered somehow
# from whole of each problem set)
const problems = vcat((
	[(ps, pname) for pname in FOMPrototypesBenchmarks.load_problem_list(ps)]
	for ps in problem_sets
)...)
const nreps = 2 # we can use minimum time in each problem

################################################################################

start_time = Dates.now()

# 1) “grid” of settings
variant   = :ADMM
memories  = [15]

anderson_intervals = [1, 10]

krylov_tries_numbers = [1, 3]

# 2) build each family by comprehension
acc_none   = [ FOMPrototypesBenchmarks.make_override(variant; acceleration=:none) ]

acc_krylov = [ FOMPrototypesBenchmarks.make_override(variant;
                acceleration=:krylov,
                accel_memory=m,
                krylov_tries_per_mem=krylov_tries,
                )
            for m in memories for krylov_tries in krylov_tries_numbers]

# Anderson configs with diff broyden types, mem_type == :restarted
acc_anderson_type2 = [ FOMPrototypesBenchmarks.make_override(variant;
                    acceleration=:anderson,
                    accel_memory=m,
                    anderson_interval=anderson_interval,
                    anderson_broyden_type=:QR2,
                    anderson_mem_type=:restarted)
                for m in memories for anderson_interval in anderson_intervals]

acc_anderson_type1 = [ FOMPrototypesBenchmarks.make_override(variant;
                    acceleration=:anderson,
                    accel_memory=m,
                    anderson_interval=anderson_interval,
                    anderson_broyden_type=Symbol(1),
                    anderson_mem_type=:rolling)
                    for m in memories for anderson_interval in anderson_intervals]

# 3) concatenate to get the full override list
overrides = [
    acc_none;
    acc_krylov;
    acc_anderson_type2;
    acc_anderson_type1;
    ]

# build full solver configurations by merging with default key-value pairs
solver_configs = [merge(copy(FOMPrototypesBenchmarks.DEFAULT_SOLVER_ARGS), o) for o in overrides]

# B) Precompute each combo’s string IDs
combo_solver_ids = FOMPrototypesBenchmarks.method_id.(solver_configs)
combo_run_ids    = FOMPrototypesBenchmarks.run_params_id.(solver_configs)

# log what we're about to run
@info "About to benchmark $(length(combo_solver_ids)) solver variants on $(length(problems)) problems, each with $nreps replicates."
@info "Solver IDs ($(length(combo_solver_ids))):"
for id in combo_solver_ids
    @info "  • $id"
end
@info "Run Param IDs (shared across problems):"
for id in unique(combo_run_ids)
    @info "  • $id"
end
@info "Problems ($(length(problems))):"
for (ps, pn) in problems
    @info "  • $ps / $pn"
end

# C) Scan for missing work
work = Tuple{Dict,String,String,String,String,Vector{Int}}[]
for (cfg,solver_id,run_id) in zip(solver_configs, combo_solver_ids, combo_run_ids)
    for (ps,pn) in problems
        missing_reps = [r for r in 1:nreps if !isfile(joinpath("results", ps, pn, solver_id, run_id, "rep$(r).jld2"))]
        !isempty(missing_reps) && push!(work, (cfg, solver_id, run_id, ps, pn, missing_reps))
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
@sync for (i, cfg) in enumerate(solver_configs)
    solver_id = combo_solver_ids[i]
    run_id    = combo_run_ids[i]
    wgroup = groups[i]
    @async begin
        @info "Combo $i ($solver_id | $run_id) using workers $wgroup"

        # F1) warm up each worker to JIT-compile
        @sync for w in wgroup
            cfg_warmup = copy(cfg)
            cfg_warmup["max-iter"] = 200
            cfg_warmup["global-timeout"] = Inf
            cfg_warmup["loop-timeout"] = Inf
            # if method is too good on the problem, the warmup run might be
            # too short to precompile as intended -- set 0 tolerance
            cfg_warmup["rel-kkt-tol"] = 0.0
            @async remotecall_wait(run_warmup, w, cfg_warmup)
        end

        # F2) only the tasks for this combo
        tasks = filter(t -> t[1] === cfg, work)

        # F2.5) randomised the order of tasks for better load balancing
        # even when number of tasks is small compared to number of workers
        shuffle!(tasks)

        # F3) dispatch via a CachingPool
        pool = CachingPool(wgroup)
        pmap(t -> begin
            (_, solver_id, run_id, ps, pn, reps) = t
            try
                @info "Starting $solver_id | $run_id on $ps/$pn reps=$(reps)"
                run_multiple(cfg, ps, pn; reps=reps)
            catch err
                @error "Failed on method=$solver_id run=$run_id problem=$ps/$pn reps=$(reps)" exception=err
                rethrow()
            end
        end, pool, tasks)

    end
end

@info "🎉 All done!"
elapsed_time = Dates.now() - start_time
t = Dates.Time(0) + elapsed_time
@info "Elapsed time: $(Dates.format(t, "HH:MM:SS.s"))"
