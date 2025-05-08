#!/usr/bin/env julia

# ——————————————————————————————————————————————————————————————————
# 0) Define grid AND compute what reps are still missing
# ——————————————————————————————————————————————————————————————————

const combos = [:ADMM, :PDHG]    # your variant / hyper combos
const problems = [
    ("sslsq","NYPA_Maragal_5_lasso"),
    ("sslsq", "HB_ash219_lasso"),
    ("sslsq","HB_ash958_huber"),
    ("sslsq", "HB_abb313_lasso"),
    # …etc…
]
const nreps = 3                  # total replicates per problem
const resdir = "results"         # same as run_multiple’s default

# build a list of tasks (variant, problem_set, problem_name, reps_missing)
work = []
for variant in combos
  for (ps, pn) in problems
    outdir = joinpath(resdir, ps, pn, string(variant))
    missing = Int[]
    for rep in 1:nreps
      if !isfile(joinpath(outdir, "rep$(rep).jld2"))
        push!(missing, rep)
      end
    end
    if !isempty(missing)
      push!(work, (variant, ps, pn, missing))
    end
  end
end

if isempty(work)
  @info "✅ All results already exist!  Nothing to do."
  exit(0)
end

# ——————————————————————————————————————————————————————————————————
# 1) Now that we know there *is* work, load up and dispatch it
# ——————————————————————————————————————————————————————————————————

using Distributed, Logging

# how many workers we get from SLURM
nworkers = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
@info "Spawning $nworkers Julia workers..."
addprocs(nworkers; exeflags="--project=.")

@everywhere using FOMPrototypesBenchmarks

# sanity‐check PIDs (optional)
workers_list = workers()
pids = [remotecall_fetch(getpid, w) for w in workers_list]
@info "Worker PIDs: $pids"

# partition workers *by variant* (outer grouping)
ncombos = length(combos)
assign = mod1.(1:length(workers_list), ncombos)
groups = [ workers_list[assign .== i] for i in 1:ncombos ]

# ——————————————————————————————————————————————————————————————————
# 2) For each variant group: warm up, then pmap each (ps,pn,reps) tuple
# ——————————————————————————————————————————————————————————————————

@sync for (combo_idx, variant) in enumerate(combos)
  wgroup = groups[combo_idx]
  @async begin
    @info "Group $combo_idx - variant=$variant on workers=$(wgroup)"

    # warm‐up each worker in parallel
    @sync for w in wgroup
      @async remotecall_wait(FOMPrototypesBenchmarks.run_warmup, w, variant)
    end

    # collect only those tasks in `work` that match this variant
    mytasks = filter(x -> x[1] == variant, work)
    # mytasks is a Vector of (variant, ps, pn, missing_reps::Vector{Int})

    # build a pool and dispatch
    pool = Distributed.CachingPool(wgroup)
    pmap(t -> begin
        (v, ps, pn, reps) = t
        FOMPrototypesBenchmarks.run_multiple(v, 1000, ps, pn; reps = reps)
      end,
      pool,
      mytasks)
  end
end

@info "🎉 All done!"
