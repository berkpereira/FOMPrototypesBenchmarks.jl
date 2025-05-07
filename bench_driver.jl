#!/usr/bin/env julia
using FOMPrototypesBenchmarks

# --- Define your experiment grid ---
variants = [:ADMM, :PDHG]  # extend as needed
problems = [
    ("sslsq", "NYPA_Maragal_5_lasso"),
    ("sslsq", "HB_ash958_huber")
]  # extend as needed
nreps = 3

# Build flattened list of tasks
tasks = [(v, ps, pn, r) for v in variants for (ps,pn) in problems for r in 1:nreps]

# Read SLURM array index (1-based)
idx = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
variant, problem_set, problem_name, rep = tasks[idx]

# Run one bench
FOMPrototypesBenchmarks.run_single(variant, problem_set, problem_name, rep)
