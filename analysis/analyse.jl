using FOMPrototypes, DataFrames, JLD2, Glob, Statistics, Plots

# 1) Locate all result files
files = glob("results/*/*/*/rep*.jld2", dirname(@__DIR__))

# 2) Parse each file into a row
function load_row(f)
    d = load(f)
    # extract metadata from path: problem-set, problem-name, method_id
    parts = split(f, '/')
    # assume .../results/ps/pn/id/repN.jld2
    ps, pn, mid, repfile = parts[end-3:end]
    rep = parse(Int, replace(basename(repfile), r"rep(\d+)\.jld2" => s"\1"))
    setup_time  = d["to"].inner_timers["setup"].accumulated_data.time  / 1e9 # (seconds)
    solver_time = d["to"].inner_timers["solver"].accumulated_data.time / 1e9 # (seconds)
    status = d["results"].exit_status
    return (
        file        = f,
        problem_set = ps,
        problem_name= pn,
        method_id   = mid,
        rep         = rep,
        setup_time  = setup_time,
        solver_time = solver_time,
        total_time  = setup_time + solver_time,
        status      = status,
    )
end

# helper: parse a method‐ID of the form "k1=v1_k2=v2_…"
function parse_combo_id(id::AbstractString)
    d = Dict{String,String}()
    for part in split(id, "_")
        if occursin('=', part)
            k,v = split(part, '='; limit=2)
            d[k] = v
        end
    end
    return d
end

rows = map(load_row, files)
df = DataFrame(rows)

# 3) Pivot to one row per (problem, method), aggregate reps
agg = combine(groupby(df, [:problem_set, :problem_name, :method_id]), 
    :setup_time => mean => :μ_setup_time,
    :setup_time => std  => :σ_setup_time,
    :solver_time => mean => :μ_solver_time,
    :solver_time => std  => :σ_solver_time,
    :total_time => mean => :μ_total_time,
    :total_time => std  => :σ_total_time,
    :status      => (sts -> count(sts .== :loop_timeout .|| sts .== :global_timeout)) => :n_timeouts,
    )

println("Summary:")
show(first(agg, 10))

# 4) Performance profile
# compute best time per problem, then ratio
best = combine(groupby(agg, [:problem_set, :problem_name]),
    :μ_total_time => minimum => :best_time)

agg2 = leftjoin(agg, best, on=[:problem_set, :problem_name])
agg2.ratio = agg2.μ_total_time ./ agg2.best_time

# For a grid of τ values, compute fraction solved ≤ τ
taus = 0.9:0.1:50
methods = unique(agg2.method_id)

# build a short legend label for each ID:
labels = String[]
variants = String[]
for id in methods
    d = parse_combo_id(id)
    push!(variants, d["variant"])  # collect for title
    # always show acceleration
    lbl = "$(d["acceleration"])"
    # if accel-memory key exists, show it too
    println(keys(d))
    if haskey(d, "accel-memory")
        lbl *= " (mem=$(d["accel-memory"]))"
    end
    if haskey(d, "anderson-mem-type")
        lbl *= " ($(d["anderson-mem-type"]))"
    end
    push!(labels, lbl)
end
# plot expects matrix of labels
labels = reshape(labels, 1, length(labels))

# dedupe variants for title
variants = unique(variants)
title_str = "Variants: " * join(variants, ", ")

perf    = DataFrame(τ = taus)

# generate performance profile
for m in methods
    sub = @view agg2[agg2.method_id .== m, :]
    
    # problem is only counted as solved within τ threshold
    # if none of the replicates timed out
    # mean of the boolean vector gives fraction of problems
    # solved (with this definition)
    perf[!, m] = [
        mean((sub.ratio .<= τ) .& (sub.n_timeouts .== 0))
        for τ in taus]
end

plotlyjs()
default(size=(800,600))
ys = Matrix(perf[:, Not(:τ)])
plt = plot(perf.τ, ys,
    xlabel="τ = tᵢ / min_j tⱼ", ylabel="Fraction of problems",
    label = labels,
    legend = :bottomright,
    title=title_str,
    linewidth=2,)