using FOMPrototypes, DataFrames, JLD2, Glob, Statistics, Plots

# 1) Locate all result files
target_variant = :ADMM
files = glob("results/*/*/*$target_variant*/rep*.jld2", dirname(@__DIR__))

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
    k_final = d["results"].k_final
    k_operator_final = d["results"].k_operator_final

    return (
        file             = f,
        problem_set      = ps,
        problem_name     = pn,
        method_id        = mid,
        rep              = rep,
        setup_time       = setup_time,
        solver_time      = solver_time,
        total_time       = setup_time + solver_time,
        status           = status,
        k_final          = k_final,
        k_operator_final = k_operator_final,
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

# parse each method_id into a Dict of traits
combo = [parse_combo_id(string(id)) for id in df.method_id]

# collect the superset of all trait‐names
all_traits = unique(vcat([collect(keys(d)) for d in combo]...))

# helper to convert strings to Int/Float if possible
function convert_trait(v::String)
    if tryparse(Int, v) !== nothing
        return parse(Int, v)
    elseif tryparse(Float64, v) !== nothing
        return parse(Float64, v)
    else
        return v
    end
end

# unspool into one column per trait, filling missing with `missing`
for t in all_traits
    df[!, Symbol(t)] = [haskey(d, t) ? convert_trait(d[t]) : missing for d in combo]
end

# drop the original long `method_id` column if desired
# select!(df, Not(:method_id))

# Pivot to one row per (problem, method), aggregate reps
agg = combine(groupby(df, [:problem_set, :problem_name, :method_id]), 
    :k_final => minimum => :min_k_final,
    :k_operator_final => minimum => :min_k_operator_final,
    :setup_time => median => :μ_setup_time,
    :setup_time => std  => :σ_setup_time,
    :setup_time => minimum  => :min_setup_time,
    :solver_time => median => :median_solver_time,
    :solver_time => std  => :σ_solver_time,
    :solver_time => minimum  => :min_solver_time,
    :total_time => median => :median_total_time,
    :total_time => std  => :σ_total_time,
    :total_time => minimum  => :min_total_time,
    :status      => (sts -> count(sts .== :loop_timeout .|| sts .== :global_timeout)) => :n_timeouts,
    )

# 4) Performance profile
# compute best minimum time per problem, then ratio
best = combine(groupby(agg, [:problem_set, :problem_name]),
    :min_total_time => minimum => :best_time)
agg2 = leftjoin(agg, best, on=[:problem_set, :problem_name])
agg2.ratio = agg2.min_total_time ./ agg2.best_time

# For a grid of τ values, compute fraction solved ≤ τ
taus = 0.9:0.1:100
methods = unique(agg2.method_id)

# build a short legend label for each ID:
labels = String[]
variants = String[]
for id in methods
    d = parse_combo_id(id)
    lbl = ""

    if haskey(d, "accel-memory")
        lbl *= "mem=$(d["accel-memory"])"
    end
    if haskey(d, "krylov-tries-per-mem")
        lbl *= " $(d["krylov-tries-per-mem"])"
    end
    lbl *= " $(d["acceleration"])"
    if haskey(d, "anderson-interval")
        lbl *= " $(d["anderson-interval"])"
    end
    if haskey(d, "anderson-mem-type")
        lbl *= " $(d["anderson-mem-type"])"
    end

    push!(variants, d["variant"])  # collect for title
    
    # if accel-memory key exists, show it too
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

gr()
default(size=(800,600))
ys = Matrix(perf[:, Not(:τ)])
plt = plot(perf.τ, ys,
    xlabel="τ = tᵢ / min_j tⱼ", ylabel="Fraction of problems",
    label = labels,
    legend = :bottomright,
    title=title_str,
    linewidth=2,)