using FOMPrototypes, DataFrames, JLD2, Glob, Statistics, Plots

# Modular utilities to load, aggregate, and profile solver results.

# Find result files by variant under the default results/ folder.
function find_result_files(; variant::Union{Symbol,String}=:ADMM,
    root::AbstractString=joinpath(dirname(@__DIR__), "results"))
    v = String(variant)
    return glob("*/*/*$v*/rep*.jld2", "$root/")
end

# Parse a single JLD2 result file into a NamedTuple row.
function load_row(f::AbstractString)
    d = load(f)
    parts = split(f, '/')
    # .../results/<problem_set>/<problem_name>/<method_id>/repN.jld2
    ps, pn, mid, repfile = parts[end-3:end]
    rep = parse(Int, replace(basename(repfile), r"rep(\d+)\.jld2" => s"\1"))
    setup_time  = d["to"].inner_timers["setup"].accumulated_data.time  / 1e9
    solver_time = d["to"].inner_timers["solver"].accumulated_data.time / 1e9
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

# Load all rows from a list of files (or discovered by variant).
function load_results(; variant::Union{Symbol,String}=:ADMM,
    files::Union{Nothing,Vector{<:AbstractString}}=nothing)
    fs = isnothing(files) ? find_result_files(variant=variant) : files
    return DataFrame(map(load_row, fs))
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

# Convert string traits to Int/Float when possible
convert_trait(v::String) = (tryparse(Int, v) !== nothing) ? parse(Int, v) :
                           (tryparse(Float64, v) !== nothing) ? parse(Float64, v) : v

# Expand derived trait columns from method_id
function add_trait_columns!(df::DataFrame)
    combo = [parse_combo_id(string(id)) for id in df.method_id]
    all_traits = unique(vcat([collect(keys(d)) for d in combo]...))
    for t in all_traits
        df[!, Symbol(t)] = [haskey(d, t) ? convert_trait(d[t]) : missing for d in combo]
    end
    return df
end

# Aggregate replicates to one row per (problem_set, problem_name, method_id).
function aggregate_replicates(df::DataFrame)
    combine(groupby(df, [:problem_set, :problem_name, :method_id]),
        :k_final          => minimum => :min_k_final,
        :k_operator_final => minimum => :min_k_operator_final,
        :setup_time       => median  => :μ_setup_time,
        :setup_time       => std     => :σ_setup_time,
        :setup_time       => minimum => :min_setup_time,
        :solver_time      => median  => :median_solver_time,
        :solver_time      => std     => :σ_solver_time,
        :solver_time      => minimum => :min_solver_time,
        :total_time       => median  => :median_total_time,
        :total_time       => std     => :σ_total_time,
        :total_time       => minimum => :min_total_time,
        :status           => (sts -> count(sts .== :loop_timeout .|| sts .== :global_timeout)) => :n_timeouts,
    )
end

# Compute method labels and a title string from method IDs.
function build_labels(ids::AbstractVector{<:AbstractString})
    labels = String[]
    variants = String[]
    for id in ids
        d = parse_combo_id(id)
        lbl = ""
        if haskey(d, "accel-memory");           lbl *= "mem=$(d["accel-memory"])"; end
        if haskey(d, "krylov-tries-per-mem");    lbl *= " $(d["krylov-tries-per-mem"])"; end
        if haskey(d, "acceleration");            lbl *= " $(d["acceleration"])"; end
        if haskey(d, "anderson-interval");       lbl *= " $(d["anderson-interval"])"; end
        if haskey(d, "anderson-mem-type");       lbl *= " $(d["anderson-mem-type"])"; end
        if haskey(d, "variant");                 push!(variants, d["variant"]); end
        push!(labels, lbl)
    end
    title_str = !isempty(variants) ? ("Variants: " * join(unique(variants), ", ")) : ""
    # plot wants a matrix of labels
    return reshape(labels, 1, length(labels)), title_str
end

# Generic performance-profile computation on a chosen metric column.
# metric must be a Symbol of a column present in agg_df, e.g. :min_total_time or :min_k_operator_final.
function performance_profile(
    agg_df::DataFrame;
    metric::Symbol=:min_total_time,
    taus=0.9:0.1:100,
    success_col::Symbol=:n_timeouts,
    success_ok = (x)->(x == 0)
    )

    # Best per problem on the chosen metric
    best = combine(groupby(agg_df, [:problem_set, :problem_name]), metric => minimum => :best)
    agg2 = leftjoin(agg_df, best, on=[:problem_set, :problem_name])
    agg2.ratio = agg2[!, metric] ./ agg2.best

    methods = unique(String.(agg2.method_id))
    perf = DataFrame(τ = collect(taus))
    for m in methods
        sub = @view agg2[agg2.method_id .== m, :]
        perf[!, m] = [
            mean((sub.ratio .<= τ) .& success_ok.(sub[!, success_col]))
            for τ in taus
        ]
    end
    return perf
end

# Plot a performance profile DataFrame returned by performance_profile.
function plot_performance_profile(perf::DataFrame; labels::Union{Nothing,AbstractMatrix{<:AbstractString}}=nothing,
    title::AbstractString="", xlabel::AbstractString="τ = mᵢ / minⱼ mⱼ", ylabel::AbstractString="Fraction of problems")
    plotlyjs(); default(size=(800,600))
    ys = Matrix(perf[:, Not(:τ)])
    plot(perf.τ, ys,
        xlabel=xlabel, ylabel=ylabel,
        label = (labels === nothing ? reshape([names(perf)[2:end]...], 1, :) : labels),
        legend = :bottomright,
        title=title,
        linewidth=2,
    )
end

# Example usage (kept minimal). Run this file directly to test defaults.
function example_workflow()
    df = load_results(variant=:ADMM)
    add_trait_columns!(df)
    agg = aggregate_replicates(df)
    # Time-based profile (minimum total time across reps)
    perf_time = performance_profile(agg; metric=:min_total_time)
    labs, title = build_labels(String.(names(perf_time)[2:end]))
    plt_time = plot_performance_profile(perf_time; labels=labs, title=title)

    # k_operator-based profile (minimum final_k_operator across reps)
    perf_kop = performance_profile(agg; metric=:min_k_operator_final)
    plt_kop = plot_performance_profile(perf_kop; labels=labs, title=title,
        xlabel="τ = kᵢ / minⱼ kⱼ")

    display(plt_time)
    display(plt_kop)
end
