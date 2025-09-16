using FOMPrototypes, DataFrames, JLD2, Glob, Statistics, Plots
using FOMPrototypesBenchmarks

# Modular utilities to load, aggregate, and profile solver results.

# Align plotting style with analysis/spmv_utils.jl
include(joinpath(@__DIR__, "spmv_utils.jl"))

const RUN_PARAM_KEYS = ["global-timeout", "max-k-operator", "rel-kkt-tol"]

# Normalize run-parameter selectors to Dict{String,Any} with hyphenated keys
function _normalize_run_params(rp)
    rp === nothing && return nothing
    if rp isa NamedTuple
        d = Dict{String,Any}()
        for k in keys(rp)
            ks = String(k)
            ks = replace(ks, '_' => '-')
            d[ks] = getfield(rp, k)
        end
        return d
    elseif rp isa Dict
        d = Dict{String,Any}()
        for (k, v) in rp
            ks = k isa Symbol ? replace(String(k), '_' => '-') : String(k)
            d[ks] = v
        end
        return d
    elseif rp isa AbstractVector{<:Pair}
        d = Dict{String,Any}()
        for (k, v) in rp
            ks = k isa Symbol ? replace(String(k), '_' => '-') : String(k)
            d[ks] = v
        end
        return d
    else
        error("Unsupported run_params type: $(typeof(rp))")
    end
end

function _run_params_id_for(rp_norm::Dict{String,Any})
    cfg = merge(copy(FOMPrototypesBenchmarks.DEFAULT_SOLVER_ARGS), rp_norm)
    return FOMPrototypesBenchmarks.run_params_id(cfg)
end

# Find result files by variant and optional run-parameter selector under the default results/ folder.
function find_result_files(; variant::Union{Symbol,String}=:ADMM,
    run_params=nothing,
    root::AbstractString=joinpath(dirname(@__DIR__), "results"))
    v = String(variant)
    rp_norm = _normalize_run_params(run_params)
    if rp_norm === nothing
        # layout: .../<ps>/<pn>/<method_id>/<run_params_id>/rep*.jld2
        return glob("*/*/*$v*/*/rep*.jld2", "$root/")
    else
        rid = _run_params_id_for(rp_norm)
        return glob("*/*/*$v*/$rid/rep*.jld2", "$root/")
    end
end

# Parse a single JLD2 result file into a NamedTuple row.
function load_row(f::AbstractString)
    d = load(f)
    parts = split(f, '/')
    # .../results/<problem_set>/<problem_name>/<method_id>/<run_params_id>/repN.jld2
    ps, pn, mid, rpid, repfile = parts[end-4:end]
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
        run_params_id    = rpid,
        rep              = rep,
        setup_time       = setup_time,
        solver_time      = solver_time,
        total_time       = setup_time + solver_time,
        status           = status,
        k_final          = k_final,
        k_operator_final = k_operator_final,
    )
end

# Load all rows from a list of files (or discovered by variant and run_params).
function load_results(; variant::Union{Symbol,String}=:ADMM,
    run_params=nothing,
    files::Union{Nothing,Vector{<:AbstractString}}=nothing,
    root::AbstractString=joinpath(dirname(@__DIR__), "results"))
    fs = isnothing(files) ? find_result_files(variant=variant, run_params=run_params, root=root) : files
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

# Expand run-parameter columns from run_params_id
function add_run_param_columns!(df::DataFrame)
    rcombo = [parse_combo_id(string(id)) for id in df.run_params_id]
    for t in RUN_PARAM_KEYS
        df[!, Symbol(replace(t, '-' => '_'))] = [haskey(d, t) ? convert_trait(d[t]) : missing for d in rcombo]
    end
    return df
end

# Aggregate replicates to one row per (problem_set, problem_name, method_id, run_params_id).
function aggregate_replicates(df::DataFrame)
    combine(groupby(df, [:problem_set, :problem_name, :method_id, :run_params_id]),
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

    # Best per (problem, run_params_id) on the chosen metric to avoid mixing runs
    best = combine(groupby(agg_df, [:problem_set, :problem_name, :run_params_id]), metric => minimum => :best)
    agg2 = leftjoin(agg_df, best, on=[:problem_set, :problem_name, :run_params_id])
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
function plot_performance_profile(perf::DataFrame;
    labels::Union{Nothing,AbstractMatrix{<:AbstractString}}=nothing,
    title::AbstractString="",
    xlabel::AbstractString="τ = mᵢ / minⱼ mⱼ",
    ylabel::AbstractString="Fraction of problems",
    legend=:bottomright,
    lw::Real=1.5,
    linealpha::Real=1.0,
    xlims=nothing,
    ylims=(0, 1.0),
    outfile::Union{Nothing,AbstractString}=nothing,
    plotkwargs...)

    ys = Matrix(perf[:, Not(:τ)])
    labmat = (labels === nothing ? reshape([names(perf)[2:end]...], 1, :) : labels)
    plt = plot(perf.τ, ys;
        xlabel=xlabel,
        ylabel=ylabel,
        label=labmat,
        legend=legend,
        title=title,
        linewidth=lw,
        alpha=linealpha,
        xlims=xlims,
        ylims=ylims,
        plotkwargs...)

    if outfile !== nothing
        save_pdf(plt, String(outfile))
    end
    return plt
end

# ————————————————————————————————————————————————
# Convenience wrappers for quick profiles
# ————————————————————————————————————————————————

function make_profile(
    variant::Union{Symbol,String};
    run_params=nothing,
    metric::Symbol=:min_total_time,
    taus=0.9:0.1:100,
    root::AbstractString=joinpath(dirname(@__DIR__), "results"),
    success_col::Symbol=:n_timeouts,
    success_ok = (x)->(x == 0),
)
    df = load_results(variant=variant, run_params=run_params, root=root)
    if isempty(df)
        error("No results found for variant=$(variant) and run_params=$(run_params)")
    end
    add_trait_columns!(df)
    add_run_param_columns!(df)
    agg = aggregate_replicates(df)
    perf = performance_profile(agg; metric=metric, taus=taus, success_col=success_col, success_ok=success_ok)
    labels, title = build_labels(String.(names(perf)[2:end]))
    return (perf=perf, labels=labels, title=title, agg=agg, df=df)
end

function plot_profile_for(
    variant::Union{Symbol,String};
    run_params=nothing,
    metric::Symbol=:min_total_time,
    taus=0.9:0.1:100,
    root::AbstractString=joinpath(dirname(@__DIR__), "results"),
    success_col::Symbol=:n_timeouts,
    success_ok = (x)->(x == 0),
    xlabel::AbstractString = (metric == :min_total_time ? "τ = timeᵢ / minⱼ timeⱼ" : "τ = mᵢ / minⱼ mⱼ"),
    ylabel::AbstractString = "Fraction of problems",
    outfile::Union{Nothing,AbstractString}=nothing,
    plotkwargs...)
    prof = make_profile(variant; run_params=run_params, metric=metric, taus=taus, root=root, success_col=success_col, success_ok=success_ok)
    return plot_performance_profile(prof.perf;
        labels=prof.labels,
        title=prof.title,
        xlabel=xlabel,
        ylabel=ylabel,
        outfile=outfile,
        plotkwargs...)
end

# Example usage (kept minimal). Run this file directly to test defaults.
function example_workflow()
    # One-liner plot for a given (variant, run_params)
    st = paper_plot_kwargs(; column=:single, fontsize=9, tight=true)
    plt_time = plot_profile_for(
        :ADMM;
        run_params=(global_timeout=Inf, max_k_operator=3000, rel_kkt_tol=1e-9),
        metric=:min_total_time,
        legend=:bottomright,
        lw=1.5,
        ylims=(0,1),
        st...,
    )
    display(plt_time)

    # Or prepare data once and plot multiple metrics
    prof = make_profile(:ADMM; run_params=(global_timeout=Inf, max_k_operator=3000, rel_kkt_tol=1e-9))
    perf_kop = performance_profile(prof.agg; metric=:min_k_operator_final)
    plt_kop = plot_performance_profile(perf_kop;
        labels=prof.labels,
        title=prof.title,
        xlabel="τ = kᵢ / minⱼ kⱼ",
        legend=:bottomright,
        lw=1.5,
        ylims=(0,1),
        st...,
    )
    display(plt_kop)
end
