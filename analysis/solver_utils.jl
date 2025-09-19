using FOMPrototypes, DataFrames, JLD2, Glob, Statistics, Plots
using Base: AbstractSet
using FOMPrototypesBenchmarks

# Modular utilities to load, aggregate, and profile solver results.

# Align plotting style with analysis/spmv_utils.jl
include(joinpath(@__DIR__, "spmv_utils.jl"))

const RUN_PARAM_KEYS = ["global-timeout", "max-k-operator", "rel-kkt-tol"]

# Normalize arbitrary selector data (method traits or run params) to
# Dict{String,Vector{String}} with hyphenated keys and string values.
function _normalize_selector(sel)
    sel === nothing && return nothing
    dict = Dict{String,Vector{String}}()
    pairs_iter = sel isa NamedTuple ? pairs(sel) : sel
    if !(sel isa Dict || sel isa AbstractVector{<:Pair} || sel isa NamedTuple)
        error("Unsupported selector type: $(typeof(sel))")
    end
    for (k, v_raw) in pairs_iter
        ks = k isa Symbol ? replace(String(k), '_' => '-') : String(k)
        is_vector_value = (v_raw isa AbstractVector) && !(v_raw isa AbstractString)
        values = is_vector_value ? v_raw : (v_raw,)
        dict[ks] = [string(v) for v in values]
    end
    return dict
end

function _normalize_problem_sets(ps)
    ps === nothing && return nothing
    if ps isa AbstractString || ps isa Symbol
        return [String(ps)]
    elseif ps isa Tuple || ps isa AbstractVector || ps isa AbstractSet
        return [String(v) for v in ps]
    else
        error("Unsupported problem_sets type: $(typeof(ps))")
    end
end

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

function _selector_matches(sel::Dict{String,Vector{String}}, candidate::Dict{String,String})
    for (k, vs) in sel
        if !haskey(candidate, k)
            return false
        end
        if !(candidate[k] in vs)
            return false
        end
    end
    return true
end

# Find result files by variant and optional run-parameter selector under the default results/ folder.
function find_result_files(; variant::Union{Symbol,String}=:ADMM,
    run_params=nothing,
    root::AbstractString=joinpath(dirname(@__DIR__), "results"),
    problem_sets=nothing)
    v = String(variant)
    rp_norm = _normalize_run_params(run_params)
    ps_norm = _normalize_problem_sets(problem_sets)
    if rp_norm === nothing
        if ps_norm === nothing
            # layout: .../<ps>/<pn>/<method_id>/<run_params_id>/rep*.jld2
            return glob("*/*/*$v*/*/rep*.jld2", "$root/")
        else
            files = String[]
            for ps in ps_norm
                append!(files, glob("$ps/*/*$v*/*/rep*.jld2", "$root/"))
            end
            return files
        end
    else
        rid = _run_params_id_for(rp_norm)
        if ps_norm === nothing
            return glob("*/*/*$v*/$rid/rep*.jld2", "$root/")
        else
            files = String[]
            for ps in ps_norm
                append!(files, glob("$ps/*/*$v*/$rid/rep*.jld2", "$root/"))
            end
            return files
        end
    end
end

# Enumerate available (problem_set, problem_name, method_id, run_params_id)
# combinations for a variant before loading JLD2 files. Optional selectors can
# narrow down the runs at the directory level, making it easier to inspect what
# data is available prior to loading it.
function index_variant_runs(; variant::Union{Symbol,String}=:ADMM,
    root::AbstractString=joinpath(dirname(@__DIR__), "results"),
    method_traits=nothing,
    run_params=nothing,
    include_files::Bool=false,
    problem_sets=nothing)

    variant_str = String(variant)
    trait_sel = _normalize_selector(method_traits)
    rp_sel    = _normalize_selector(run_params)
    ps_sel    = _normalize_problem_sets(problem_sets)
    rows = NamedTuple[]
    for method_dir in filter(isdir, glob("*/*/*$variant_str*", "$root/"))
        parts = splitpath(method_dir)
        length(parts) < 3 && continue
        ps, pn, mid = parts[end-2:end]
        if ps_sel !== nothing && !(ps in ps_sel)
            continue
        end
        method_traits_dict = parse_combo_id(mid)
        if get(method_traits_dict, "variant", nothing) != variant_str
            continue
        end
        if trait_sel !== nothing && !_selector_matches(trait_sel, method_traits_dict)
            continue
        end
        run_dirs = filter(isdir, readdir(method_dir; join=true))
        for run_dir in run_dirs
            rid = basename(run_dir)
            run_dict = parse_combo_id(rid)
            if rp_sel !== nothing && !_selector_matches(rp_sel, run_dict)
                continue
            end
            rep_files = sort(glob("rep*.jld2", joinpath(run_dir, "")))
            isempty(rep_files) && continue
            files_col = include_files ? rep_files : missing
            push!(rows, (
                problem_set   = ps,
                problem_name  = pn,
                method_id     = mid,
                run_params_id = rid,
                method_path   = method_dir,
                run_path      = run_dir,
                n_reps        = length(rep_files),
                files         = files_col,
            ))
        end
    end
    df = DataFrame(rows)
    isempty(df) && return df
    add_trait_columns!(df)
    add_run_param_columns!(df)
    if !include_files
        select!(df, Not(:files))
    end
    return df
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
    root::AbstractString=joinpath(dirname(@__DIR__), "results"),
    problem_sets=nothing)
    fs = isnothing(files) ? find_result_files(variant=variant, run_params=run_params, root=root, problem_sets=problem_sets) : files
    ps_norm = _normalize_problem_sets(problem_sets)
    if ps_norm !== nothing
        fs = [f for f in fs if splitpath(f)[end-4] in ps_norm]
    end
    rows = map(load_row, fs)
    return isempty(rows) ? DataFrame() : DataFrame(rows)
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
    isempty(df) && return DataFrame()
    groups = groupby(df, [:problem_set, :problem_name, :method_id, :run_params_id])
    rows = NamedTuple[]
    for g in groups
        solved_mask = g.status .== :kkt_solved
        solved = view(g, solved_mask, :)
        # if not solved at all, set these markers appropriately
        # eg "time to solve = infty" and so on
        if isempty(solved)
            min_k_final = Inf
            min_k_operator_final = Inf
            median_solver_time = Inf
            σ_solver_time = NaN
            min_solver_time = Inf
            median_total_time = Inf
            σ_total_time = NaN
            min_total_time = Inf
        else
            min_k_final = minimum(solved.k_final)
            min_k_operator_final = minimum(solved.k_operator_final)
            median_solver_time = median(solved.solver_time)
            σ_solver_time = std(solved.solver_time)
            min_solver_time = minimum(solved.solver_time)
            median_total_time = median(solved.total_time)
            σ_total_time = std(solved.total_time)
            min_total_time = minimum(solved.total_time)
        end
        # note that we let the SETUP time remain regardless.
        # rationale is that we won't make a performance profile
        # based on the setup time, and it can still be useful
        # information to read in aggregate
        μ_setup_time = median(g.setup_time)
        σ_setup_time = std(g.setup_time)
        min_setup_time = minimum(g.setup_time)

        n_timeouts = count(st -> st == :loop_timeout || st == :global_timeout, g.status)
        push!(rows, (
            problem_set = g.problem_set[1],
            problem_name = g.problem_name[1],
            method_id = g.method_id[1],
            run_params_id = g.run_params_id[1],
            min_k_final = min_k_final,
            min_k_operator_final = min_k_operator_final,
            μ_setup_time = μ_setup_time,
            σ_setup_time = σ_setup_time,
            min_setup_time = min_setup_time,
            median_solver_time = median_solver_time,
            σ_solver_time = σ_solver_time,
            min_solver_time = min_solver_time,
            median_total_time = median_total_time,
            σ_total_time = σ_total_time,
            min_total_time = min_total_time,
            n_timeouts = n_timeouts,
        ))
    end
    return DataFrame(rows)
end

# Compute method labels and a title string from method IDs.
function build_labels(
    ids::AbstractVector{<:AbstractString},
    unwanted_label_keys::AbstractVector{<:AbstractString}=nothing,
    problem_sets::Union{AbstractVector{<:AbstractString}, Nothing}=nothing,
)
    if isnothing(unwanted_label_keys); unwanted_label_keys = [""]; end
    
    labels = String[]
    variants = String[]
    for id in ids
        d = parse_combo_id(id)
        lbl = ""
        if haskey(d, "acceleration") && !("acceleration" in unwanted_label_keys); lbl *= "$(uppercasefirst(d["acceleration"]))"; end
        if haskey(d, "accel-memory") && !("accel-memory" in unwanted_label_keys); lbl *= " mem=$(d["accel-memory"])"; end
        if haskey(d, "krylov-tries-per-mem") && !("krylov-tries-per-mem" in unwanted_label_keys); lbl *= ", tries = $(d["krylov-tries-per-mem"])"; end
        if haskey(d, "anderson-interval") && !("anderson-interval" in unwanted_label_keys); lbl *= ", interval = $(d["anderson-interval"])"; end
        if haskey(d, "anderson-mem-type") && !("anderson-mem-type" in unwanted_label_keys); lbl *= ", $(d["anderson-mem-type"]) memory"; end
        push!(labels, lbl)
        
        if haskey(d, "variant"); push!(variants, d["variant"]); end
    end

    # assign figure title string
    title_str = !isnothing(problem_sets) ? ("Problem sets: " * join(unique(problem_sets), ", ")) : ""
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
    ylabel::AbstractString="Fraction solved",
    legend=:bottomright,
    linealpha::Real=1.0,
    xlims=(1.0, Inf),
    ylims=(0, 1.05),
    outfile::Union{Nothing,AbstractString}=nothing,
    plotkwargs...)

    ys = Matrix(perf[:, Not(:τ)])
    labmat = (labels === nothing ? reshape([names(perf)[2:end]...], 1, :) : labels)
    kwargs = Dict{Symbol,Any}(plotkwargs)
    if !haskey(kwargs, :linestyle)
        base_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
        n_series = size(ys, 2)
        repeats = max(1, ceil(Int, n_series / length(base_styles)))
        kwargs[:linestyle] = reshape(repeat(base_styles, repeats)[1:n_series], 1, n_series)
    end

    plt = plot(perf.τ, ys;
        xlabel=xlabel,
        ylabel=ylabel,
        label=labmat,
        legend=legend,
        title=title,
        alpha=linealpha,
        xlims=xlims,
        ylims=ylims,
        kwargs...,
    )

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
    problem_sets=nothing,
)
    df = load_results(variant=variant, run_params=run_params, root=root, problem_sets=problem_sets)
    if isempty(df)
        error("No results found for variant=$(variant), run_params=$(run_params), problem_sets=$(problem_sets)")
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
    xlabel::AbstractString = (metric == :min_total_time ? "Performance ratio "*L"\tau" : "τ = mᵢ / minⱼ mⱼ"),
    ylabel::AbstractString = "Fraction solved",
    outfile::Union{Nothing,AbstractString}=nothing,
    problem_sets=nothing,
    plotkwargs...)
    prof = make_profile(variant; run_params=run_params, metric=metric, taus=taus, root=root, success_col=success_col, success_ok=success_ok, problem_sets=problem_sets)
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
