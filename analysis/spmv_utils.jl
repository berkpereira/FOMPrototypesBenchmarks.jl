# Utilities to load and analyze SpMV benchmark results

using JLD2
using Glob
using DataFrames
using Statistics
using Plots

"""
    load_spmv_results(; root="results_spmv") -> DataFrame

Load all SpMV result files from `root` and return a DataFrame with one row per
recorded measurement. Each row includes problem identifiers, matrix/op metadata,
timings, and derived quantities like density.
"""
function load_spmv_results(; root::AbstractString = "results_spmv")
    # Resolve path relative to repo root when called from this file
    files = glob(joinpath(root, "*", "*", "spmv_*.jld2"))

    rows = NamedTuple[]
    for f in files
        d = load(f)
        meta    = haskey(d, "meta")    ? d["meta"]    : Dict{String,Any}()
        records = haskey(d, "records") ? d["records"] : NamedTuple[]

        for r in records
            m = getfield(r, :m)
            n = getfield(r, :n)
            nnz = getfield(r, :nnz)
            time_s = getfield(r, :time_s)
            dens = isnothing(m) || isnothing(n) || m == 0 || n == 0 ? missing : nnz / (m * n)

            push!(rows, (
                file        = f,
                timestamp   = get(meta, "timestamp", missing),
                problem_set = getfield(r, :problem_set),
                problem_name= getfield(r, :problem_name),
                matrix      = getfield(r, :matrix),    # "P", "A", or "AT"
                op          = getfield(r, :op),        # "P*x", "A*x", or "A'*y"
                vec         = getfield(r, :vec),       # "real" or "complex"
                m           = m,
                n           = n,
                nnz         = nnz,
                density     = dens,
                eltype      = getfield(r, :eltype),
                time_s      = time_s,
            ))
        end
    end

    return DataFrame(rows)
end

"""
    ratios_by_op(df::DataFrame) -> DataFrame

From a long DF returned by `load_spmv_results`, compute per-(problem, matrix, op)
ratios of complex/real times and attach matrix descriptors for plotting.
Columns in the output:
  problem_set, problem_name, matrix, op, m, n, nnz, density,
  time_real_s, time_complex_s, ratio_cr
"""
function ratios_by_op(df::DataFrame)
    # pivot by vec real/complex then compute ratios
    g = groupby(df, [:problem_set, :problem_name, :matrix, :op])
    rows = NamedTuple[]
    for sub in g
        # Expect up to two rows per group: vec in {real, complex}
        # Use a helper to avoid errors when group is missing one of them
        _first_or_missing(v) = isempty(v) ? missing : v[1]
        t_real    = _first_or_missing(sub[sub.vec .== "real", :time_s])
        t_complex = _first_or_missing(sub[sub.vec .== "complex", :time_s])

        m = first(sub.m)
        n = first(sub.n)
        nnz = first(sub.nnz)
        dens = first(sub.density)

        ratio = (t_real isa Missing || t_complex isa Missing) ? missing : t_complex / t_real

        push!(rows, (
            problem_set = first(sub.problem_set),
            problem_name= first(sub.problem_name),
            matrix      = first(sub.matrix),
            op          = first(sub.op),
            m           = m,
            n           = n,
            nnz         = nnz,
            density     = dens,
            time_real_s = t_real,
            time_complex_s = t_complex,
            ratio_cr    = ratio,
        ))
    end
    return DataFrame(rows)
end

"""
    plot_ratio_vs(df_ratio; x=:density, by_op=true)

Scatter plot of complex/real time ratio vs a matrix descriptor.
Arguments:
  - df_ratio: output of `ratios_by_op`
  - x: Symbol for x-axis column (e.g., :density, :nnz, :m, :n)
  - by_op: if true, separate series per operation; otherwise all together.
Returns a Plots.jl plot object.
"""
function plot_ratio_vs(df_ratio::DataFrame; x::Symbol=:density, by_op::Bool=true)
    dfp = df_ratio[.!ismissing.(df_ratio.ratio_cr) .& .!ismissing.(df_ratio[:, x]), :]

    xlabel = String(x)
    ylabel = "complex / real time"

    if by_op
        ops = unique(dfp.op)
        plt = plot()
        for op in ops
            sub = dfp[dfp.op .== op, :]
            scatter!(plt, sub[:, x], sub.ratio_cr; label=op, xlabel=xlabel, ylabel=ylabel)
        end
        return plt
    else
        return scatter(dfp[:, x], dfp.ratio_cr; label="all ops", xlabel=xlabel, ylabel=ylabel)
    end
end

"""
    example_workflow()

Quick example of how to use these helpers interactively:
    using .spmv_utils
    df  = load_spmv_results()
    rf  = ratios_by_op(df)
    plt = plot_ratio_vs(rf; x=:density, by_op=true)
    savefig(plt, joinpath(dirname(@__DIR__), "analysis", "figs", "spmv", "ratio_vs_density.png"))
"""
function example_workflow()
    df  = load_spmv_results()
    rf  = ratios_by_op(df)
    mkpath(joinpath(pwd(), "analysis", "figs", "spmv"))
    
    plt = plot_ratio_vs(rf; x=:m, by_op=true)
    out = joinpath(dirname(@__DIR__), "analysis", "figs", "spmv", "ratio_vs_density.png")
    savefig(plt, out)
    return (; df, rf, plt, out)
end
