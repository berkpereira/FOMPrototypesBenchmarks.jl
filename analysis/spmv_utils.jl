# Utilities to load and analyze SpMV benchmark results

using JLD2
using Glob
using DataFrames
using Statistics
using Plots
using LaTeXStrings
using Measures

const _TIME_METRIC_COLUMNS = Dict(
    :min    => :time_min_s,
    :median => :time_median_s,
    :max    => :time_max_s,
)

_validate_time_metric(metric::Symbol) = haskey(_TIME_METRIC_COLUMNS, metric) || throw(ArgumentError("Unsupported time metric: $(metric). Choose one of :min, :median, :max."))

_normalize_time(val) = val === missing ? missing : float(val)

function _resolve_time_column(df::DataFrame, metric::Symbol)
    _validate_time_metric(metric)
    col = _TIME_METRIC_COLUMNS[metric]
    if col in Symbol.(names(df))
        return col
    elseif :time_s in names(df)
        return :time_s
    else
        available = filter(c -> c in Symbol.(names(df)), values(_TIME_METRIC_COLUMNS))
        if !isempty(available)
            return first(available)
        end
        throw(ArgumentError("DataFrame is missing timing columns needed for metric $(metric)."))
    end
end

"""
    load_spmv_results(; root="results_spmv", problem_sets=String[],
                         time_metric::Symbol=:median) -> DataFrame

Load all SpMV result files from `root` and return a DataFrame with one row per
recorded measurement. Each row includes problem identifiers, matrix/op metadata,
timings, and derived quantities like density. Timing columns include
`time_min_s`, `time_median_s`, `time_max_s`, and a convenience `time_s` column
that reflects the requested `time_metric` (default: median).
"""
function load_spmv_results(; root::AbstractString = "results_spmv",
        problem_sets::AbstractVector{<:AbstractString} = String[],
        time_metric::Symbol = :median)
    _validate_time_metric(time_metric)

    # Resolve path relative to repo root when called from this file
    pattern = joinpath(root, "*", "*", "spmv*.jld2")
    files = isempty(problem_sets) ?
        glob(pattern) :
        vcat((glob(joinpath(root, ps, "*", "spmv*.jld2")) for ps in problem_sets)...)

    rows = NamedTuple[]
    for f in files
        d = load(f)
        meta    = haskey(d, "meta")    ? d["meta"]    : Dict{String,Any}()
        records = haskey(d, "records") ? d["records"] : NamedTuple[]

        for r in records
            m = getfield(r, :m)
            n = getfield(r, :n)
            nnz = getfield(r, :nnz)

            dens = isnothing(m) || isnothing(n) || m == 0 || n == 0 ? missing : nnz / (m * n)

            time_legacy   = hasproperty(r, :time_s)        ? _normalize_time(getfield(r, :time_s)) : missing
            time_min      = hasproperty(r, :time_min_s)     ? _normalize_time(getfield(r, :time_min_s)) : missing
            time_median   = hasproperty(r, :time_median_s)  ? _normalize_time(getfield(r, :time_median_s)) : missing
            time_max      = hasproperty(r, :time_max_s)     ? _normalize_time(getfield(r, :time_max_s)) : missing

            time_min    === missing && (time_min    = time_legacy)
            time_median === missing && (time_median = something(time_legacy, time_min))
            time_max    === missing && (time_max    = time_legacy)

            time_s_val = if time_metric === :min
                time_min
            elseif time_metric === :max
                time_max
            else
                time_median
            end
            time_s_val === missing && (time_s_val = something(time_median, time_min, time_max, time_legacy))

            push!(rows, (
                file          = f,
                timestamp     = get(meta, "timestamp", missing),
                problem_set   = getfield(r, :problem_set),
                problem_name  = getfield(r, :problem_name),
                matrix        = getfield(r, :matrix),    # "P", "A", or "AT"
                op            = getfield(r, :op),        # "P*x", "A*x", or "A'*y"
                vec           = getfield(r, :vec),       # "real" or "complex"
                m             = m,
                n             = n,
                nnz           = nnz,
                density       = dens,
                eltype        = getfield(r, :eltype),
                time_min_s    = time_min,
                time_median_s = time_median,
                time_max_s    = time_max,
                time_s        = time_s_val,
                time_metric   = time_metric,
            ))
        end
    end

    return DataFrame(rows)
end

"""
    ratios_by_op(df::DataFrame; time_metric::Symbol=:median) -> DataFrame

From a long DF returned by `load_spmv_results`, compute per-(problem, matrix,
op) ratios of complex/real times and attach matrix descriptors for plotting.
The `time_metric` keyword controls which timing column is used (min/median/max,
default median). Columns in the output:
  problem_set, problem_name, matrix, op, m, n, nnz, density,
  time_real_s, time_complex_s, ratio_cr, time_metric
"""
function ratios_by_op(df::DataFrame; time_metric::Symbol=:median)
    _validate_time_metric(time_metric)
    time_col = _resolve_time_column(df, time_metric)

    # pivot by vec real/complex then compute ratios (keep simple and robust)
    g = groupby(df, [:problem_set, :problem_name, :matrix, :op])
    rows = NamedTuple[]
    first_or_missing(v) = isempty(v) ? missing : v[1]
    for sub in g
        # time in chosen representative metric (min, median, max), time_col
        t_real    = first_or_missing(sub[sub.vec .== "real", time_col])
        t_complex = first_or_missing(sub[sub.vec .== "complex", time_col])

        ratio = (t_real isa Missing || t_complex isa Missing) ? missing : t_complex / t_real

        push!(rows, (
            problem_set   = first(sub.problem_set),
            problem_name  = first(sub.problem_name),
            matrix        = first(sub.matrix),
            op            = first(sub.op),
            m             = first(sub.m),
            n             = first(sub.n),
            nnz           = first(sub.nnz),
            density       = first(sub.density),
            time_real_s   = t_real,
            time_complex_s= t_complex,
            ratio_cr      = ratio,
            time_metric   = time_metric,
        ))
    end
    return DataFrame(rows)
end

"""
    _ensure_pdf_path(outfile::AbstractString) -> String

Ensure the output filename ends with `.pdf`.
"""
function _ensure_pdf_path(outfile::AbstractString)
    endswith(lowercase(outfile), ".pdf") && return String(outfile)
    return outfile * ".pdf"
end

"""
    op_label(op; latex::Bool=false)

Return a display label for an operation identifier like "P*x", "A*x", or
"A'*y". When `latex=true`, returns a LaTeXString with math formatting:
  P*x  -> L"P x"
  A*x  -> L"A x"
  A'*y -> L"A^T y"
Falls back to a reasonable string if op is unrecognized.
"""
function op_label(op; latex::Bool=false)
    s = String(op)
    if !latex
        return s
    end
    if s == "P*x"
        return L"P x"
    elseif s == "A*x"
        return L"A x"
    elseif s == "A'*y"
        return L"A^T y"
    else
        # simple fallback: remove * and show as math text
        return LaTeXString(replace(s, "*" => " "))
    end
end

"""
    paper_plot_kwargs(; column::Symbol=:single,
                        width_pt::Union{Nothing,Real}=nothing,
                        height_pt::Union{Nothing,Real}=nothing,
                        aspect::Real=1.0,
                        fontfamily::AbstractString="Computer Modern",
                        fontsize::Real=8,
                        tickfontsize::Union{Nothing,Real}=nothing,
                        legendfontsize::Union{Nothing,Real}=nothing,
                        titlefontsize::Union{Nothing,Real}=nothing,
                        lw::Real=1,
                        ms::Real=3,
                        grid::Bool=false,
                        tight::Bool=false,
                        framestyle::Symbol=:box,
                        left_margin=0mm, right_margin=0mm,
                        top_margin=0mm, bottom_margin=0mm)

Return a NamedTuple of Plots.jl kwargs suitable for “paper-ready” figures.

- column: `:single` or `:double` (used if `width_pt` is not provided). Defaults
  to ~3.5 in (≈245–252 pt) for `:single` and ~7.16 in (≈500+ pt) for `:double`.
- width_pt/height_pt: explicit figure size in points (1 pt = 1/72 in). If one
  dimension is missing, the other is computed from `aspect` (height = aspect × width).
- fontfamily: try "Computer Modern" for LaTeX-like look; adjust if unavailable.
- fontsize: base font size (guides). Tick/legend/title sizes default relative to this
  unless set explicitly.

For PDF/SVG backends (vector), Plots/GR effectively treats `size=(w,h)` as point
units in the exported document. We therefore set `size=(round(Int,width_pt), round(Int,height_pt))`
directly with no DPI, so the PDF's bounding box matches your typographic width.
"""
function paper_plot_kwargs(; column::Symbol=:single,
        xlim::Tuple{Float64, Float64}=(-Inf, Inf),
        width_pt::Union{Nothing,Real}=nothing,
        height_pt::Union{Nothing,Real}=nothing,
        aspect::Real=1.0,
        fontfamily::AbstractString="Computer Modern",
        fontsize::Real=8,
        tickfontsize::Union{Nothing,Real}=nothing,
        legendfontsize::Union{Nothing,Real}=nothing,
        titlefontsize::Union{Nothing,Real}=nothing,
        lw::Real=1,
        ms::Real=3,
        grid::Bool=false,
        tight::Bool=false,
        framestyle::Symbol=:box,
        left_margin=0mm, right_margin=0mm,
        top_margin=0mm, bottom_margin=0mm)

    # Compute figure size in points, mapped directly to Plots size.
    default_wpt = column === :double ? 490.0 : 245.0
    wpt = float(something(width_pt, default_wpt))
    hpt = float(something(height_pt, aspect * wpt))
    size_px = (Int(round(wpt)), Int(round(hpt)))

    # Font sizes (keep sensible defaults, allow overrides)
    tickfs   = Int(round(something(tickfontsize, 0.85 * fontsize)))
    legendfs = Int(round(something(legendfontsize, fontsize)))
    titlefs  = Int(round(something(titlefontsize, 1.1 * fontsize)))

    # Optional compact margins
    if tight
        left_margin = -1mm; right_margin = -1mm; top_margin = -1mm; bottom_margin = -0.6mm
    end

    return (
        size = size_px,
        guidefont = Plots.font(fontsize, fontfamily),
        tickfont = Plots.font(tickfs, fontfamily),
        legendfont = Plots.font(legendfs, fontfamily),
        titlefont = Plots.font(titlefs, fontfamily),
        lw = lw,
        ms = ms,
        grid = grid,
        framestyle = framestyle,
        left_margin = left_margin,
        right_margin = right_margin,
        top_margin = top_margin,
        bottom_margin = bottom_margin,
    )
end

"""
    apply_paper_style!(plt; kwargs...)

Apply paper-ready style kwargs to an existing Plots.jl plot `plt`.
Typical usage:
    st = paper_plot_kwargs(; column=:single, fontsize=8)
    apply_paper_style!(plt; st...)
"""
function apply_paper_style!(plt; kwargs...)
    plot!(plt; kwargs...)
    return plt
end

"""
    save_pdf(plt, outfile; crop=false, crop_tool=:auto, crop_margin_pt=1.0)

Ensure `.pdf` extension, create parent directory if needed, and save.
Optionally try to crop whitespace using `pdfcrop` if present.
Returns the final path.
"""
function save_pdf(plt, outfile::AbstractString; crop::Bool=false, crop_tool::Symbol=:auto, crop_margin_pt::Real=1.0)
    out = _ensure_pdf_path(String(outfile))
    mkpath(dirname(out))
    savefig(plt, out)
    if crop
        try
            _ = crop_pdf!(out; tool=crop_tool, margin_pt=crop_margin_pt)
        catch err
            @info "PDF crop failed; returning uncropped file" error=err
        end
    end
    return out
end

"""
    crop_pdf!(path; tool=:auto, margin_pt=1.0) -> Bool

Try to crop whitespace from a PDF using external tools if available.
Prefers `pdfcrop` (TeX Live). Returns true on success, false otherwise.
Does nothing if no tool is found.
"""
function crop_pdf!(path::AbstractString; tool::Symbol=:auto, margin_pt::Real=1.0)
    inpath = abspath(path)
    crop = Sys.which("pdfcrop")
    if (tool in (:auto, :pdfcrop)) && crop !== nothing
        outpath = inpath * ".tmp.pdf"
        margins = string(margin_pt, " ", margin_pt, " ", margin_pt, " ", margin_pt)
        cmd = `$(crop) --hires --margins $margins $inpath $outpath`
        try
            run(cmd)
            mv(outpath, inpath; force=true)
            return true
        catch
            isfile(outpath) && rm(outpath; force=true)
            return false
        end
    end
    return false
end

"""
    plot_scatter_by_group(df; x=:density, y=:ratio_cr, group=:op, markersize=4,
                          alpha=1.0, xscale=:identity, yscale=:identity,
                          xlabel=nothing, ylabel=nothing, legend=:best,
                          title=nothing, outfile=nothing,
                          use_latex_op_labels=true,
                          tight_axis=false, pad_frac=0.02)

Generic scatter plot helper with flexible x/y quantities and optional grouping.
- df: DataFrame with columns for x, y, and optionally group
- x, y: Symbols naming columns in df
- group: Symbol or nothing; different series per unique group value
- markersize, alpha: styling controls
- xscale, yscale: :identity or :log10
- outfile: if provided, saves a vector PDF to this path (ensuring .pdf)

Returns the Plots.jl plot object.
"""
function plot_scatter_by_group(df::DataFrame; x::Symbol=:density, y::Symbol=:ratio_cr,
        group::Union{Symbol,Nothing}=:op, markersize::Real=4, alpha::Real=1.0,
        xscale::Symbol=:identity, yscale::Symbol=:identity,
        xlabel=nothing, ylabel=nothing, legend=:best, title=nothing,
        outfile::Union{Nothing,AbstractString}=nothing,
        use_latex_op_labels::Bool=true,
        tight_axis::Bool=false, pad_frac::Real=0.02, plotkwargs...)

    @assert hasproperty(df, x) "DataFrame is missing x column $(x)"
    @assert hasproperty(df, y) "DataFrame is missing y column $(y)"
    if group !== nothing
        @assert hasproperty(df, group) "DataFrame is missing group column $(group)"
    end

    # filter out rows with missing x/y
    mask = .!ismissing.(df[:, x]) .& .!ismissing.(df[:, y])
    dfp = df[mask, :]

    xlabel === nothing && (xlabel = String(x))
    ylabel === nothing && (ylabel = String(y))

    plt = plot(xscale=xscale, yscale=yscale, legend=legend, title=title; plotkwargs...)
    if group === nothing
        scatter!(plt, dfp[:, x], dfp[:, y]; label="", markersize=markersize, alpha=alpha,
            xlabel=xlabel, ylabel=ylabel, plotkwargs...)
    else
        groups = unique(dfp[:, group])
        for gval in groups
            sub = dfp[dfp[:, group] .== gval, :]
            lab = group === :op ? op_label(gval; latex=use_latex_op_labels) : string(gval)
            scatter!(plt, sub[:, x], sub[:, y]; label=lab, markersize=markersize, alpha=alpha,
                xlabel=xlabel, ylabel=ylabel, plotkwargs...)
        end
    end

    # Optionally tighten axes to data bounds with a small padding
    if tight_axis && !isempty(dfp)
        padded_limits(v) = begin
            lo, hi = extrema(v)
            if lo == hi
                δ = (abs(lo) > 0 ? 0.01 * abs(lo) : 1.0)
                lo -= δ; hi += δ
            end
            pad = (hi - lo) * pad_frac
            (lo - pad, hi + pad)
        end
        !haskey(plotkwargs, :xlims) && xlims!(plt, padded_limits(dfp[:, x]))
        !haskey(plotkwargs, :ylims) && ylims!(plt, padded_limits(dfp[:, y]))
    end

    if outfile !== nothing
        out = save_pdf(plt, String(outfile))
    end
    return plt
end

"""
    plot_ratio_vs(data; x=:density, by_op=true, markersize=4, alpha=0.7,
                  xscale=:identity, yscale=:identity, legend=:best,
                  title=nothing, outfile=nothing, time_metric=:median)

Compatibility wrapper around `plot_scatter_by_group` to plot ratio vs a descriptor.
`data` may be the long results DataFrame or a precomputed ratio table. When
given the long form, ratios are computed on the fly using the requested
`time_metric`.
"""
function plot_ratio_vs(df_ratio::DataFrame; x::Symbol=:density, by_op::Bool=true,
        markersize::Real=4, alpha::Real=0.7, xscale::Symbol=:identity,
        yscale::Symbol=:identity, legend=:best, title=nothing,
        outfile::Union{Nothing,AbstractString}=nothing,
        time_metric::Symbol=:median, plotkwargs...)

    _validate_time_metric(time_metric)
    df_ratio = :ratio_cr in Symbol.(names(df_ratio)) ? df_ratio : ratios_by_op(df_ratio; time_metric=time_metric)

    group = by_op ? :op : nothing
    return plot_scatter_by_group(df_ratio; x=x, y=:ratio_cr, group=group,
        markersize=markersize, alpha=alpha, xscale=xscale, yscale=yscale,
        xlabel=String(x), ylabel="complex / real time", legend=legend,
        title=isnothing(title) ? "" : title, outfile=outfile, plotkwargs...)
end

"""
    plot_hist_by_group(df; value=:ratio_cr, group=:op, bins=nothing,
                       normalize=:none, alpha=0.7, legend=:best, title=nothing,
                       xlabel=nothing, ylabel=nothing, outfile=nothing,
                       use_latex_op_labels=true)

Generic histogram helper for a single value column, optionally grouped into
overlaid series (e.g., by op). Returns the Plots.jl plot object.
"""
function plot_hist_by_group(df::DataFrame; value::Symbol=:ratio_cr,
        group::Union{Symbol,Nothing}=:op, bins=nothing,
        normalize::Symbol=:none, alpha::Real=0.7, legend=:best, title=nothing,
        xlabel=nothing, ylabel=nothing,
        outfile::Union{Nothing,AbstractString}=nothing,
        use_latex_op_labels::Bool=true, plotkwargs...)

    @assert hasproperty(df, value) "DataFrame is missing column $(value)"
    if group !== nothing
        @assert hasproperty(df, group) "DataFrame is missing group column $(group)"
    end

    # filter out rows with missing values
    mask = .!ismissing.(df[:, value])
    dfp = df[mask, :]

    xlabel === nothing && (xlabel = String(value))
    if ylabel === nothing
        ylabel = normalize === :none ? "count" : String(normalize)
    end

    plt = plot(legend=legend, title=title; plotkwargs...)

    hist!(v; lbl) = begin
        if bins === nothing
            histogram!(
                plt, v;
                alpha=alpha, normalize=normalize,
                label=lbl, xlabel=xlabel, ylabel=ylabel, plotkwargs...
            )
        else
            histogram!(
                plt, v;
                bins=bins, alpha=alpha, normalize=normalize,
                label=lbl, xlabel=xlabel, ylabel=ylabel, plotkwargs...
            )
        end
    end

    if group === nothing
        hist!(dfp[:, value]; lbl="")
    else
        groups = unique(dfp[:, group])
        for gval in groups
            sub = dfp[dfp[:, group] .== gval, :]
            lab = group === :op ? op_label(gval; latex=use_latex_op_labels) : string(gval)
            hist!(sub[:, value]; lbl=lab)
        end
    end

    if outfile !== nothing
        out = save_pdf(plt, String(outfile))
    end
    return plt
end

"""
    plot_ratio_hist(data; by_op=true, bins=nothing, normalize=:none,
                    alpha=0.7, legend=:best, title=nothing, outfile=nothing,
                    time_metric=:median)

Compatibility wrapper to plot histograms of complex/real time ratios. `data`
may be the long results DataFrame or a precomputed ratio table. When given the
long form, ratios are computed using `time_metric`.
"""
function plot_ratio_hist(df_ratio::DataFrame; by_op::Bool=true, bins=nothing,
        normalize::Symbol=:none, alpha::Real=0.7, legend=:best, title=nothing,
        outfile::Union{Nothing,AbstractString}=nothing,
        time_metric::Symbol=:median, plotkwargs...)

    _validate_time_metric(time_metric)
    df_ratio = :ratio_cr in Symbol.(names(df_ratio)) ? df_ratio : ratios_by_op(df_ratio; time_metric=time_metric)

    group = by_op ? :op : nothing
    return plot_hist_by_group(df_ratio; value=:ratio_cr, group=group, bins=bins,
        normalize=normalize, alpha=alpha, legend=legend,
        title=isnothing(title) ? "" : title, xlabel="complex / real time",
        outfile=outfile, plotkwargs...)
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
    out = joinpath(dirname(@__DIR__), "analysis", "figs", "spmv", "ratio_vs_density.pdf")
    st = paper_plot_kwargs(; column=:single, fontsize=10, fontfamily="Computer Modern",
        aspect=0.6, tight=true)
    plt = plot_ratio_vs(rf; x=:density, by_op=true, xscale=:log10, markersize=3,
        alpha=0.9, outfile=out, st...)
    return (; df, rf, plt, out)
end
