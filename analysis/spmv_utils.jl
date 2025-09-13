# Utilities to load and analyze SpMV benchmark results

using JLD2
using Glob
using DataFrames
using Statistics
using Plots
using LaTeXStrings
using Measures
using LaTeXStrings

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
                        aspect::Real=0.62,
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
                        left_margin=4mm, right_margin=2mm,
                        top_margin=2mm, bottom_margin=3mm)

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
        width_pt::Union{Nothing,Real}=nothing,
        height_pt::Union{Nothing,Real}=nothing,
        aspect::Real=0.62,
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
        left_margin=4mm, right_margin=2mm,
        top_margin=2mm, bottom_margin=3mm)

    # Point (pt) units
    default_width_pt = column === :double ? (245.0*2) : 245.0
    wpt = width_pt === nothing ? default_width_pt : float(width_pt)
    hpt = height_pt === nothing ? aspect * wpt : float(height_pt)
    # Use point dimensions directly as Plots size for vector exports.
    wpx = Int(round(wpt))
    hpx = Int(round(hpt))

    # Resolve font sizes
    tickfs   = something(tickfontsize, max(6, round(Int, 0.85 * fontsize)))
    legendfs = something(legendfontsize, fontsize)
    titlefs  = something(titlefontsize, round(Int, 1.1 * fontsize))

    # Optionally tighten margins
    if tight
        left_margin = -2mm; right_margin = -1mm; top_margin = -1mm; bottom_margin = -2mm
    end

    # Build kwargs NamedTuple
    return (
        size = (wpx, hpx),
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
        if !haskey(plotkwargs, :xlims)
            xmin, xmax = extrema(dfp[:, x])
            if xmin == xmax
                δ = (abs(xmin) > 0 ? 0.01 * abs(xmin) : 1.0)
                xmin -= δ; xmax += δ
            end
            pad = (xmax - xmin) * pad_frac
            xlims!(plt, (xmin - pad, xmax + pad))
        end
        if !haskey(plotkwargs, :ylims)
            ymin, ymax = extrema(dfp[:, y])
            if ymin == ymax
                δ = (abs(ymin) > 0 ? 0.01 * abs(ymin) : 1.0)
                ymin -= δ; ymax += δ
            end
            pad = (ymax - ymin) * pad_frac
            ylims!(plt, (ymin - pad, ymax + pad))
        end
    end

    if outfile !== nothing
        out = save_pdf(plt, String(outfile))
    end
    return plt
end

"""
    plot_ratio_vs(df_ratio; x=:density, by_op=true, markersize=4, alpha=0.7,
                  xscale=:identity, yscale=:identity, legend=:best,
                  title=nothing, outfile=nothing)

Compatibility wrapper around `plot_scatter_by_group` to plot ratio vs a descriptor.
"""
function plot_ratio_vs(df_ratio::DataFrame; x::Symbol=:density, by_op::Bool=true,
        markersize::Real=4, alpha::Real=0.7, xscale::Symbol=:identity,
        yscale::Symbol=:identity, legend=:best, title=nothing,
        outfile::Union{Nothing,AbstractString}=nothing, plotkwargs...)

    group = by_op ? :op : nothing
    return plot_scatter_by_group(df_ratio; x=x, y=:ratio_cr, group=group,
        markersize=markersize, alpha=alpha, xscale=xscale, yscale=yscale,
        xlabel=String(x), ylabel="complex / real time", legend=legend,
        title=isnothing(title) ? "" : title, outfile=outfile, plotkwargs...)
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
