include("spmv_utils.jl")

problem_sets = String[
    "sslsq",
    "mpc",
    "maros",
    # "netlib_feasible"
]

gr()

time_metric = :min # in {:min, :median, :max}
df = load_spmv_results(; problem_sets=problem_sets, time_metric=time_metric)
rf = ratios_by_op(df; time_metric=time_metric)
out_dir = joinpath(dirname(@__DIR__), "analysis", "figs", "spmv")


# scatter plot
scatter_st = paper_plot_kwargs(
    ;
    column=:single,
    fontsize=5,
    fontfamily="Computer Modern",
    aspect=0.8,
    tight=true,
)

scatter_plt = plot_ratio_vs(rf; x=:density, by_op=true, xscale=:log10, markersize=3, alpha=0.9, outfile=nothing, scatter_st...)

# histogram
hist_st = paper_plot_kwargs(
    ;
    column=:single,
    fontsize=5,
    fontfamily="Computer Modern",
    aspect=0.3,
    tight=true,
    grid=true,
    xlim=(-Inf, Inf),
)

FILTER_NONSENSE = true
# filter out ratios below 1
if FILTER_NONSENSE
    rf_hist = filter(:ratio_cr => x -> ismissing(x) || x ≥ 1.0, rf)
    @info "Filtered out $(size(rf, 1) - size(rf_hist, 1)) records for histogram."
else
    rf_hist = rf
end

lo, hi = extrema(skipmissing(rf_hist.ratio_cr))
tick_start = 0.5 * floor(lo / 0.5)
tick_stop  = 0.5 * ceil(hi / 0.5)

hist_plt = plot_ratio_hist(
    rf_hist;
    column=:single,
    fontsize=5,
    fontfamily="Computer Modern",
    tight=true,
    by_op=false,
    bins=50,
    normalize=:probability,
    alpha=0.9,
    xticks = tick_start:0.25:tick_stop,
    outfile=joinpath(out_dir, "spmv_cr_ratio_hist.pdf"),
    hist_st...
)
