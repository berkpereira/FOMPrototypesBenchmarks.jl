include("spmv_utils.jl")

problem_sets = String[
    "sslsq",
    # "mpc",
    # "maros",
    # "netlib_feaisble"
]

gr()

time_metric = :median # in {:min, :median, :max}
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

hist_plt = plot_ratio_hist(
    rf;
    column=:single,
    fontsize=5,
    fontfamily="Computer Modern",
    tight=true,
    by_op=false,
    # bins=30,
    normalize=:probability,
    alpha=0.9,
    outfile=joinpath(out_dir, "spmv_cr_ratio_hist.pdf"),
    hist_st...
)
