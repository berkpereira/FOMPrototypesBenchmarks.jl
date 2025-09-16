include("spmv_utils.jl")

gr()
df  = load_spmv_results()
rf  = ratios_by_op(df)
out = joinpath(dirname(@__DIR__), "analysis", "figs", "spmv")


# scatter plot
scatter_st = paper_plot_kwargs(; column=:single, fontsize=7, fontfamily="Computer Modern", aspect=0.8, tight=true)

scatter_plt = plot_ratio_vs(rf; x=:density, by_op=true, xscale=:log10, markersize=3, alpha=0.9, outfile=nothing, scatter_st...)

# histogram
hist_st = paper_plot_kwargs(; column=:single, fontsize=7, fontfamily="Computer Modern", aspect=0.6, tight=true)

hist_plt = plot_ratio_hist(rf; column=:single, fontsize=7, fontfamily="Computer Modern", tight=true, by_op=false, bins=20, normalize=:probability, alpha=0.9, outfile=joinpath(out, "ratio_hist.pdf"), hist_st...)
