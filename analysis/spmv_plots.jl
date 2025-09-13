include("spmv_utils.jl")

df  = load_spmv_results()
rf  = ratios_by_op(df)
out = joinpath(dirname(@__DIR__), "analysis", "figs", "spmv", "ratio_vs_density.pdf")
st = paper_plot_kwargs(; column=:single, fontsize=7, fontfamily="Computer Modern", aspect=0.8, tight=true)
plt = plot_ratio_vs(rf; x=:density, by_op=true, xscale=:log10, markersize=3,
    alpha=0.9, outfile=out, st...)
