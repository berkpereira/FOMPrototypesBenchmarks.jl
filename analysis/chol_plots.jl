include("chol_utils.jl")
include("spmv_utils.jl")
using Plots

gr()

# toggle font: "Computer Modern" for papers, "Helvetica" for posters
const FONTFAMILY = "Helvetica"

problem_sets = String[
    "sslsq",
    "mpc",
    # "maros",
]

time_metric = :min

df = load_cholsolve_results(; problem_sets=problem_sets, time_metric=time_metric)
isempty(df) && error("No chol solve results found for the current selection.")

rf = ratios_matrix_vs_vector(df; time_metric=time_metric)
isempty(rf) && error("Could not compute ratios; ensure both vector and matrix measurements are available.")

out_dir = joinpath(dirname(@__DIR__), "analysis", "figs", "cholsolve")
mkpath(out_dir)

hist_st = paper_plot_kwargs(
    ;
    column=:single,
    fontsize=5,
    fontfamily=FONTFAMILY,
    aspect=0.3,
    tight=true,
    grid=true,
)

FILTER_BELOW_ONE = true
if FILTER_BELOW_ONE
    rf_hist = filter(:ratio_matrix_to_vec => x -> ismissing(x) || x ≥ 1.0, rf)
    removed = size(rf, 1) - size(rf_hist, 1)
    removed > 0 && @info "Filtered out $removed records with ratio < 1 for histogram."
else
    rf_hist = rf
end

ratio_values = collect(skipmissing(rf_hist.ratio_matrix_to_vec))
isempty(ratio_values) && error("No valid ratio data available for histogram.")

tick_step = 0.25
lo, hi = extrema(ratio_values)
tick_start = tick_step * floor(lo / tick_step)
tick_stop  = tick_step * ceil(hi / tick_step)
xticks = tick_start ≤ tick_stop ? (tick_start:tick_step:tick_stop) : nothing

hist_plot = histogram(
    ratio_values;
    bins=30,
    normalize=:probability,
    color="#002147",
    alpha=0.85,
    xlabel="complex / real time",
    ylabel="probability",
    # title="Chol solve timing ratios (time metric: $time_metric)",
    legend=false,
    xticks=xticks,
    hist_st...,
)

outfile = joinpath(out_dir, "cholsolve_two_vs_one_rhs_ratio_hist_$(time_metric).pdf")
save_pdf(hist_plot, outfile)
display(hist_plot)
