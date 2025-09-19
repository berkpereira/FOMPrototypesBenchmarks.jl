include("solver_utils.jl")

gr()

# shared publication styling helpers.
st = paper_plot_kwargs(
    ;
    column=:single,
    fontsize=5,
    tight=true,
    aspect=0.6,
    lw=2.0,
    grid=true,
)

# fig save directory
out_dir = joinpath(dirname(@__DIR__), "analysis", "figs", "solver")
mkpath(out_dir)

# selection knobs
chosen_problem_sets = [
    # "sslsq",
    # "maros",
    # "mpc",
    "netlib_feasible",
]

run_params = (
    global_timeout=Inf,
    max_k_operator=50_000,
    rel_kkt_tol=1e-3,
)

unwanted_label_keys = [ # comment out wanted keys
    # "acceleration",
    "accel-memory",
    # "krylov-tries-per-mem",
    # "anderson-interval",
    "anderson-mem-type",
]
if unwanted_label_keys == []; unwanted_label_keys = nothing; end

# 1. Inspect available runs for the variant and keep only a subset.
inventory = index_variant_runs(
    variant=:ADMM, # choose method "variant"
    problem_sets=chosen_problem_sets, # choose problem sets
    method_traits=( # choose subset of method traits, eg accel values
        acceleration=[
            "anderson",
            "krylov",
            "none",
        ],
        rho=[
            0.1,
        ]
    ),
    run_params=run_params,
    include_files=true,
)
isempty(inventory) && error("No ADMM runs matching the selection were found")

# filter for just QR2 broyden type among Anderson-accelerated solvers
abt = Symbol("anderson-broyden-type")
acc = inventory.acceleration
abt_vals = inventory[!, abt]
keep_mask = (acc .== "none") .| (acc .== "krylov") .|
    ((acc .== "anderson") .& coalesce.(abt_vals .== "QR2", false))
selected = inventory[keep_mask, :]
isempty(selected) && error("Filtered selection is empty")

# 2. Load the raw replicate files for the chosen runs.
files = collect(Iterators.flatten(selected.files))
isempty(files) && error("Selected runs do not contain replicate files")

df = load_results(files=files)
add_trait_columns!(df)
add_run_param_columns!(df)

# 3. Aggregate the replicates and prepare the profile metric.
agg = aggregate_replicates(df)
perf_time = performance_profile(
    agg;
    metric=:min_total_time,
    taus=1.0:0.1:15,
)
labels, title = build_labels(String.(names(perf_time)[2:end]), unwanted_label_keys, nothing)

perf_kop = performance_profile(
    agg;
    metric=:min_k_operator_final,
    taus=0.1:0.1:15,
)
labels, title = build_labels(String.(names(perf_kop)[2:end]), unwanted_label_keys, nothing)

ps_tag = isempty(chosen_problem_sets) ? "all" : join(chosen_problem_sets, "-")
rp_tag = "gt$(run_params.global_timeout)_k$(run_params.max_k_operator)_tol$(run_params.rel_kkt_tol)"
fname_suffix = "$(ps_tag)_$(rp_tag)"

# create plots
plt_time = plot_performance_profile(
    perf_time;
    labels=labels,
    title=title,
    xlabel="Time performance ratio " * L"\tau",
    legend=:bottomright,
    ylims=(0, 1.05),
    xlims=(1.0, Inf),
    xscale=:log10,
    outfile=joinpath(out_dir, "admm-time-profile-$(fname_suffix).pdf"),
    st...,
)
display(plt_time)

plt_kop = plot_performance_profile(
    perf_kop;
    labels=labels,
    title=title,
    xlabel="Iterations performance ratio " * L"\tau",
    legend=:bottomright,
    ylims=(0, 1.05),
    xlims=(1.0, Inf),
    xscale=:log10,
    outfile=joinpath(out_dir, "admm-iter-profile-$(fname_suffix).pdf"),
    st...,
)
display(plt_kop)
