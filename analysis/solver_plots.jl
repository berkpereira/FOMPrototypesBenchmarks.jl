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

prof_type = :relative # relative or absolute
# if this flag is true, proportion of problems solved is taken with respect
# to all problems which at least one solver could solve.
if prof_type == :relative
    scale_prof_by_solved_union = true
else # always false here
    scale_prof_by_solved_union = false
end

legend_place = :bottomright

# selection knobs
CHOSEN_PROBLEM_SETS = [
    # "mpc",
    # "sslsq",
    # "maros",
    "opf_socp",
]

RUN_PARAMS = (
    global_timeout=Inf,
    max_k_operator=20_000,
    rel_kkt_tol=1e-3, # in {1e-3, 1e-6}
    )
    
if length(CHOSEN_PROBLEM_SETS) > 1
    throw(ErrorException("Choose only one problem set at a time for ADMM profiles"))
end

# comment out wanted keys
unwanted_label_keys = [
    # "acceleration",
    "accel-memory",
    # "krylov-tries-per-mem",
    # "anderson-interval",
    "anderson-mem-type",
    "krylov-operator",
]
if unwanted_label_keys == []; unwanted_label_keys = nothing; end

# 1. Inspect available runs for the variant and keep only a subset.
inventory = index_variant_runs(
    variant=:ADMM, # choose method "variant"
    problem_sets=CHOSEN_PROBLEM_SETS, # choose problem sets
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
    run_params=RUN_PARAMS,
    include_files=true,
)
isempty(inventory) && error("No ADMM runs matching the selection were found")

# filter for just QR2 broyden type among Anderson-accelerated solvers
# also filter OUT :B krylov-operator among Krylov-accelerated solvers
abt = Symbol("anderson-broyden-type")
ko  = Symbol("krylov-operator")
acc = inventory.acceleration
abt_vals = inventory[!, abt]
ko_vals  = inventory[!, ko]

keep_mask = (acc .== "none") .| ((acc .== "krylov") .& coalesce.(ko_vals .!= "B", false)) .|
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
    agg,
    prof_type;
    metric=:min_total_time,
    scale_by_solved_union=scale_prof_by_solved_union,
)
labels, title = build_labels(String.(names(perf_time)[2:end]), unwanted_label_keys, nothing)

perf_kop = performance_profile(
    agg,
    prof_type;
    metric=:min_k_operator_final,
    # taus=logspace(2.7, 24_000., 500),
    scale_by_solved_union=scale_prof_by_solved_union,
)
labels, title = build_labels(String.(names(perf_kop)[2:end]), unwanted_label_keys, nothing)

prof_type_tag = "$prof_type"
ps_tag = isempty(CHOSEN_PROBLEM_SETS) ? "all" : join(CHOSEN_PROBLEM_SETS, "-")
rp_tag = "gt$(RUN_PARAMS.global_timeout)_k$(RUN_PARAMS.max_k_operator)_tol$(RUN_PARAMS.rel_kkt_tol)"
fname_suffix = "$(prof_type)_$(ps_tag)_$(rp_tag)"

# create plots
plt_time = plot_performance_profile(
    perf_time,
    prof_type,
    :min_total_time;
    labels=labels,
    title=title,
    legend=legend_place,
    xscale=:log10,
    outfile=joinpath(out_dir, "admm-time-profile-$(fname_suffix).pdf"),
    st...,
)
display(plt_time)

plt_kop = plot_performance_profile(
    perf_kop,
    prof_type,
    :min_k_operator_final;
    labels=labels,
    title=title,
    legend=legend_place,
    xscale=:log10,
    outfile=joinpath(out_dir, "admm-iter-profile-$(fname_suffix).pdf"),
    st...,
)
display(plt_kop)
