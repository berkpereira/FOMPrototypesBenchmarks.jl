# Utilities to load and analyze chol solve benchmark results

using JLD2
using Glob
using DataFrames

const _CHOLESOLVE_TIME_METRIC_COLUMNS = Dict(
    :min    => :time_min_s,
    :median => :time_median_s,
    :max    => :time_max_s,
)

_validate_chol_time_metric(metric::Symbol) = haskey(_CHOLESOLVE_TIME_METRIC_COLUMNS, metric) || throw(ArgumentError("Unsupported time metric: $(metric). Choose one of :min, :median, :max."))

function _resolve_chol_time_column(metric::Symbol)
    _validate_chol_time_metric(metric)
    return _CHOLESOLVE_TIME_METRIC_COLUMNS[metric]
end

"""
    load_cholsolve_results(; root="results_cholsolve",
                             problem_sets=String[],
                             time_metric::Symbol=:median) -> DataFrame

Load results from chol solve benchmarks stored under `root`. Each record
captures the rhs type (vector or matrix), matrix metadata, and timing
statistics. A convenience column `time_s` mirrors the requested
`time_metric`.
"""
function load_cholsolve_results(; root::AbstractString="results_cholsolve",
        problem_sets::AbstractVector{<:AbstractString}=String[],
        time_metric::Symbol=:median)
    time_col = _resolve_chol_time_column(time_metric)
    pattern = joinpath(root, "*", "*", "cholsolve_fullstats_*.jld2")
    files = isempty(problem_sets) ?
        glob(pattern) :
        vcat((glob(joinpath(root, ps, "*", "cholsolve_fullstats_*.jld2")) for ps in problem_sets)...)

    rows = NamedTuple[]
    for f in files
        d = load(f)
        meta    = get(d, "meta", Dict{String,Any}())
        records = get(d, "records", NamedTuple[])

        for r in records
            rhs_cols = get(r, :rhs_cols, missing)
            time_val = haskey(r, time_col) ? r[time_col] : missing
            time_per_rhs = (rhs_cols === missing || rhs_cols == 0 || time_val === missing) ? missing : time_val / rhs_cols
            push!(rows, (
                file            = f,
                timestamp       = get(meta, "timestamp", missing),
                problem_set     = get(r, :problem_set, missing),
                problem_name    = get(r, :problem_name, missing),
                rhs_type        = get(r, :rhs_type, missing),
                rhs_cols        = rhs_cols,
                n               = get(r, :n, missing),
                nnz_W           = get(r, :nnz_W, missing),
                nnz_L           = get(r, :nnz_L, missing),
                rho             = get(r, :rho, missing),
                diag_shift      = get(r, :diag_shift, missing),
                factor_attempts = get(r, :factor_attempts, missing),
                factor_time_s   = get(r, :factor_time_s, missing),
                time_min_s      = get(r, :time_min_s, missing),
                time_median_s   = get(r, :time_median_s, missing),
                time_max_s      = get(r, :time_max_s, missing),
                time_metric     = time_metric,
                time_s          = time_val,
                time_per_rhs_s  = time_per_rhs,
            ))
        end
    end

    return DataFrame(rows)
end

"""
    ratios_matrix_vs_vector(df::DataFrame; time_metric::Symbol=:median) -> DataFrame

Compute total and per-RHS timing ratios between the two-column matrix variant
and the vector variant for each problem.
"""
function ratios_matrix_vs_vector(df::DataFrame; time_metric::Symbol=:median)
    time_col = _resolve_chol_time_column(time_metric)
    g = groupby(df, [:problem_set, :problem_name, :rho, :diag_shift])
    rows = NamedTuple[]
    for sub in g
        vec_rows = filter(row -> row.rhs_type == "vector", sub)
        mat_rows = filter(row -> row.rhs_type == "matrix", sub)
        isempty(vec_rows) && continue
        isempty(mat_rows) && continue

        vec_time = first(vec_rows)[time_col]
        mat_time = first(mat_rows)[time_col]
        vec_cols = first(vec_rows).rhs_cols
        mat_cols = first(mat_rows).rhs_cols
        vec_per_rhs = first(vec_rows).time_per_rhs_s
        mat_per_rhs = first(mat_rows).time_per_rhs_s

        ratio_total = (ismissing(vec_time) || ismissing(mat_time)) ? missing : mat_time / vec_time
        ratio_per_rhs = (ismissing(vec_per_rhs) || ismissing(mat_per_rhs) || vec_per_rhs == 0) ? missing : mat_per_rhs / vec_per_rhs

        push!(rows, (
            problem_set          = first(sub.problem_set),
            problem_name         = first(sub.problem_name),
            rho                  = first(sub.rho),
            diag_shift           = first(sub.diag_shift),
            n                    = first(vec_rows).n,
            nnz_W                = first(vec_rows).nnz_W,
            nnz_L                = first(vec_rows).nnz_L,
            factor_attempts      = first(vec_rows).factor_attempts,
            factor_time_s        = first(vec_rows).factor_time_s,
            time_vector_s        = vec_time,
            time_matrix_s        = mat_time,
            time_vector_per_rhs_s= vec_per_rhs,
            time_matrix_per_rhs_s= mat_per_rhs,
            ratio_matrix_to_vec  = ratio_total,
            ratio_per_rhs        = ratio_per_rhs,
            time_metric          = time_metric,
        ))
    end
    return DataFrame(rows)
end
