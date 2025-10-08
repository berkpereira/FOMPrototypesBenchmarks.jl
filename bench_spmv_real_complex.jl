# I use this file to run some benchmarks of SpMV costs
# using matrices derived from QP formulations of
# least-squares problems (huber and lasso) as done in
# the benchmarks for Clarabel.
# idea is to compare the costs of computing SpMV for
# real and complex vectors.

using Logging, Random, BenchmarkTools
using LinearAlgebra, SparseArrays
using JLD2
import Dates
using FOMPrototypesBenchmarks
using FOMPrototypes

const problem_sets = [
    # "sslsq",
    "mpc",
    "maros",
    # "netlib_feasible",
]

const problems = vcat((
	[(ps, pname) for pname in FOMPrototypesBenchmarks.load_problem_list(ps, :spmv)]
	for ps in problem_sets
)...)

nnz_safe(M) = try
    nnz(M)
catch
    count(!iszero, M)
end

resdir = "results_spmv"

estimate_value(est) = hasproperty(est, :value) ? getproperty(est, :value) : est

function extract_time_stats(trials::BenchmarkTools.Trial)
    min_est = BenchmarkTools.minimum(trials)
    median_est = BenchmarkTools.median(trials)
    max_est = BenchmarkTools.maximum(trials)
    return (
        time_min_s = estimate_value(min_est.time) * 1.0e-9,
        time_median_s = estimate_value(median_est.time) * 1.0e-9,
        time_max_s = estimate_value(max_est.time) * 1.0e-9,
    )
end

# --- main ------------------------------------------------------------------
function run_bench_spmv()
    Random.seed!(42)
    compiled = false # flag for initial compilation path
    @info "Will iterate over $(length(problems)) problems' data."
    for (ps, pname) in problems
        @info "Fetching problem: $ps / $pname"
        prob = FOMPrototypes.fetch_data(ps, string(pname))
        P, A = prob.P, prob.A

        # check types
        @info "Type of P: $(typeof(P))"
        @info "Type of A: $(typeof(A))"

        # first run just for compilation
        if !compiled
            @info "Running initial mul! calls for compilation."
            AT = transpose(A)
            m, n = size(A)
            xr = randn(n)
            xr_out = randn(n)
            xc = randn(n) .+ im * randn(n)
            xc_out = randn(n) .+ im * randn(n)
            
            yr = zeros(eltype(xr), m)
            yc = zeros(ComplexF64, m)
            
            mul!(yr, A, xr)
            mul!(yc, A, xc)
            mul!(xr, AT, yr)
            mul!(xc, AT, yc)
            mul!(xr_out, P, xr)
            mul!(xc_out, P, xc)
            compiled = true
        end

        # Records for this problem
        records = NamedTuple[]

        # Benchmark P*x (square n×n)
        if P !== nothing
            nP1, nP2 = size(P)
            if nP1 == nP2
                n = nP1
                xr = randn(n)
                xc = randn(n) .+ im * randn(n)
                yr = zeros(eltype(xr), n)
                yc = zeros(ComplexF64, n)

                @info "P: size=$(size(P)), eltype=$(eltype(P))"
                stats_r = extract_time_stats(@benchmark mul!($yr, $P, $xr))
                stats_c = extract_time_stats(@benchmark mul!($yc, $P, $xc))
                push!(records, (problem_set=ps, problem_name=pname, matrix="P", op="P*x", vec="real",    m=n, n=n, nnz=nnz_safe(P), eltype=string(eltype(P)), stats_r...))
                push!(records, (problem_set=ps, problem_name=pname, matrix="P", op="P*x", vec="complex", m=n, n=n, nnz=nnz_safe(P), eltype=string(eltype(P)), stats_c...))
            else
                @warn "P is not square (?!), skipping" size=size(P)
            end
        else
            @warn "No P found; skipping P benchmarks"
        end

        # Benchmark A*x (m×n) and A'*y
        if A !== nothing
            m, n = size(A)
            xr = randn(n)
            xc = randn(n) .+ im * randn(n)
            ya = zeros(m)
            yac = zeros(ComplexF64, m)

            @info "A: size=$(size(A)), eltype=$(eltype(A))"
            stats_ax_r = extract_time_stats(@benchmark mul!($ya, $A, $xr))
            stats_ax_c = extract_time_stats(@benchmark mul!($yac, $A, $xc))
            push!(records, (problem_set=ps, problem_name=pname, matrix="A", op="A*x",  vec="real",    m=m, n=n, nnz=nnz_safe(A), eltype=string(eltype(A)), stats_ax_r...))
            push!(records, (problem_set=ps, problem_name=pname, matrix="A", op="A*x",  vec="complex", m=m, n=n, nnz=nnz_safe(A), eltype=string(eltype(A)), stats_ax_c...))

            # A' * y
            AT = transpose(A)
            yr_in  = randn(m)
            yc_in  = randn(m) .+ im * randn(m)
            out_rT = zeros(n)
            out_cT = zeros(ComplexF64, n)
            stats_aty_r = extract_time_stats(@benchmark mul!($out_rT, $AT, $yr_in))
            stats_aty_c = extract_time_stats(@benchmark mul!($out_cT, $AT, $yc_in))
            push!(records, (problem_set=ps, problem_name=pname, matrix="AT", op="A'*y", vec="real",    m=n, n=m, nnz=nnz_safe(A), eltype=string(eltype(A)), stats_aty_r...))
            push!(records, (problem_set=ps, problem_name=pname, matrix="AT", op="A'*y", vec="complex", m=n, n=m, nnz=nnz_safe(A), eltype=string(eltype(A)), stats_aty_c...))
        else
            @warn "No A found; skipping A benchmarks"
        end

        # Persist results for this problem
        outdir = joinpath(resdir, ps, pname)
        mkpath(outdir)
        ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        outfile = joinpath(outdir, "spmv_fullstats_" * ts * ".jld2")
        meta = Dict(
            "julia_version" => string(VERSION),
            "timestamp"     => ts,
            "problem_set"   => ps,
            "problem_name"  => pname,
        )
        @info "Saving $(length(records)) records to $outfile"
        @save outfile meta records
    end
end

start_time = Dates.now()

run_bench_spmv()

@info "🎉 spmv benchmark finished!"
elapsed_time = Dates.now() - start_time
t = Dates.Time(0) + elapsed_time
@info "Elapsed time: $(Dates.format(t, "HH:MM:SS.s"))"
