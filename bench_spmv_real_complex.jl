# I use this file to run some benchmarks of SpMV costs
# using matrices derived from QP formulations of
# least-squares problems (huber and lasso) as done in
# the benchmarks for Clarabel.
# idea is to compare the costs of computing SpMV for
# real and complex vectors.

# these benchmarks are run locally, as they are quite simple, cheap

using Logging, Random, BenchmarkTools
using LinearAlgebra, SparseArrays
using JLD2
import Dates
using FOMPrototypesBenchmarks
using FOMPrototypes


const problem_sets = [
    "sslsq"
]

const problems = vcat((
	[(ps, pname) for pname in FOMPrototypesBenchmarks.load_problem_list(ps)]
	for ps in problem_sets
)...)

nnz_safe(M) = try
    nnz(M)
catch
    count(!iszero, M)
end

resdir = "results_spmv"

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
            n = size(P, 1)
            xr = randn(n)
            xc = randn(n) .+ im * randn(n)
            yr = zeros(eltype(xr), n)
            yc = zeros(ComplexF64, n)
            mul!(yr, A, xr)
            mul!(yc, A, xc)
            mul!(yr, AT, xr)
            mul!(yc, AT, xc)
            mul!(yr, P, xr)
            mul!(yc, P, xc)
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
                # print timings to console
                @btime mul!($yr, $P, $xr)
                @btime mul!($yc, $P, $xc)

                # capture best-of measurements in seconds
                t_r = @belapsed mul!($yr, $P, $xr)
                t_c = @belapsed mul!($yc, $P, $xc)
                push!(records, (problem_set=ps, problem_name=pname, matrix="P", op="P*x", vec="real",    m=n, n=n, nnz=nnz_safe(P), eltype=string(eltype(P)), time_s=t_r))
                push!(records, (problem_set=ps, problem_name=pname, matrix="P", op="P*x", vec="complex", m=n, n=n, nnz=nnz_safe(P), eltype=string(eltype(P)), time_s=t_c))
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
            # A * x
            @btime mul!($ya, $A, $xr)
            @btime mul!($yac, $A, $xc)
            t_ax_r = @belapsed mul!($ya, $A, $xr)
            t_ax_c = @belapsed mul!($yac, $A, $xc)
            push!(records, (problem_set=ps, problem_name=pname, matrix="A", op="A*x",  vec="real",    m=m, n=n, nnz=nnz_safe(A), eltype=string(eltype(A)), time_s=t_ax_r))
            push!(records, (problem_set=ps, problem_name=pname, matrix="A", op="A*x",  vec="complex", m=m, n=n, nnz=nnz_safe(A), eltype=string(eltype(A)), time_s=t_ax_c))

            # A' * y
            AT = transpose(A)
            yr_in  = randn(m)
            yc_in  = randn(m) .+ im * randn(m)
            out_rT = zeros(n)
            out_cT = zeros(ComplexF64, n)
            @btime mul!($out_rT, $AT, $yr_in)
            @btime mul!($out_cT, $AT, $yc_in)
            t_aty_r = @belapsed mul!($out_rT, $AT, $yr_in)
            t_aty_c = @belapsed mul!($out_cT, $AT, $yc_in)
            push!(records, (problem_set=ps, problem_name=pname, matrix="AT", op="A'*y", vec="real",    m=n, n=m, nnz=nnz_safe(A), eltype=string(eltype(A)), time_s=t_aty_r))
            push!(records, (problem_set=ps, problem_name=pname, matrix="AT", op="A'*y", vec="complex", m=n, n=m, nnz=nnz_safe(A), eltype=string(eltype(A)), time_s=t_aty_c))
        else
            @warn "No A found; skipping A benchmarks"
        end

        # Persist results for this problem
        outdir = joinpath(resdir, ps, pname)
        mkpath(outdir)
        ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        outfile = joinpath(outdir, "spmv_" * ts * ".jld2")
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

run_bench_spmv()