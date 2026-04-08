# Benchmark forward/backward solve costs using custom sparse Cholesky routines.

using Logging, Random, BenchmarkTools
using LinearAlgebra, SparseArrays
using SuiteSparse
using JLD2
import Dates
using FOMPrototypes
using FOMPrototypesBenchmarks
import FOMPrototypes: sparse_cholmod_solve!

const problem_sets = [
    # "sslsq",
    "mpc",
    "maros",
]

const problems = vcat((
    [(ps, pname) for pname in FOMPrototypesBenchmarks.load_problem_list(ps, :cholsolve)]
    for ps in problem_sets
)...)

const resdir = "results_cholsolve"
const DEFAULT_RHO = 0.1
const DIAG_SHIFT_SCHEDULE = (nothing, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4)

nnz_safe(M) = try
    nnz(M)
catch
    count(!iszero, M)
end

estimate_value(est) = hasproperty(est, :value) ? getproperty(est, :value) : est

function extract_time_stats(trial::BenchmarkTools.Trial)
    min_est = BenchmarkTools.minimum(trial)
    median_est = BenchmarkTools.median(trial)
    max_est = BenchmarkTools.maximum(trial)
    return (
        time_min_s = estimate_value(min_est.time) * 1.0e-9,
        time_median_s = estimate_value(median_est.time) * 1.0e-9,
        time_max_s = estimate_value(max_est.time) * 1.0e-9,
    )
end

function _sparse_square(P, n::Integer)
    if P === nothing
        return spzeros(n, n)
    elseif P isa UniformScaling
        return spdiagm(0 => fill(Float64(P.λ), n))
    else
        S = sparse(P)
        size(S, 1) == n || error("P has incompatible dimensions: expected $n, got $(size(S,1))")
        size(S, 2) == n || error("P has incompatible dimensions: expected $n, got $(size(S,2))")
        return S
    end
end

function assemble_system_matrix(P, A, rho::Float64)
    @assert rho > 0.0 "rho must be positive"
    if A === nothing
        P === nothing && error("Problem data must contain at least one of P or A.")
        P isa UniformScaling && error("Cannot infer dimension from UniformScaling P without A.")
        n = size(P, 1)
        size(P, 2) == n || error("P must be square when A is missing.")
        return Float64.(_sparse_square(P, n))
    end

    n = size(A, 2)
    W = _sparse_square(P, n)

    AtA = sparse(transpose(A) * A)
    W = sparse(W + rho * AtA)
    
    return Float64.(W)
end

function factorize_with_regularization(W::SparseMatrixCSC{Float64,Int};
        shifts::Tuple=DIAG_SHIFT_SCHEDULE)
    F = nothing
    last_error = nothing
    shift_used = 0.0
    t_start = 0.0
    attempts = 0
    for (idx, shift) in pairs(shifts)
        attempts = idx
        t_start = time()
        try
            F = shift === nothing ? SparseArrays.cholesky(W) : SparseArrays.cholesky(W; shift=shift)
            shift_used = shift === nothing ? 0.0 : float(shift)
            break
        catch err
            last_error = err
            if idx < length(shifts)
                next_shift = shifts[idx + 1]
                next_msg = next_shift === nothing ? "δ = 0.0" : "δ = $(next_shift)"
                if idx == 1
                    @warn "Cholesky failed; retrying with $next_msg" exception=err
                else
                    curr = shift === nothing ? 0.0 : shift
                    @warn "Cholesky failed even with shift δ=$(curr); retrying with $next_msg" exception=err
                end
            else
                rethrow(err)
            end
        end
    end
    F === nothing && rethrow(last_error)
    factor_time = time() - t_start
    Lsp = sparse(F.L)
    perm = Vector{Int64}(F.p)
    n = size(W, 1)
    W_factored = shift_used == 0.0 ? W : sparse(W + spdiagm(0 => fill(shift_used, n)))
    return (
        matrix=W_factored,
        Lsp=Lsp,
        perm=perm,
        inv_perm=invperm(perm),
        shift=shift_used,
        attempts=attempts,
        factor_time_s=factor_time,
    )
end

function benchmark_vector_solve(Lsp, perm, inv_perm, rhs_template::Vector{Float64}, scratch::Vector{Float64})
    x = copy(rhs_template)
    trial = @benchmark begin
        copyto!($x, $rhs_template)
        fill!($scratch, 0.0)
        sparse_cholmod_solve!($Lsp, $perm, $inv_perm, $x, $scratch)
    end
    return extract_time_stats(trial)
end

function benchmark_matrix_solve(Lsp, perm, inv_perm, rhs_template::Matrix{Float64}, temp_n_vec::Vector{ComplexF64}, perm_scratch::Vector{ComplexF64})
    x = copy(rhs_template)
    trial = @benchmark begin
        copyto!($x, $rhs_template)
        fill!($temp_n_vec, 0.0 + 0.0im)
        fill!($perm_scratch, 0.0 + 0.0im)
        sparse_cholmod_solve!($Lsp, $perm, $inv_perm, $x, $temp_n_vec, $perm_scratch)
    end
    return extract_time_stats(trial)
end

"""
    run_bench_cholsolve(; rho::Float64=DEFAULT_RHO)

Run the chol solve benchmarks for the configured `problem_sets`. The routine
forms `W = P + ρ A' A`, applies a diagonal regularisation schedule if needed,
and times the custom `sparse_cholmod_solve!` for a single RHS vector and a
matrix with two columns. Results are saved under `$(resdir)` using the same
layout as the SpMV benchmarks. Adjust `problem_sets`, `DEFAULT_RHO`, or
`DIAG_SHIFT_SCHEDULE` near the top of this file to explore different scenarios.
"""
function run_bench_cholsolve(; rho::Float64=DEFAULT_RHO)
    Random.seed!(42)
    compiled = false
    mkpath(resdir)
    @info "Will iterate over $(length(problems)) problems' data."
    for (ps, pname) in problems
        @info "Fetching problem: $ps / $pname"
        prob = FOMPrototypes.fetch_data(ps, string(pname))
        P, A = prob.P, prob.A

        if A === nothing
            throw(ErrorException("Cannot run cholsolve benchmark without A matrix"))
            continue
        end

        W = assemble_system_matrix(P, A, rho)
        factor = factorize_with_regularization(W)

        Lsp = factor.Lsp
        perm = factor.perm
        inv_perm = factor.inv_perm
        diag_shift_used = factor.shift
        attempts = factor.attempts
        factor_time_s = factor.factor_time_s
        W_factored = factor.matrix

        n = size(Lsp, 1)
        temp_n_vec = zeros(ComplexF64, n)
        perm_scratch = zeros(ComplexF64, n)
        vec_scratch = zeros(Float64, n)

        rhs_vec = randn(n)
        rhs_mat = randn(n, 2)

        # warm-up for compilation once
        if !compiled
            xvec_tmp = copy(rhs_vec)
            xmat_tmp = copy(rhs_mat)
            fill!(vec_scratch, 0.0)
            fill!(temp_n_vec, zero(eltype(temp_n_vec)))
            fill!(perm_scratch, zero(eltype(perm_scratch)))
            sparse_cholmod_solve!(Lsp, perm, inv_perm, xvec_tmp, vec_scratch)
            sparse_cholmod_solve!(Lsp, perm, inv_perm, xmat_tmp, temp_n_vec, perm_scratch)
            compiled = true
        end

        records = NamedTuple[]

        stats_vec = benchmark_vector_solve(Lsp, perm, inv_perm, rhs_vec, vec_scratch)
        push!(records, (
            problem_set=ps,
            problem_name=pname,
            rhs_type="vector",
            rhs_cols=1,
            n=n,
            nnz_W=nnz_safe(W_factored),
            nnz_L=nnz_safe(Lsp),
            rho=rho,
            diag_shift=diag_shift_used,
            factor_attempts=attempts,
            factor_time_s=factor_time_s,
            stats_vec...
        ))

        stats_mat = benchmark_matrix_solve(Lsp, perm, inv_perm, rhs_mat, temp_n_vec, perm_scratch)
        push!(records, (
            problem_set=ps,
            problem_name=pname,
            rhs_type="matrix",
            rhs_cols=2,
            n=n,
            nnz_W=nnz_safe(W_factored),
            nnz_L=nnz_safe(Lsp),
            rho=rho,
            diag_shift=diag_shift_used,
            factor_attempts=attempts,
            factor_time_s=factor_time_s,
            stats_mat...
        ))

        outdir = joinpath(resdir, ps, pname)
        mkpath(outdir)
        ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        outfile = joinpath(outdir, "cholsolve_fullstats_" * ts * ".jld2")
        meta = Dict(
            "julia_version" => string(VERSION),
            "timestamp"     => ts,
            "problem_set"   => ps,
            "problem_name"  => pname,
            "rho"           => rho,
            "diag_shift_used" => diag_shift_used,
            "diag_shift_schedule" => collect(DIAG_SHIFT_SCHEDULE),
            "factor_attempts" => attempts,
            "factor_time_s" => factor_time_s,
            "n" => n,
            "nnz_W" => nnz_safe(W_factored),
            "nnz_L" => nnz_safe(Lsp),
        )
        @info "Saving $(length(records)) records to $outfile"
        @save outfile meta records
    end
end

start_time = Dates.now()

run_bench_cholsolve()

@info "🎉 chol solve benchmark finished!"
elapsed_time = Dates.now() - start_time
t = Dates.Time(0) + elapsed_time
@info "Elapsed time: $(Dates.format(t, "HH:MM:SS.s"))"
