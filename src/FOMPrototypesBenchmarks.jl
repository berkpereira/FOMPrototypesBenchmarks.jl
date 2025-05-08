module FOMPrototypesBenchmarks

using FOMPrototypes, JLD2, Dates, Random

export run_multiple, run_warmup

"""
    run_multiple(
      variant::Symbol,
      max_iter::Int,
      problem_set::String,
      problem_name::String;
      reps::Vector{Int},
      resdir::String = "results",
      save_results::Bool = true,
      args = nothing
    )

Runs *all* replicates in `reps`.  If `save_results`, only the `reps` whose
`resdir/problem_set/problem_name/variant/repN.jld2` files do *not* exist
will actually be solved; if none are missing, returns immediately.
Problem data and reference solve are done only once.
"""
function run_multiple(variant::Symbol,
                      max_iter::Int,
                      problem_set::String,
                      problem_name::String;
                      reps::Vector{Int},
                      resdir::String="results",
                      save_results::Bool=true,
                      args=nothing)

    # prepare output directory & filter reps
    outdir = joinpath(resdir, problem_set, problem_name, string(variant))
    missing = Int[]
    if save_results
        for rep in reps
            fname = joinpath(outdir, "rep$(rep).jld2")
            if !isfile(fname)
                push!(missing, rep)
            end
        end
        if isempty(missing)
            @info "[skip] all reps done for $variant / $problem_set / $problem_name"
            return
        end
        reps = missing
    end

    # --- Fetch data & reference only once ---
    if args === nothing
        # can customise the defaults inside here
        args = Dict(
            "ref-solver"  => :SCS,
            "variant"     => variant,
            "problem-set" => problem_set,
            "problem-name"=> problem_name,
            
            "res-norm"     => Inf,
            "max-iter"     => max_iter,
            "rel-kkt-tol"  => 1e-10,
            
            "acceleration"    => :none,
            "accel-memory"    => 200,
            "krylov-operator" => :tilde_A,
            
            "anderson-broyden-type" => Symbol(1), # in {Symbol(1), :normal2, :QR2}
            "anderson-mem-type"     => :rolling, # in {:rolling, :restarted}
            "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}
            "anderson-period"       => 2,

            "rho"   => 1.0,
            "theta" => 1.0,
            
            "restart-period"    => Inf,
            "linesearch-period" => Inf,
            "linesearch-eps"    => 0.001,

            "print-mod"          => 50,
            "residuals-relative" => true,
            "show-vlines"        => true,
            "run-fast"           => true,
        )
    end

    # load + solve reference
    problem = FOMPrototypes.fetch_data(args)
    x_ref, s_ref, y_ref, obj_ref = FOMPrototypes.solve_reference(problem, args)

    # run each missing replicate
    for rep in reps
        @info "→ Running rep $rep for $variant / $problem_set / $problem_name"
        Random.seed!(rep)
        t = @elapsed ws, results = FOMPrototypes.run_prototype(
            problem, args; x_ref=x_ref, y_ref=y_ref)

        if save_results
            mkpath(outdir)
            fname = joinpath(outdir, "rep$(rep).jld2")
            timestamp = Dates.now()
            @info "Writing $fname"
            @save fname variant problem_set problem_name rep t obj_ref ws results timestamp
        end
    end
end

"""
    run_warmup(variant)

Do one tiny solve with the given `variant` to trigger all compilation paths.
"""
function run_warmup(variant)
    # construct a trivial 1×1 quadratic problem
    # e.g. problem_set = "toy", problem_name = "one_by_one"
    # Here we assume your module knows how to make a dummy problem.
    warmup_problem_set, warmup_problem_name = "sslsq", "NYPA_Maragal_1_lasso"
    run_multiple(variant, 10, warmup_problem_set, warmup_problem_name; reps=[1], save_results=false)
end

end # module FOMPrototypesBenchmarks