module FOMPrototypesBenchmarks

using FOMPrototypes, JLD2, Dates, Random, Infiltrator

export run_single

"""
    run_single(variant, problem_set, problem_name, rep; logdir="results")

Run one benchmark for `variant` on (`problem_set`, `problem_name`), replicate `rep`,
save into JLD2 under `logdir` (skipping if already exists).
"""
function run_single(variant::Symbol, problem_set::String, problem_name::String,
                    rep::Int; logdir::String="results")
    mkpath(logdir)
    fname = joinpath(logdir, "$(variant)_$(problem_set)_$(problem_name)_rep$(rep).jld2")
    if isfile(fname)
        println("[skip] already exists: $fname")
        return
    end

    # --- Prepare FOMPrototypes arguments ---
    args = Dict(
        "ref-solver" => :SCS,
        "variant"    => variant,
        "problem-set"=> problem_set,
        "problem-name"=> problem_name,
        "max-iter"   => 1000,
        "print-mod"  => 0,
        "run-fast"   => true,
        "res-norm"   => Inf,
        "rho"        => 1.0,
        "theta"      => 1.0,
        "acceleration" => :none,
        "accel-memory" => 20,
        "krylov-operator" => :tilde_A,
        "anderson-period" => 10,
        "anderson-broyden-type" => :normal2,
        "anderson-mem-type" => :rolling,
        "anderson-reg" => :none,
        "restart-period" => Inf,
        "linesearch-period" => Inf,
        "linesearch-eps"    => 0.001
    )

    # Fetch problem and solve reference
    println(args)
    problem = FOMPrototypes.fetch_data(args)
    x_ref, s_ref, y_ref, obj_ref = FOMPrototypes.solve_reference(problem, args)

    # Reproducible RNG and timing
    Random.seed!(rep)
    t = @elapsed ws, results = FOMPrototypes.run_prototype(problem, args; x_ref=x_ref, y_ref=y_ref)

    # Save
    @info "Writing results to $fname"
    timestamp = Dates.now()
    @save fname variant problem_set problem_name rep t obj_ref ws results timestamp
end

end # module FOMPrototypesBenchmarks
