const DEFAULT_SOLVER_ARGS = Dict{String,Any}(
    # reference solver
    "ref-solver"            => :SCS,
    # method‐defining defaults
    "variant"               => :ADMM,
    "res-norm"              => Inf,
    "acceleration"          => :none,
    "rho"                   => 1.0,
    "theta"                 => 1.0,
    "accel-memory"          => 10,
    "krylov-operator"       => :tilde_A,
    "anderson-period"       => 2,
    "anderson-broyden-type" => :normal2, # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :rolling, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}
    # non‐defining defaults
    "max-iter"              => Inf,
    "rel-kkt-tol"           => 1e-6,
    "print-mod"             => 5000,
    "print-res-rel"         => true,
    "show-vlines"           => false,
    "run-fast"              => true,
	"global-timeout"        => 60.0, # include set-up time (seconds)
	"loop-timeout"          => Inf,  # exclude set-up time (seconds)
    
	# not in use
    "restart-period"        => Inf,
    "linesearch-period"     => Inf,
    "linesearch-eps"        => 1e-3,
)