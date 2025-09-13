const DEFAULT_SOLVER_ARGS = Dict{String,Any}(
    # reference solver
    "ref-solver"            => :SCS,
    
    # method‐defining defaults
    "variant"               => :PDHG,
    "res-norm"              => Inf,
    "rho"                   => 1.0,
    "theta"                 => 1.0,
    
    "acceleration"          => :none,
    "accel-memory"          => 10,
    "safeguard-norm"        => :euclid, # in {:euclid, :char, :none}
    
    "krylov-tries-per-mem"  => 1,
    "krylov-operator"       => :tilde_A,
    
    "anderson-interval"     => 10,
    "anderson-broyden-type" => :normal2, # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :rolling, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}
    
    # non‐defining defaults
    "max-iter"              => Inf,
    "rel-kkt-tol"           => 1e-6,
    "print-mod"             => 10_000,
    "print-res-rel"         => true,
    "show-vlines"           => false,
    "run-fast"              => true,
	"global-timeout"        => 120.0, # include set-up time (seconds)
	"loop-timeout"          => Inf,   # exclude set-up time (seconds)
    
	# not in use
    "restart-period"        => Inf,
    "linesearch-period"     => Inf,
    "linesearch-eps"        => 1e-3,
)