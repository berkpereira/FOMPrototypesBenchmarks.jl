const DEFAULT_SOLVER_ARGS = Dict{String,Any}(
    # reference solver
    "ref-solver"            => :SCS,
    
    # method‐defining defaults
    "variant"               => :ADMM,
    "res-norm"              => Inf,
    "rho"                   => 1.0,
    "theta"                 => 1.0,
    
    "acceleration"          => :none,
    "accel-memory"          => 10,
    "safeguard-norm"        => :char, # in {:euclid, :char, :none}
    "safeguard-factor"      => 1.0, # factor for fixed-point residual safeguard check in accelerated methods
    
    "krylov-tries-per-mem"  => 1,
    "krylov-operator"       => :B,
    
    "anderson-interval"     => 10,
    "anderson-broyden-type" => :normal2, # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :rolling, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}
    
    # non‐defining defaults
    
    #######
    # MAKE SURE max-iter AND max-k-operator EQUAL
    #######
    
    "max-iter"              => 50_000, # only has an effect for no acceleration!
    "max-k-operator"        => 50_000, # only has an effect for accelerated (Krylov/Anderson)!
    "rel-kkt-tol"           => 1e-9,
    "print-mod"             => 10_000,
    "print-res-rel"         => true,
    "show-vlines"           => false,
    "run-fast"              => true,
	"global-timeout"        => Inf,   # include set-up time (seconds)
	"loop-timeout"          => Inf,   # exclude set-up time (seconds)
    
	# not in use
    "restart-period"        => Inf,
    "linesearch-period"     => Inf,
    "linesearch-eps"        => 1e-3,
)