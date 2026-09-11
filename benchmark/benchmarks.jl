using BorderedLinearSolve, BenchmarkTools
using StableRNGs, LinearAlgebra

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Bordered system: [J b; c' d] [v; σ] = [f; g]
for_ = (n) -> (
    rand(rng, n, n) + n * I, rand(rng, n), rand(rng, n), 1.5,
    rand(rng, n), 0.7,
)
J50, b50, c50, d50, f50, g50 = for_(50)
J500, b500, c500, d500, f500, g500 = for_(500)

prob50 = BorderedLinearProblem(J50, b50, c50, d50, f50, g50)
prob500 = BorderedLinearProblem(J500, b500, c500, d500, f500, g500)

# =============================================================================
# Solves
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

SUITE["solve"]["bordering_50"] = @benchmarkable solve($prob50, BorderingBLS())
SUITE["solve"]["direct_50"] = @benchmarkable solve($prob50, DirectBLS())
SUITE["solve"]["bordering_500"] = @benchmarkable solve($prob500, BorderingBLS())
SUITE["solve"]["direct_500"] = @benchmarkable solve($prob500, DirectBLS())

# =============================================================================
# Construction + init
# =============================================================================

SUITE["construct"] = BenchmarkGroup()

SUITE["construct"]["problem"] = @benchmarkable BorderedLinearProblem(
    $J50, $b50, $c50, $d50, $f50, $g50
)
SUITE["construct"]["init"] = @benchmarkable init($prob50, BorderingBLS())
