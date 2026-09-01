"""
    BorderingBLS(; inner = nothing, k = 0, tol = 1e-12)

Block elimination. Solves `J x₁ = f` and `J δx = b`, then

    σ  = (g - c'x₁) / (d - c'δx)
    v  = x₁ - σ δx

Both inner solves use the same `J`, so they share one `LinearSolve` cache; `inner` is
the `LinearSolve` algorithm for them (`nothing` picks the default).

`k` is the number of correction sweeps in the sense of Govaerts' BEC+k: after the first
pass the true residual of the bordered system is formed and used to solve for a
correction, which recovers accuracy when `J` is close to singular. That happens exactly
where continuation is interesting, near a fold, so it is worth having, but each sweep
costs two more inner solves and the default of `0` skips it. A sweep is skipped early
once the residual is below `tol`.

# Reference

Govaerts, W. "Stable Solvers and Block Elimination for Bordered Systems." SIAM Journal
on Matrix Analysis and Applications 12, no. 3 (1991): 469-83.
"""
Base.@kwdef struct BorderingBLS{S, T <: Real}
    inner::S = nothing
    k::Int = 0
    tol::T = 1.0e-12
end

BorderingBLS(inner) = BorderingBLS(; inner)

"""
    BorderedCache

Holds the `LinearSolve` cache for `J` across solves. Reusing it is the whole reason to
prefer this over assembling the bordered matrix: the factorization or Krylov workspace
built for `J` is kept, and `J` can be updated in place between outer steps.
"""
mutable struct BorderedCache{A, C, P}
    alg::A
    inner::C
    prob::P
end

"""
    init(prob::BorderedLinearProblem, alg)

Build the reusable cache. The `LinearSolve` cache for `J` is created once here, not per
solve, so a factorization survives repeated `solve!` calls and successive right-hand
sides.
"""
function SciMLBase.init(prob::BorderedLinearProblem, alg::BorderingBLS; kwargs...)
    inner = init(LinearProblem(prob.J, _first_rhs(prob.f)), alg.inner; kwargs...)
    return BorderedCache(alg, inner, prob)
end

_first_rhs(f::AbstractVector) = copy(f)

# One block-elimination pass. `f` and `g` are passed explicitly so the correction
# sweeps can reuse this with the residual in place of the original right-hand side.
#
#   J x₁ = f,  J Δ = b        (same J, so one cache serves every solve)
#   σ = (d - c'Δ) \ (g - c'x₁)
#   v = x₁ - Δ σ
#
# For a width-1 border the last two lines are a scalar division and an axpy.
function _bec(cache::BorderedCache, f, g)
    prob = cache.prob
    inner = cache.inner

    # The right-hand side is copied in. Some backends solve in place and overwrite
    # `b`; on a GPU this was observed to destroy the caller's array, which both
    # corrupts `prob.b` for the correction sweeps and silently mutates the problem the
    # user handed us.
    inner.b = _rhs_copy(f)
    sol1 = solve!(inner)
    x1 = copy(sol1.u)
    ok = SciMLBase.successful_retcode(sol1.retcode)

    # Same `J`, further right-hand sides: these are the solves that reuse the cache.
    # The border is solved a column at a time rather than as a matrix right-hand side,
    # which keeps one vector-shaped cache serving every solve.
    Δ, ok2 = _solve_border(inner, prob.b)
    ok &= ok2

    σ = _border_solve(prob.d, prob.c, x1, Δ, g)
    v = x1 - _apply_border(Δ, σ)

    return v, σ, ok ? ReturnCode.Success : ReturnCode.Failure
end

function _solve_border(inner, b::AbstractVector)
    inner.b = _rhs_copy(b)
    sol = solve!(inner)
    return copy(sol.u), SciMLBase.successful_retcode(sol.retcode)
end

function _solve_border(inner, b::AbstractMatrix)
    m = size(b, 2)
    cols = map(1:m) do j
        inner.b = _rhs_copy(@view b[:, j])
        sol = solve!(inner)
        (copy(sol.u), SciMLBase.successful_retcode(sol.retcode))
    end
    return reduce(hcat, first.(cols)), all(last.(cols))
end

# `collect` rather than `copy` so a view of a border column becomes a dense vector of
# the same array type, which is what the inner cache expects.
_rhs_copy(b::AbstractVector) = collect(b)

# `c'x` without indexing either argument, so a GPU array never hits a scalar read.
_dotc(c::AbstractVector, x::AbstractVector) = dot(c, x)
_dotc(c::AbstractMatrix, x::AbstractVector) = c' * x
_dotc(c::AbstractMatrix, x::AbstractMatrix) = c' * x
_dotc(c::AbstractVector, x::AbstractMatrix) = vec(c' * x)

# Width-1: a scalar division, which stays on whatever device the data is on.
_border_solve(d::Number, c, x1, Δ::AbstractVector, g) =
    (g - _dotc(c, x1)) / (d - _dotc(c, Δ))

# Width m: a dense m x m solve. `m` is small by assumption, and doing it on the host
# keeps it off a GPU `\`, whose pivoting would scalar-index for these sizes.
function _border_solve(d, c, x1, Δ::AbstractMatrix, g)
    S = Array(d) - Array(_dotc(c, Δ))
    rhs = Array(g) - Array(_dotc(c, x1))
    return S \ rhs
end

_apply_border(Δ::AbstractVector, σ::Number) = Δ * σ
_apply_border(Δ::AbstractMatrix, σ::AbstractVector) = Δ * oftype(similar(Δ, size(Δ, 2)), σ)

function SciMLBase.solve!(cache::BorderedCache{<:BorderingBLS})
    alg = cache.alg
    prob = cache.prob

    v, σ, rc = _bec(cache, prob.f, prob.g)

    # BEC+k. The residual is formed against the original system, so a correction can
    # only help; when it is already at tolerance the sweep is skipped.
    for _ in 1:(alg.k)
        rf = prob.f - _apply(prob.J, v) - _apply_border(prob.b, σ)
        rg = prob.g - _dotc(prob.c, v) - prob.d * σ
        if max(norm(rf), norm(rg)) <= alg.tol
            break
        end
        dv, dσ, rc2 = _bec(cache, rf, rg)
        v = v + dv
        σ = σ + dσ
        SciMLBase.successful_retcode(rc2) || (rc = ReturnCode.Failure)
    end

    return BorderedLinearSolution(v, σ, rc)
end

_apply(J, v) = J * v
