"""
    DirectBLS(; inner = nothing)

Assemble the full `(n+m) x (n+m)` bordered matrix and hand it to `LinearSolve`.

This gives up the cache reuse that [`BorderingBLS`](@ref) exists for, and for a sparse
`J` the assembled matrix has a dense row and column that the factorization has to carry.
It is here because it is the reference the bordering result should be checked against,
and because for a small `J` it is simply faster than two inner solves.
"""
Base.@kwdef struct DirectBLS{S}
    inner::S = nothing
end


struct DirectCache{A, P}
    alg::A
    prob::P
end

SciMLBase.init(prob::BorderedLinearProblem, alg::DirectBLS; kwargs...) =
    DirectCache(alg, prob)

function _assemble(prob::BorderedLinearProblem)
    J = prob.J
    b = reshape(prob.b, size(prob.b, 1), _border_width(prob.b))
    c = reshape(prob.c, size(prob.c, 1), _border_width(prob.c))
    d = prob.d isa Number ? fill(prob.d, 1, 1) : prob.d
    return [J b; c' d]
end

function SciMLBase.solve!(cache::DirectCache)
    prob = cache.prob
    M = _assemble(prob)
    rhs = vcat(prob.f, prob.g isa Number ? [prob.g] : prob.g)
    sol = solve!(init(LinearProblem(M, rhs), cache.alg.inner))
    n = length(prob.f)
    v = sol.u[1:n]
    tail = sol.u[(n + 1):end]
    σ = prob.g isa Number ? tail[1] : tail
    return BorderedLinearSolution(v, σ, sol.retcode)
end
