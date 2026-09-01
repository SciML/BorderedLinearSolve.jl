"""
    BorderedLinearProblem(J, b, c, d, f, g)

The bordered system

```
┌      ┐┌   ┐   ┌   ┐
│ J  b ││ v │ = │ f │
│ c' d ││ σ │   │ g │
└      ┘└   ┘   └   ┘
```

`J` is `n x n`, `b` and `c` are `n`-vectors (or `n x m` matrices for a wider border),
`d` is a scalar (or `m x m`), `f` is an `n`-vector and `g` a scalar (or `m`-vector).

Note that `c` is given as it appears above, i.e. the border row is `c'`, so `c` has the
same shape as `b`.
"""
struct BorderedLinearProblem{TJ, Tb, Tc, Td, Tf, Tg}
    J::TJ
    b::Tb
    c::Tc
    d::Td
    f::Tf
    g::Tg

    function BorderedLinearProblem(J, b, c, d, f, g)
        n = size(J, 1)
        size(J, 2) == n || throw(DimensionMismatch("J must be square, got $(size(J))"))
        size(b, 1) == n ||
            throw(DimensionMismatch("b has leading size $(size(b, 1)), expected $n"))
        size(c, 1) == n ||
            throw(DimensionMismatch("c has leading size $(size(c, 1)), expected $n"))
        length(f) == n ||
            throw(DimensionMismatch("f has length $(length(f)), expected $n"))
        _border_width(b) == _border_width(c) || throw(
            DimensionMismatch(
                "b and c disagree on border width: $(_border_width(b)) and $(_border_width(c))"
            )
        )
        return new{
            typeof(J), typeof(b), typeof(c), typeof(d), typeof(f), typeof(g),
        }(J, b, c, d, f, g)
    end
end

_border_width(x::AbstractVector) = 1
_border_width(x::AbstractMatrix) = size(x, 2)

"""
    BorderedLinearSolution(v, σ, retcode)

`v` solves the `n`-block and `σ` the border block. `retcode` is the `ReturnCode` of the
underlying `LinearSolve` solve, so a failure in the inner solver is visible here rather
than being silently absorbed.
"""
struct BorderedLinearSolution{Tv, Tσ}
    v::Tv
    σ::Tσ
    retcode::SciMLBase.ReturnCode.T
end

SciMLBase.successful_retcode(sol::BorderedLinearSolution) =
    SciMLBase.successful_retcode(sol.retcode)
