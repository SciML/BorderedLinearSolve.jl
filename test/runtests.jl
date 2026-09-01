using BorderedLinearSolve
using LinearAlgebra, SparseArrays, Random, Test
using LinearSolve: LUFactorization, KrylovJL_GMRES
using SciMLBase: successful_retcode

# Reference: assemble the bordered matrix and solve it densely. Everything below is
# checked against this rather than against another code path in the package.
function reference(J, b, c, d, f, g)
    M = [Matrix(J) reshape(b, :, 1); reshape(c, 1, :) fill(d, 1, 1)]
    x = M \ vcat(f, g)
    return x[1:length(f)], x[end]
end

@testset "BorderedLinearSolve" begin
    Random.seed!(560)

    @testset "bordering matches the assembled system" begin
        for n in (5, 50)
            J = rand(n, n) + n * I
            b = rand(n); c = rand(n); d = 1.5
            f = rand(n); g = 0.7
            vref, σref = reference(J, b, c, d, f, g)

            sol = solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS())
            @test successful_retcode(sol)
            @test sol.v ≈ vref rtol = 1.0e-9
            @test sol.σ ≈ σref rtol = 1.0e-9
        end
    end

    @testset "direct assembly matches too" begin
        n = 20
        J = rand(n, n) + n * I
        b = rand(n); c = rand(n); d = 2.0
        f = rand(n); g = -0.3
        vref, σref = reference(J, b, c, d, f, g)

        sol = solve(BorderedLinearProblem(J, b, c, d, f, g), DirectBLS())
        @test sol.v ≈ vref rtol = 1.0e-9
        @test sol.σ ≈ σref rtol = 1.0e-9
    end

    @testset "the inner cache is reused across the two solves" begin
        # This is the reason the package exists, so it is asserted rather than assumed:
        # after the first inner solve the cache is no longer fresh, meaning the second
        # solve reuses the factorization instead of rebuilding it.
        n = 30
        J = rand(n, n) + n * I
        prob = BorderedLinearProblem(J, rand(n), rand(n), 1.0, rand(n), 0.5)
        cache = init(prob, BorderingBLS(LUFactorization()))
        @test cache.inner.isfresh
        solve!(cache)
        @test !cache.inner.isfresh
    end

    @testset "a sparse J works and keeps J sparse" begin
        n = 60
        J = spdiagm(-1 => -ones(n - 1), 0 => 4ones(n), 1 => -ones(n - 1))
        b = rand(n); c = rand(n); d = 1.0
        f = rand(n); g = 0.25
        vref, σref = reference(J, b, c, d, f, g)

        sol = solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS())
        @test sol.v ≈ vref rtol = 1.0e-8
        @test sol.σ ≈ σref rtol = 1.0e-8
    end

    @testset "an iterative inner solver works" begin
        n = 40
        J = rand(n, n) + n * I
        b = rand(n); c = rand(n); d = 1.0
        f = rand(n); g = 0.1
        vref, σref = reference(J, b, c, d, f, g)

        sol = solve(
            BorderedLinearProblem(J, b, c, d, f, g),
            BorderingBLS(KrylovJL_GMRES())
        )
        @test sol.v ≈ vref rtol = 1.0e-6
        @test sol.σ ≈ σref rtol = 1.0e-6
    end

    @testset "BEC+k corrects a nearly singular J" begin
        # Block elimination loses accuracy as J approaches singular, which is exactly
        # where continuation spends its time. The correction sweeps should recover it.
        n = 30
        U, _ = qr(rand(n, n))
        svals = vcat(1.0e-11, range(1.0, 2.0; length = n - 1))
        J = Matrix(U) * Diagonal(svals) * Matrix(U)'
        b = rand(n); c = rand(n); d = 1.0
        f = rand(n); g = 0.4
        vref, σref = reference(J, b, c, d, f, g)

        plain = solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS(k = 0))
        corrected = solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS(k = 3))

        err(s) = norm(s.v - vref) / norm(vref) + abs(s.σ - σref) / abs(σref)
        println("  nearly singular: k=0 err=", err(plain), "  k=3 err=", err(corrected))
        @test err(corrected) <= err(plain)
    end

    @testset "a wide border" begin
        n, m = 8, 2
        J = rand(n, n) + n * I
        b = rand(n, m); c = rand(n, m); d = rand(m, m)
        f = rand(n); g = rand(m)
        M = [J b; c' d]
        xref = M \ vcat(f, g)
        prob = BorderedLinearProblem(J, b, c, d, f, g)

        # DirectBLS assembles the whole thing, so a wide border is no different.
        sol = solve(prob, DirectBLS())
        @test sol.v ≈ xref[1:n] rtol = 1.0e-9
        @test sol.σ ≈ xref[(n + 1):end] rtol = 1.0e-9

        # Bordering handles it too: m inner solves for the border block, then a
        # small m x m solve. Checked against the same assembled reference.
        solb = solve(prob, BorderingBLS())
        @test successful_retcode(solb)
        @test solb.v ≈ xref[1:n] rtol = 1.0e-9
        @test solb.σ ≈ xref[(n + 1):end] rtol = 1.0e-9

        # And the correction sweeps work for a wide border as well.
        solk = solve(prob, BorderingBLS(k = 2))
        @test solk.v ≈ xref[1:n] rtol = 1.0e-9
        @test solk.σ ≈ xref[(n + 1):end] rtol = 1.0e-9
    end

    @testset "every inner algorithm gives the right answer" begin
        # The GPU run found LUFactorization returning a wrong answer with a Success
        # retcode, which this suite could not have caught: the only test using an
        # explicit inner algorithm checked cache state, never correctness.
        n = 40
        J = rand(n, n) + n * I
        b = rand(n); c = rand(n); d = 1.7; f = rand(n); g = 0.35
        vref, σref = reference(J, b, c, d, f, g)

        for (name, inner) in (
                ("default", nothing),
                ("LUFactorization", LUFactorization()),
                ("KrylovJL_GMRES", KrylovJL_GMRES()),
            )
            for k in (0, 2)
                sol = solve(
                    BorderedLinearProblem(J, b, c, d, f, g),
                    BorderingBLS(inner = inner, k = k)
                )
                @test successful_retcode(sol)
                @test sol.v ≈ vref rtol = 1.0e-6
                @test sol.σ ≈ σref rtol = 1.0e-6
            end
        end
    end

    @testset "the caller's arrays are not mutated" begin
        # A solve that overwrites `b` in place would destroy the problem the user
        # handed us, and corrupt the BEC+k residual, which re-reads it.
        n = 30
        J = rand(n, n) + n * I
        b = rand(n); c = rand(n); d = 1.0; f = rand(n); g = 0.5
        J0, b0, c0, f0 = copy(J), copy(b), copy(c), copy(f)

        for inner in (nothing, LUFactorization(), KrylovJL_GMRES())
            solve(BorderedLinearProblem(J, b, c, d, f, g), BorderingBLS(inner = inner, k = 2))
            @test b == b0
            @test f == f0
            @test c == c0
            @test J == J0
        end
    end

    @testset "shape errors are caught" begin
        n = 5
        J = rand(n, n)
        @test_throws DimensionMismatch BorderedLinearProblem(
            rand(n, n + 1), rand(n), rand(n), 1.0, rand(n), 1.0
        )
        @test_throws DimensionMismatch BorderedLinearProblem(
            J, rand(n + 1), rand(n), 1.0, rand(n), 1.0
        )
        @test_throws DimensionMismatch BorderedLinearProblem(
            J, rand(n), rand(n), 1.0, rand(n + 1), 1.0
        )
    end
end
