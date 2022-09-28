using MMOptimizationAlgorithms
using LinearAlgebra, Random
using Test

MM = MMOptimizationAlgorithms

@testset "Projections" begin

    @testset "L0Projection" begin
        rng = Xoshiro(1234)
        # randomized
        for _ in 1:5
            x = 10 * randn(rng, 1000)
            k = 50
            P = MM.L0Projection(length(x), rng)
            x_proj = P(x, k)

            perm = sortperm(x, lt=isless, by=abs, rev=true, order=Base.Order.Forward)
            @test sort!(P.idx[1:k]) == sort!(perm[1:k])
        end
        
        # ties
        x = Float64[2, 1, 1, 1, 1, -0.6, 0.5, 0.5]
        shuffle!(rng, x)
        P = MM.L0Projection(length(x))
        x_sorted = sort(x, lt=isless, by=abs, rev=true, order=Base.Order.Forward)
        x_zero = P(copy(x), 0)
        @test x_zero == zeros(length(x))
        for k in 1:length(x)
            x_proj = P(copy(x), k)
            idx = findall(!isequal(0), x_proj)
            topk = sort!(x_proj[idx], lt=isless, by=abs, rev=true)
            @test topk == x_sorted[1:k]
        end
    end

    @testset "SimplexProjection" begin
        rng = Xoshiro(1234)
        (n, r) = (100, 10.0);
        x = randn(rng, n)
        P = MM.SimplexProjection{Float64}(n)
        y = P(copy(x), r)
        @test all(>=(0), y)
        @test norm(y) <= r
    end

    @testset "L1BallProjection" begin
        rng = Xoshiro(1234)
        (n, r) = (5, 4.0);
        x = randn(rng, n);
        P = MM.L1BallProjection{Float64}(n)
        y = P(copy(x), r)
        @test norm(y) <= r
        @test all(i -> sign(x[i]) == sign(y[i]), 1:n)
    end
end
