import Pkg; Pkg.activate("./scripts"); Pkg.status()
mkpath("./results")

using CairoMakie, LaTeXStrings
using Distributions, LinearAlgebra, MMOptimizationAlgorithms, Random, Roots, SpecialFunctions, StatsBase

const MMOA = MMOptimizationAlgorithms

import Logging
Logging.disable_logging(Logging.Warn)

import CairoMakie: Makie.get_tickvalues

struct IntegerTicks
    step::Int
end

IntegerTicks() = IntegerTicks(1)

function Makie.get_tickvalues(x::IntegerTicks, vmin, vmax)
    ceil(Int, vmin) : x.step : floor(Int, vmax)
end

function simulate_poisson_regression(rng, n, p)
    # coefficients
    beta = 1/(2p)*randn(rng, p)

    # covariates
    X = MMOA.simulate_constant_correlation(n, p, p;
        rng=rng,
        rho=0.9*ones(p),
        noisedim=10,
        delta=1e-8,
        epsilon=1e-6
    )
    F = StatsBase.fit(ZScoreTransform, X, dims=1)
    StatsBase.transform!(F, X)
    X = [X ones(n)]
    beta = [beta; abs(minimum(X)) / sqrt(p)]
    Xbeta = X*beta

    # response
    mu = exp.(Xbeta)
    y = map(mui -> rand(rng, Poisson(mui)), mu)

    return y, mu, X, beta
end

struct NegativePoissonLogLikelihood{T,f}
    inverse_link::f
    y::Vector{Int}
    X::Matrix{T}
end

function (F::NegativePoissonLogLikelihood)(beta)
    g, y, X = F.inverse_link, F.y, F.X
    yhat = X*beta
    mu = map(g, yhat)
    nlogL = zero(eltype(yhat))
    for i in eachindex(y)
        nlogL -= y[i]*log(mu[i]) - mu[i] - logfactorial(y[i])
    end
    return nlogL
end

#### Exponential inverse Link ####

struct EstimateFExp{T}
    X::Matrix{T}
    xnorm::Vector{T}
    spectral_norm::T
end

function (F::EstimateFExp)(beta, grad, lambda)
    function local_L(c, beta, g, X, xnorm, s)
        v = -Inf
        for (i, x) in enumerate(eachrow(X))
            v = max(v, xnorm[i] * exp(big( dot(x, beta) + c*g*xnorm[i] )))
        end
        return v*s^2
    end

    g = sqrt(norm(grad))
    if iszero(lambda)
        a, b = 0.0, max(g, 1/g)
        c = fzero(
            y -> local_L(y, beta, g, X, xnorm, s)*y^2 / 3 - 1,
            a, b;
            maxiters=10^5
        )
    else
        error("Expected lambda = 0")
    end
    L = local_L(c, beta, g, X, xnorm, s)

    return L, c
end

#### Cubic inverse link ####

struct CubicInverseLink{T}
    c::T
end

function (F::CubicInverseLink)(x)
    c = F.c
    if x >= c
        s = x - c
        exp(c) * (1 + s + 1//2*s^2 + 1//6*s^3)
    else
        exp(x)
    end
end

struct EstimateFCubic{T}
    c::T
    xnorm::Vector{T}
    spectral_norm::T
end

function (F::EstimateFCubic)(beta, grad, lambda)
    c, xnorm, s = F.c, F.xnorm, F.spectral_norm
    L = 1.623 * exp(c) * s^2 * maximum(xnorm)
    if iszero(lambda)
        c = sqrt(3 / L)
    else
        error("Expected lambda = 0")
        # c = (-lambda + sqrt(lambda^2 + L/3)) / (2*L/3)
    end
    return L, c
end

mse(x, y) = mean(abs2, x - y)

function deviance_residuals(invlink, y, X, beta)
    fitted = map(invlink, X*beta)
    residual = sign.(y - fitted) .* sqrt.(
        2*(y .* log.(y ./ fitted) - (y - fitted))
    )

    fig = Figure()
    ax = Axis(fig[1,1],
        xlabel="Fitted",
        ylabel="Deviance Residual",
        xticks=LinearTicks(10),
        yticks=IntegerTicks(),
    )
    scatter!(ax, fitted, residual, color=y)
    xlims!(ax, low=0.0, high=ceil(maximum(fitted)))
    return fig
end

function histogram(y)
    fig = Figure()
    ax = Axis(fig[1,1],
        xlabel="Count",
        ylabel="Frequency",
        xticks=IntegerTicks(),
    )
    hist!(ax, y; bins=maximum(y)+1, normalization=:probability)
    return fig
end

struct CustomCallback{T}
    data::Dict{Symbol,Vector{T}}
    every::Int

    function CustomCallback{T}(every::Int) where T
        data = Dict{Symbol,Vector{T}}()
        data[:objective] = T[]
        data[:gradient] = T[]
        data[:change] = T[]
        data[:lipschitz] = T[]
        new{T}(data, every)
    end
end

CustomCallback() = CustomCallback{Float64}(1) # default to Float64 eltype

function (F::CustomCallback)((iter, state), ::Nothing, ::Nothing)
    data, every = F.data, F.every
    if iter == -1
        foreach(empty!, values(data))
        iter = 0
    end
    if iter % every == 0
        push!(data[:objective], state.objective)
        push!(data[:gradient], state.gradient)
        push!(data[:change], state.residual)
        push!(data[:lipschitz], state.lipschitz_L)
    end
    return nothing
end

#######################################

rng = Xoshiro()
# seed = UInt64[rng.s0, rng.s1, rng.s2, rng.s3]
# seed = UInt64[0x2a1e93cbc05af656, 0x3db124ebe7807e31, 0x38d35225258cfb63, 0x2658565170240da3]
seed = UInt64[0x1fb70007464cadd9, 0xd0695427620556eb, 0x35c524fdabae0163, 0xa3c013663e2faf39]
Random.seed!(rng, seed)
println("RNG Seed: $(seed)")

nhalf = 10
options = set_options(gtol=1e-10, maxiter=100)
# callback = VerboseCallback()
outdir = "./results"
example = "Poisson Regression"
ext = "pdf"

n, p = 10^3, 10
y, mu, X, beta0 = simulate_poisson_regression(rng, n, p)
xnorm = [norm(xi) for xi in eachrow(X)]
s = opnorm(X, 2)
beta_init = beta0 .+ 1e0*rand(rng, p+1)
# beta_init = zeros(p); beta_init[end] = beta0[end]

println("\nPoisson Regression")
println()
println("$(n) samples, $(p) predictors")
println("Condition number of X: $(cond(X))")
println("Extrema for norms of rows: $(extrema(xnorm))")
println("Spectral norm: $(s)")
println()
println("Mean of response: $(mean(y))")
println("Variance of response: $(var(y))")
println("Extrema for response: $(extrema(y))")

sfig = histogram(y)
save(
    joinpath(outdir, "$(example)-histogram.$(ext)"), sfig, pt_per_unit=1,
)

patch_w = 20 # width of patch inside legend
scalef = 0.9 # scale factor for resolution
fig = Figure(resolution=(scalef*1000, scalef*600))
fgrid = fig[1,1] = GridLayout()
ax = Axis(fgrid[1:2,1],
    xlabel=latexstring("Iteration", " ", L"n"),
    ylabel=latexstring("", L"\log_{10}[\mathcal{L}(\beta_{n}) - \mathcal{L}(\hat{\beta})]"),
    xticks=IntegerTicks(5),
    yticks=LogTicks(IntegerTicks(2)),
    yscale=log10,
)

println("\n[Exponential inverse link]")

f_exp = NegativePoissonLogLikelihood(exp, y, X)
estimatef_exp = EstimateFExp(X, xnorm, s)

tmp = newton(f_exp, beta_init; options=set_options(gtol=1e-16, maxiter=1000), nhalf=nhalf)
logL = f_exp(tmp)

println("initial loglikelihood: $(-f_exp(beta_init))")
println("final loglikelihood: $(-logL)")

callback = CustomCallback()
t1 = @time trnewton(f_exp, beta_init; callback=callback, options=options, estimatef=estimatef_exp)
scatterlines!(ax, abs.(callback.data[:objective] .- logL) .+ 1e-16;
    color=:red,
    linestyle=:solid,
    marker=:circle,
)

callback = CustomCallback()
n1 = @time newton(f_exp, beta_init; callback=callback, options=options, nhalf=nhalf)
scatterlines!(ax, abs.(callback.data[:objective] .- logL) .+ 1e-16;
    color=:red,
    linestyle=:dash,
    marker=:circle,
)

println()
println("  TR Newton loglikelihood: $(-f_exp(t1))")
println("  SH Newton loglikelihood: $(-f_exp(n1))")
println("  MSE(TR, GT) = $(mse(t1, beta0))")
println("  MSE(SH, GT) = $(mse(n1, beta0))")
println("  TR signs: $(count(>=(0), t1 .* beta0)) / $(p+1)")
println("  SH signs: $(count(>=(0), n1 .* beta0)) / $(p+1)")

sfig = deviance_residuals(exp, y, X, t1)
save(
    joinpath(outdir, "$(example)-loglink.$(ext)"), sfig, pt_per_unit=1,
)

cs = [:blue, :orange, :purple]
ms = [:rect, :utriangle, :star5]
for (k, c) in enumerate((1.0, 2.0, 3.0))
    cubic_invlink = CubicInverseLink(c)
    f_cubic = NegativePoissonLogLikelihood(cubic_invlink, y, X)
    estimatef_cubic = EstimateFCubic(c, xnorm, s)

    println("\n[Cubic link / c = $(c)]")

    callback = CustomCallback()
    t3 = @time trnewton(f_cubic, beta_init; callback=callback, options=options, estimatef=estimatef_cubic)
    scatterlines!(ax, abs.(callback.data[:objective] .- logL) .+ 1e-16;
        color=cs[k],
        linestyle=:solid,
        marker=ms[k],
    )

    # callback = CustomCallback()
    # n3 = @time newton(f_cubic, beta_init; callback=callback, options=options, nhalf=nhalf)
    # scatterlines!(ax, abs.(callback.data[:objective] .- logL) .+ 1e-16;
    #     color=cs[k],
    #     linestyle=:dash,
    #     marker=ms[k],
    # )

    # println()
    # println("  TR Newton loglikelihood: $(-f_exp(t3))")
    # println("  SH Newton loglikelihood: $(-f_exp(n3))")
    # println("  MSE(TR, GT) = $(mse(t3, beta0))")
    # println("  MSE(SH, GT) = $(mse(n3, beta0))")
    # println("  TR signs: $(count(>=(0), t3 .* beta0)) / $(p+1)")
    # println("  SH signs: $(count(>=(0), n3 .* beta0)) / $(p+1)")

    sfig = deviance_residuals(cubic_invlink, y, X, t3)
    save(
        joinpath(outdir, "$(example)-cubic-p=$(c).$(ext)"), sfig, pt_per_unit=1,
    )
end

elementsA = [
    [
        LineElement(color=:red, linestyle=:solid),
        MarkerElement(color=:red, marker=:circle),
    ],
    [
        LineElement(color=cs[1], linestyle=:solid),
        MarkerElement(color=cs[1], marker=ms[1]),
    ],
    [
        LineElement(color=cs[2], linestyle=:solid),
        MarkerElement(color=cs[2], marker=ms[2]),
    ],
    [
        LineElement(color=cs[3], linestyle=:solid),
        MarkerElement(color=cs[3], marker=ms[3]),
    ],
]
elementsB = [
    [
        LineElement(color=:red, linestyle=:dash),
        MarkerElement(color=:red, marker=:circle),
    ],
]

labelsA = [
    "Exponential",
    "Cubic (p = 1.0)",
    "Cubic (p = 2.0)",
    "Cubic (p = 3.0)",
]

labelsB = [
    "Exponential"
]

Legend(fgrid[1,2], elementsA, labelsA, "Trust Region";
    framevisible=false,
    orientation=:vertical,
    patchsize = (2*patch_w, patch_w),
)

Legend(fgrid[2,2], elementsB, labelsB, "Newton with Step-Halving";
    framevisible=false,
    orientation=:vertical,
    patchsize = (2*patch_w, patch_w),
)

# colgap!(fgrid, 1, 0)
# colgap!(fgrid, 2, 0)
resize_to_layout!(fig)

save(
    joinpath(outdir, "$(example)-objective.$(ext)"), fig, pt_per_unit=1,
)
