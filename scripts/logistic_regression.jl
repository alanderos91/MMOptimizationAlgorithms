import Pkg; Pkg.activate("./scripts"); Pkg.status()

using Distributions, LinearAlgebra, MMOptimizationAlgorithms, Random, SpecialFunctions, StatsBase

const MMOA = MMOptimizationAlgorithms

import Logging
Logging.disable_logging(Logging.Warn)

logistic(x) = 1 / (1 + exp(-x))

simulate_logistic_regression(n, p) = simulate_logistic_regression(Random.GLOBAL_RNG, n, p)

function simulate_logistic_regression(rng, n, p)
    # coefficients
    beta = 1/p * randn(rng, p)

    # covariates
    X = MMOA.simulate_constant_correlation(n, p, p; rng=rng, noisedim=3, delta=5e-3, epsilon=1e-1)
    F = StatsBase.fit(ZScoreTransform, X, dims=1)
    StatsBase.transform!(F, X)
    X = [X ones(n)]
    beta = [beta; randn(rng)]
    Xbeta = X*beta

    # response
    mu = logistic.(Xbeta)
    y = map(mu_i -> Float64(rand(rng, Bernoulli(mu_i))), mu)

    return y, mu, X, beta
end

function logistic_nloglikelihood(beta, y, X)
    yhat = X*beta
    mu = map(logistic, yhat)
    nlogL = -sum(y[i]*log(mu[i]) + (1-y[i])*log(1-mu[i]) for i in eachindex(mu))
    return nlogL
end

function global_estimate(evalf, beta, beta_old, iter, r, lambda_max)
    return 1//4 * s^2
end

function local_estimate(evalf, beta, beta_old, iter, r, X, xnorm, s)
    function h(x)
        -(exp(x)-1) * exp(x) / (1 + exp(x))^3
    end

    L = zero(r)
    root = log(2 + sqrt(3))
    for i in eachindex(xnorm)
        x0 = dot(beta, view(X, i, :))
        a, b = -r*xnorm[i] + x0, r*xnorm[i] + x0
        if a <= -root <= b || a <= root <= b
            c = abs( h(root) )
        else
            c = abs( min(h(a), h(b)) )
        end
        L += c * xnorm[i]^3
    end

    return L
end

#######################################

rng = Xoshiro()
# seed = UInt64[rng.s0, rng.s1, rng.s2, rng.s3]
# seed = UInt64[0x691cbaa0d236e1dc, 0x3d9446b0ed97f550, 0xd8a2cf88c3668519, 0x08e045b82b2343cf]
seed = UInt64[0xaaa8a458af45b873, 0xd87bd949bf1d4150, 0xfe09d363fe6f39f0, 0x549cad71c97635df]
Random.seed!(rng, seed)
println("RNG Seed: $(seed)")

options = set_options(gtol=1e-10, maxiter=20)
callback = VerboseCallback()

n, p = 10^3, 25
y, mu, X, beta0 = simulate_logistic_regression(rng, n, p)
spectral_norm = opnorm(X, 2)
xnorm = [norm(xi) for xi in eachrow(X)]

# estimate_L(evalf, beta, beta_old, iter, r) = global_estimate(evalf, beta, beta_old, iter, r, lambda_max)
estimate_L(evalf, beta, beta_old, iter, r) = local_estimate(evalf, beta, beta_old, iter, r, X, xnorm, spectral_norm)

println("\nLogistic Regression")
println()
println("$(n) samples, $(p) predictors")
println("Condition number: $(cond(X))")
println("Extrema for X rows: $(extrema(xnorm))")
println()
println("Mean: $(mean(y))")
println("Variance: $(var(y))")
println("Extrema for response: $(extrema(y))")

beta_init = beta0 .+ 1/p*randn(rng, p+1)
f(beta) = logistic_nloglikelihood(beta, y, X)

println("\nTrust-Region Newton")

result = @time trnewton(f, beta_init; callback=callback, options=options, lipschitzf=estimate_L)

n_predictor_signs = count(>(0), (result .* beta0)[1:p])
intercept_sign = sign(result[end]) .== sign(beta0[end])
mse = mean(abs2, result - beta0)
logl0 = -logistic_nloglikelihood(beta_init, y, X)
logl = -logistic_nloglikelihood(result, y, X)

print("""
\n\tnumber of consistent signs: $(n_predictor_signs) / $(p)
\tmatched intercept sign: $(intercept_sign)
\tmean squared error: $(mse)
\tinitial loglikelihood: $(logl0)
\tloglikelihood: $(logl)
""")

println("\nStep-Halving Newton")

result = @time newton(f, beta_init; callback=callback, options=options, nhalf=5)

n_predictor_signs = count(>(0), (result .* beta0)[1:p])
intercept_sign = sign(result[end]) .== sign(beta0[end])
mse = mean(abs2, result - beta0)
logl0 = -logistic_nloglikelihood(beta_init, y, X)
logl = -logistic_nloglikelihood(result, y, X)

print("""
\n\tnumber of consistent signs: $(n_predictor_signs) / $(p)
\tmatched intercept sign: $(intercept_sign)
\tmean squared error: $(mse)
\tinitial loglikelihood: $(logl0)
\tloglikelihood: $(logl)
""")
