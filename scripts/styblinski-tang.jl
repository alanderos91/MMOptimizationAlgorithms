import Pkg; Pkg.activate("./scripts"); Pkg.status()

using LinearAlgebra, MMOptimizationAlgorithms, Random, Roots

const MMOA = MMOptimizationAlgorithms

import Logging
Logging.disable_logging(Logging.Warn)

# Styblinski-Tang
function f(x)
    fx = zero(eltype(x))
    for i in eachindex(x)
        fx += x[i]^4 - 16*x[i]^2 +5*x[i]
    end
    return 1//2*fx
end

function estimatef(x, grad, lambda)
    g = sqrt(norm(grad))

    if iszero(lambda)
        c = (4*g)^(-1/3)
    elseif lambda < 0
        c = fzero(y -> 4*g*y^3 + lambda*y - 1, one(lambda))
    else
        error("Expected lambda <= 0")
    end
    L = 12*c*g

    return L, c
end

rng = Xoshiro()
seed = UInt64[rng.s0, rng.s1, rng.s2, rng.s3]
# seed = UInt64[0x4a3a10a44c7ed8a0, 0x24ef4833e2022c43, 0x404647e06ef25e5e, 0x1f7ff3743bd47407]
# seed = UInt64[0xb1a2c21164cdddf5, 0xe0fbf3394abdcaf6, 0xb52d191110128010, 0x920d57adafa8f083]
# seed = UInt64[0xa40245df1e429c9e, 0x9080eb2494cde8f5, 0xb042bce0a2ad8634, 0x7ab3687e275d6a39]
seed = UInt64[0xa0a9633bb04a9ad2, 0xffdcb16aac817989, 0x2dfbae18e4a4c23d, 0xd70043446cd4bd35]
Random.seed!(rng, seed)
println("RNG Seed: $(seed)")

d = 4
x_init = -5 .+ 5*rand(rng, d)
z = -2.903534027771177*ones(d) # found using Roots.jl
fz = f(z)
options = set_options(gtol=0.0, maxiter=30)
callback = VerboseCallback()

println("\nStyblinski-Tang w/ d=4")
println("Global minimum f(z): $(fz);\tzᵢ = $(round(z[1], digits=3)) for i=1,2,…,d")
println("Initialize each xᵢ ∈ [-5, 0], x = ", round.(x_init, digits=3))

println("\nTrust-Region Newton")
result = @time trnewton(f, x_init; callback=callback, options=options, estimatef=estimatef)
println("\n  |x₁ - z₁| = $(abs(result[1] - z[1]));\tx₁ = $(result[1])")
println("  |f(x)-f(z)| = $(abs(f(result) - fz));\tf(x) = $(f(result))")

println("\nStep-Halving Newton")
result = @time newton(f, x_init; callback=callback, options=options, nhalf=8)
println("\n  |x₁ - z₁| = $(abs(result[1] - z[1]));\tx₁ = $(result[1])")
println("  |f(x)-f(z)| = $(abs(f(result) - fz));\tf(x) = $(f(result))")
