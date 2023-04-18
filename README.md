# MM Optimization Algorithms: Fast Updates, Trust Regions, and Path Following 

---
<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg)
[![Build Status](https://travis-ci.com/alanderos91/Code.jl.svg?branch=master)](https://travis-ci.com/alanderos91/Code.jl)
[![codecov.io](http://codecov.io/github/alanderos91/Code.jl/coverage.svg?branch=master)](http://codecov.io/github/alanderos91/Code.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://alanderos91.github.io/Code.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://alanderos91.github.io/Code.jl/dev)
-->

This package implements various algorithms based on the MM principle and distance majorization.
The algorithms implemented are:

- `MML`: Majorize an objective with a quadratic surrogate and minimize it (linear solve).
- `MMAL`: Majorize an objective with a quadratic surrogate and minimize it (linear solve). In the structure of the Hessian is exploited to accelerate updates.
- `MMPS`: Majorize an objective with a surrogate that separates parameters.
- `MMBS`: Majorize an objective with a surrogate that separates blocks of parameters.
- `SD`: Majorize an objective with a surrogate that admits an exact line search in steepest descent.

In addition, the functions `newton` and `trnewton` implement Newton's method with step-halving and Newton's method with adaptive trust regions. The latter algorithm is adaptive in the sense that it simultaneously selects a trust region radius and local Lipschitz constant (over the trust region) to update a solution estimate. Derivatives are handled with forward-mode automatic differentiation (ForwardDiff.jl) but a user must supply a function to calculate constants `(L,c)` related to a (local) Lipschitz constant and trust region radius.

**Note**: Some algorithms may occasionally encounter small roundoff errors that result in loss of precision when evaluation objective functions. This translates to small violations of the descent property guaranteed by MM. Adding the `verbose=true` keyword to a solver call will trigger a warning message reporting descent violations along with the size of the violation. *In practice, we find that the violations are sufficiently small relative to the magnitude of objective values that they can be safely ignored.*

## Setup

Download the source code for this repository.

### Using Julia's package manager

```bash
Pkg.add(url="https://github.com/alanderos91/MMOptimizationAlgorithms")
```

### Using `git`

```bash
git clone https://github.com/alanderos91/MMOptimizationAlgorithms /a/path/to/MMOptimizationAlgorithms
```

### Reproducing Julia Environment

**The following assumes the top-level directory is `MMOptimizationAlgorithms`**.

In a Julia session, run the script

```julia
import Pkg

Pkg.activate("."); Pkg.instantiate()            # recreate from Manifest.toml
Pkg.activate("./scripts"); Pkg.instantiate()    # recreate from scripts/Manifest.toml
```

---

## Examples

### Constrained Least Squares via Proximal Distance Algorithms

<details>
<summary>Click to expand</summary>

The type `LeastSquaresProblem` handles minimization of objectives the form
$$
f(\beta) = \frac{1}{2}\|y - X\beta\|_{2}^{2} + \frac{\rho}{2}\mathrm{dist}(D\beta, S)^{2},
$$
where $\beta$ represents regression coefficients, $X$ is a design matrix, $y$ is a univariate response, $D$ is a fusion matrix, and $S$ is a constraint set. Specific kinds of least squares problems are handled according to the type of `problem.extras`.

</details>

#### Sparse Regression

<details>
<summary>Click to expand</summary>

In this case `problem.extras <: SparseRegression`, which sets $S \equiv S_{k}$, a sparsity set with at most $k$ nonzero components, and $D = I$, an identity matrix.

**Basic Example**

```julia
# using Revise # recommended if you plan on editing the source code
using MMOptimizationAlgorithms, Random
MMOA = MMOptimizationAlgorithms # abbreviate

n, p, k = 10^3, 2*10^3, 100     # number of samples, predictors, causal predictors
rng = Xoshiro(1234)             # random number generator w/ seed 1234

# Simulate a problem instance.
y, X, beta0 = MMOA.simulate_sparse_regression(n, p, k; rng=rng)

# Set algorithm options.
algorithm = SD()                    # steepest descent
options = set_options(algorithm;
    maxiter=500,                    # maximum iterations for fixed rho
    maxrhov=100,                    # maximum number of rho values to test
    gtol=1e-4,                      # converge for fixed rho: |∇f| < gtol OR |∇fₖ| < rtol*(1 + |∇fₖ₋₁|)
    dtol=1e-3,                      # overall convergence: dist < dtol OR distₖ < rtol*(1 + distₖ₋₁)
    rtol=1e-12,                     # relative tolerance used in both inner and outer iterations
    rhof=geometric_progression(1.2) # update rho -> 1.2 * rho in outer iterations
)
callback = VerboseCallback(10)      # print history every 10 MM steps

# Pass data to sparse regression solver and run.
result = @time sparse_regression(algorithm, y, X, k;
    options=options,
    callback=callback,
    pathf=naive_update,             # default: use warm-starts after changing rho
);

result.coefficients                 # coefficients after last iteration
result.projected                    # projection of coefficients after last iteration

# Check which of the true coefficients were selected.
findnz(x) = findall(xi -> abs(xi) > 0, x)
intersect(findnz(beta0), findnz(result.projected))
```

**Linear Extrapolation**

```julia
result = @time sparse_regression(algorithm, y, X, k;
    options=options,
    callback=callback,
    pathf=linear_update,            # update: xᵨ <- xᵨ + dxᵨ * Δρ
);
```

**Exponential Extrapolation**

```julia
result = @time sparse_regression(algorithm, y, X, k;
    options=options,
    callback=callback,
    pathf=exponential_update,       # update: xₙ <- xₙ + dxₙ * Δη; where ρ = exp(η)
);
```

</details>

#### Fused Lasso

<details>
<summary>Click to expand</summary>

In this case `problem.extras <: SparseRegression`, which sets $S \equiv \{y : \|y\|_{1} \le r\}$, the $\ell_{1}$ ball centered at the origin, and $D$ is a forward difference operator. For example, $Dx = x_{i} - x_{i-1}$.

**Basic Example**

```julia
# using Revise # recommended if you plan on editing the source code
using MMOptimizationAlgorithms, Random
MMOA = MMOptimizationAlgorithms # abbreviate

n, p, k = 10^3, 2*10^3, 100     # number of samples, predictors, causal predictors
rng = Xoshiro(1234)             # random number generator w/ seed 1234

# Simulate a problem instance.
y, X, beta0 = MMOA.simulate_sparse_regression(n, p, k; rng=rng)

# Set algorithm options.
algorithm = SD()                    # steepest descent
options = set_options(algorithm;
    maxiter=500,                    # maximum iterations for fixed rho
    maxrhov=100,                    # maximum number of rho values to test
    gtol=1e-4,                      # converge for fixed rho: |∇f| < gtol OR |∇fₖ| < rtol*(1 + |∇fₖ₋₁|)
    dtol=1e-3,                      # overall convergence: dist < dtol OR distₖ < rtol*(1 + distₖ₋₁)
    rtol=1e-12,                     # relative tolerance used in both inner and outer iterations
    rhof=geometric_progression(1.2) # update rho -> 1.2 * rho in outer iterations
)
callback = VerboseCallback(100)     # print history every 100 MM steps

# Pass data to fused lasso solver and run.
result = @time fused_lasso(algorithm, y, X, 1e1, 1e1;
    options=options,
    callback=callback,
);

result.coefficients                 # coefficients after last iteration
result.projected                    # projection of coefficients after last iteration

# Check which of the true coefficients were selected.
findnz(x) = findall(xi -> abs(xi) > 0, x)
intersect(findnz(beta0), findnz(result.projected))
```

**Linear Extrapolation**

```julia
result = @time fused_lasso(algorithm, y, X, 1e1, 1e1;
    options=options,
    callback=callback,
    pathf=linear_update,            # update: xᵨ <- xᵨ + dxᵨ * Δρ
);
```

**Exponential Extrapolation**

```julia
result = @time fused_lasso(algorithm, y, X, 1e1, 1e1;
    options=options,
    callback=callback,
    pathf=exponential_update,       # update: xₙ <- xₙ + dxₙ * Δη; where ρ = exp(η)
);
```

</details>

### Adaptive Trust Regions via Local Majorization

#### Styblinski-Tang

This example is adapted from `scripts/styblinski-tang.jl`. It requires the Roots.jl package.
The easiest way to run it is to run

```julia
import Pkg; Pkg.activate("./scripts")
```

<details>
<summary>Click here to expand</summary>

```julia
using MMOptimizationAlgorithms, LinearAlgebra, Roots, Random
MMOA = MMOptimizationAlgorithms # abbreviate

# Styblinski-Tang function; dimension is inferred from length of `x`.
function f(x)
    fx = zero(eltype(x))
    for i in eachindex(x)
        fx += x[i]^4 - 16*x[i]^2 +5*x[i]
    end
    return 1//2*fx
end

# Function to simultaneously estimate a local Lipschitz constant and trust region radius.
#
# Lambda is the smallest eigenvalue of the Hessian, estimated before this function is called.
# Taking `r` as the trust region radius, we prescribe `r = c * sqrt(norm(grad))` where `c` is to be determined.
# The formula below comes from solving the equation L * c / 3 = 1/c by taking into account the interdependency
# between the Lipschitz constant L, c, and radius r.
#
# The pair (L, c) is returned so we know the local Lipschitz constant over the trust region with
# radius c*sqrt(norm(grad)).
function estimatef(x, grad, lambda)
    g = sqrt(norm(grad))

    if iszero(lambda)
        c = (4*g)^(-1/3)
    elseif lambda < 0
        c = fzero(y -> 4*g*y^3 + lambda*y - 1, one(lambda))
    else
        error("Expected lambda <= 0")
    end
    L = 12*c*g # 12*r, as listed in the manuscript

    return L, c
end

rng = Xoshiro()
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
result = @time MMOA.trnewton(f, x_init; callback=callback, options=options, estimatef=estimatef)
println("\n  |x₁ - z₁| = $(abs(result[1] - z[1]));\tx₁ = $(result[1])")
println("  |f(x)-f(z)| = $(abs(f(result) - fz));\tf(x) = $(f(result))")

println("\nStep-Halving Newton")
result = @time MMOA.newton(f, x_init; callback=callback, options=options, nhalf=8)
println("\n  |x₁ - z₁| = $(abs(result[1] - z[1]));\tx₁ = $(result[1])")
println("  |f(x)-f(z)| = $(abs(f(result) - fz));\tf(x) = $(f(result))")
```

</details>

---
