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

# Setup

---

# Examples

---

## Least Squares

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
callback = VerboseCallback(10)      # print history every 10 MM steps

# Pass data to sparse regression solver and run.
result = @time fused_lasso(algorithm, y, X, 1e1;
    options=options,
    callback=callback,
);

result.coefficients                 # coefficients after last iteration
result.projected                    # projection of coefficients after last iteration

# Check which of the true coefficients were selected.
findnz(x) = findall(xi -> abs(xi) > 0, x)
intersect(findnz(beta0), findnz(result.projected))
```

</details>

---

## Graph Learning

#### Node Smoothing

<details>
<summary>Click to expand</summary>

```julia
using MMOptimizationAlgorithms, Random
MMOA = MMOptimizationAlgorithms # abbreviate

nnodes = 10
nsamples = 10^3
prob = 0.2

# Simulate data. Returns node signals x, adjacency matrix A, and Laplacian L
x, A, L = MMOA.simulate_erdos_renyi_instance(nnodes, nsamples, prob=prob)

# Set algorithm options.
algorithm = MMPS()                  # steepest descent
options = set_options(algorithm;
    maxiter=10^3,                   # maximum iterations for fixed rho
    maxrhov=100,                    # maximum number of rho values to test
    gtol=1e-2,                      # converge for fixed rho: |∇f| < gtol OR |∇fₖ| < rtol*(1 + |∇fₖ₋₁|)
    dtol=1e-2,                      # overall convergence: dist < dtol OR distₖ < rtol*(1 + distₖ₋₁)
    rtol=1e-12,                     # relative tolerance used in both inner and outer iterations
    rhof=geometric_progression(1.2) # update rho -> 1.2 * rho in outer iterations
)
callback = VerboseCallback(10)      # print history every 10 MM steps

result = @time node_smoothing(MMPS(), x;
    alpha=1e0,                      # penalty coefficient on node degrees
    beta=1e0,                       # strength of ridge penalty
    options=options,
    callback=callback,
);

result.matrix                       # inferred adjacency matrix
```

</details>

#### Explicit Sparsity

<details>
<summary>Click to expand</summary>

```julia
using MMOptimizationAlgorithms, Random
MMOA = MMOptimizationAlgorithms # abbreviate

nnodes = 10
nsamples = 10^3
prob = 0.2

# Simulate data. Returns node signals x, adjacency matrix A, and Laplacian L
x, A, L = MMOA.simulate_erdos_renyi_instance(nnodes, nsamples, prob=prob)

# Set algorithm options.
algorithm = MMPS()                  # steepest descent
options = set_options(algorithm;
    maxiter=10^3,                   # maximum iterations for fixed rho
    maxrhov=100,                    # maximum number of rho values to test
    gtol=1e-2,                      # converge for fixed rho: |∇f| < gtol OR |∇fₖ| < rtol*(1 + |∇fₖ₋₁|)
    dtol=1e-2,                      # overall convergence: dist < dtol OR distₖ < rtol*(1 + distₖ₋₁)
    rtol=1e-12,                     # relative tolerance used in both inner and outer iterations
    rhof=geometric_progression(1.2) # update rho -> 1.2 * rho in outer iterations
)
callback = VerboseCallback(10)      # print history every 10 MM steps
k = div(count(>(0), A), 2)

result = @time node_sparsity(MMPS(), x, k;
    alpha=1e0,                      # penalty coefficient on node degrees
    options=options,
    callback=callback,
);

result.matrix                       # inferred adjacency matrix
```

</details>

---