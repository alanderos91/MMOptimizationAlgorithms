"""
```
sparse_regression(alg, y, X, k; [options], [callback])
```

Estimate a `k`-sparse solution `beta` to the least squares problem `|y-X*beta|²`.
"""
function sparse_regression(alg::AbstractMMAlg, y::AbstractVector, X::AbstractMatrix, k::Int; kwargs...)
    # Initialize problem object.
    n, p = size(X)
    T = promote_type(eltype(y), eltype(X))
    ycp = convert(Vector{T}, y)
    Xcp = convert(Matrix{T}, X)
    prob = LeastSquaresProblem(ycp, Xcp, SparseRegression{T}(L0Projection, n, p))
    hparams = (; k=k,)

    # Solve the problem along the penalty path.
    state = proxdist!(alg, prob, hparams; kwargs...)

    return (;
        state...,
        coefficients=prob.coefficients,
        projected=prob.extras.projected,
    )
end

"""
```
fused_lasso(alg, y, X, r; [options], [callback])
```

Estimate a fused lasso solution `beta` to the least squares problem `|y-X*beta|²`.

The L1 ball radii, `r1` and `r2`, are described below:

- `r1` induces sparsity via the penalty `∑ⱼ |βⱼ| ≤ r1`.
- `r2` induces smoothness via the penalty `∑ⱼ |βⱼ-βⱼ₊₁| ≤ r2`.
"""
function fused_lasso(alg::AbstractMMAlg, y::AbstractVector, X::AbstractMatrix, r1::Real, r2::Real; kwargs...)
    # Initialize problem object.
    n, p = size(X)
    T = promote_type(eltype(y), eltype(X))
    ycp = convert(Vector{T}, y)
    Xcp = convert(Matrix{T}, X)
    prob = LeastSquaresProblem(ycp, Xcp, FusedLasso{T}(n, p))
    hparams = (; radius1=r1, radius2=r2,)

    # Solve the problem along the penalty path.
    state = proxdist!(alg, prob, hparams; kwargs...)

    return (;
        state...,
        coefficients=prob.coefficients,
        projected=prob.extras.projected1,
        differences=prob.extras.differences,
        projected_differences=prob.extras.projected2,
    )
end
