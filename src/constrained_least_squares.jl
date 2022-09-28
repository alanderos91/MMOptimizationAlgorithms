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
    prob = LeastSquaresProblem(ycp, Xcp, SparseRegression{T}(n, p))
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

The L1 ball radius `r` controls the strength of the fused lasso via the penalty `∑ⱼ |βⱼ-βⱼ₊₁| ≤ r`.
"""
function fused_lasso(alg::AbstractMMAlg, y::AbstractVector, X::AbstractMatrix, r::Real; kwargs...)
    # Initialize problem object.
    n, p = size(X)
    T = promote_type(eltype(y), eltype(X))
    ycp = convert(Vector{T}, y)
    Xcp = convert(Matrix{T}, X)
    prob = LeastSquaresProblem(ycp, Xcp, FusedLasso{T}(n, p))
    hparams = (; radius=r,)

    # Solve the problem along the penalty path.
    state = proxdist!(alg, prob, hparams; kwargs...)

    return (;
        state...,
        coefficients=prob.coefficients,
        projected=prob.extras.projected,
    )
end
