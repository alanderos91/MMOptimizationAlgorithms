"""
```
sparse_regression(alg, y, X; [options], [callback])
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
    # prob.coefficients .= (X'X) \ (X'y)
    # copyto!(prob.extras.coefficients, prob.coefficients)
    state = proxdist!(alg, prob, hparams; kwargs...)

    return (;
        state...,
        coefficients=prob.coefficients,
        projected=prob.extras.projected,
    )
end
