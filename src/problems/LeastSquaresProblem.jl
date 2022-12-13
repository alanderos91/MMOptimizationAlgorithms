"""
Represents a least squares problem, `|y - X β|²`.

Provides the following fields
- `response`: The continuous response variable, `y`, as a vector.
- `design`: The design matrix, `X`.
- `coefficients`: The coefficients vector, `β`.
- `residuals`: The unscaled residuals, `r = y - X β`.
- `gradient`: The unscaled gradient, `Xᵀr`.
- `extras`: Stores additional fields specific to constraints.
"""
struct LeastSquaresProblem{T<:AbstractFloat,S} <: AbstractProblem
    response::Vector{T}
    design::Matrix{T}
    coefficients::Vector{T}
    residuals::Vector{T}
    gradient::Vector{T}
    extras::S

    function LeastSquaresProblem(y::Vector{T}, X::Matrix{T}, extras::S) where {T<:AbstractFloat,S}
        n, p = size(X)
        new{T,S}(y, X, zeros(T, p), zeros(T, n), zeros(T, p), extras)
    end
end

"""
Returns a tuple `(nsamples, npredictors)`.
"""
probdims(prob::LeastSquaresProblem) = size(prob.design)
float_type(prob::LeastSquaresProblem) = eltype(prob.coefficients)

function nesterov_acceleration!(prob::LeastSquaresProblem, nesterov_iter, needs_reset)
    x, y = prob.coefficients, prob.extras.coefficients
    nesterov_acceleration!(x, y, nesterov_iter, needs_reset)
end

function save_for_warm_start!(prob::LeastSquaresProblem)
    copyto!(prob.extras.coefficients, prob.coefficients)
    return nothing
end

struct SparseRegression{T}
    coefficients::Vector{T}
    projected::Vector{T}
    residuals::Vector{T}
    projection::L0Projection
end

function SparseRegression{T}(n::Int, p::Int) where T
    return SparseRegression{T}(zeros(T, p), zeros(T, p), zeros(T, p), L0Projection(p))
end

function evaluate(::AbstractMMAlg, prob::LeastSquaresProblem, extras::SparseRegression, hparams)
    #
    @unpack k, rho = hparams
    T = float_type(prob)
    y, X = prob.response, prob.design
    beta, res, grad = prob.coefficients, prob.residuals, prob.gradient
    proj, dist_res, P = extras.projected, extras.residuals, extras.projection

    # Project β to sparsity set Sₖ, P(β).
    copyto!(proj, beta)
    P(proj, k)
    
    # Compute residuals, y - X β and P(β) - β.
    copyto!(res, y)
    mul!(res, X, beta, -one(T), one(T))
    copyto!(dist_res, proj)
    axpy!(-one(T), beta, dist_res)

    # Evaluate gradient, -Xᵀ(y - X β) - ρ[P(β) - β]
    mul!(grad, X', res, -one(T), zero(T))
    axpy!(-rho, dist_res, grad)

    # Evaluate current state.
    loss = dot(res, res)
    distsq = dot(dist_res, dist_res)
    objective = 0.5 * (loss + rho*distsq)
    gradsq = dot(grad, grad)

    return (; loss=loss, objective=objective, distance=sqrt(distsq), gradient=sqrt(gradsq),)
end

function mm_step!(alg::MMPS, prob::LeastSquaresProblem, extras::SparseRegression, hparams)
    #
    @unpack rho = hparams
    X, beta, r, g = prob.design, prob.coefficients, prob.residuals, prob.gradient
    proj = extras.projected
    evaluate(alg, prob, extras, hparams)
    
    mul!(g, X', r)
    normsqX = norm(X, 2)^2
    a = normsqX / (normsqX + rho)
    @batch per=core for i in eachindex(beta)
        beta[i] = a*beta[i] + (1-a)*(proj[i] + 1/rho*g[i])
    end
    
    return nothing
end

function mm_step!(alg::SD, prob::LeastSquaresProblem, extras::SparseRegression, hparams)
    #
    T = float_type(prob)
    @unpack rho = hparams
    X, beta, r, g = prob.design, prob.coefficients, prob.residuals, prob.gradient
    evaluate(alg, prob, extras, hparams)

    mul!(r, X, g)
    a, b = dot(g, g), dot(r, r)
    t = ifelse(iszero(a) && iszero(b), zero(T), a/(b + rho*a))

    axpy!(-t, g, beta)

    return nothing
end

function simulate_sparse_regression(n::Integer, p::Integer, k::Integer; rng::AbstractRNG=Xoshiro())
    X = randn(rng, n, p)
    b0 = zeros(p)
    b0[1:k] .= 1
    shuffle!(rng, b0)
    y = X * b0
    return y, X, b0
end

function rho_sensitivity(prob::LeastSquaresProblem, extras::SparseRegression, rho)
    X, q, beta = prob.design, extras.residuals, extras.projected
    dbeta = similar(beta)
    p = length(beta)
    T = float_type(prob)

    # negE = I - dP(x); negation of sparsity pattern
    negE = zeros(T, p, p)
    for j in axes(negE, 2)
        negE[j,j] = ifelse(iszero(beta[j]), one(T), zero(T))
    end

    # solve for dbeta
    A = transpose(X)*X + rho*negE
    ldiv!(dbeta, cholesky!(A), q)
    @. dbeta = -dbeta

    return prob.coefficients, dbeta
end

function eta_sensitivity(prob::LeastSquaresProblem, extras::SparseRegression, rho)
    X, q, beta = prob.design, extras.residuals, extras.projected
    dbeta = similar(beta)
    p = length(beta)
    T = float_type(prob)

    # negE = I - dP(x); negation of sparsity pattern
    negE = zeros(T, p, p)
    for j in axes(negE, 2)
        negE[j,j] = ifelse(iszero(beta[j]), one(T), zero(T))
    end

    # remember that ρ = exp(η)
    A = transpose(X)*X + rho*negE
    ldiv!(dbeta, cholesky!(A), q)
    @. dbeta = -rho * dbeta

    return prob.coefficients, dbeta
end

struct FusedLasso{T}
    coefficients::Vector{T}
    projected::Vector{T}
    differences::Vector{T}
    residuals::Vector{T}
    projection::L1BallProjection{T}
end

function FusedLasso{T}(n::Int, p::Int) where T
    return FusedLasso{T}(zeros(T, p), zeros(T, p-1), zeros(T, p-1), zeros(T, p-1), L1BallProjection{T}(p-1))
end

function evaluate(::AbstractMMAlg, prob::LeastSquaresProblem, extras::FusedLasso, hparams)
    #
    @unpack radius, rho = hparams
    T = float_type(prob)
    y, X = prob.response, prob.design
    beta, res, grad = prob.coefficients, prob.residuals, prob.gradient
    proj, diff, dist_res, P = extras.projected, extras.differences, extras.residuals, extras.projection

    # Project Dβ onto the L1 ball with the given radius, P(Dβ).
    for i in eachindex(diff)
        diff[i] = beta[i] - beta[i+1]
    end
    copyto!(proj, diff)
    P(proj, radius)
    
    # Compute residuals, y - X β and P(β) - β.
    copyto!(res, y)
    mul!(res, X, beta, -one(T), one(T))
    copyto!(dist_res, proj)
    axpy!(-one(T), diff, dist_res)

    # Evaluate gradient, -Xᵀ(y - X β) - ρDᵀ[P(Dβ) - Dβ]
    mul!(grad, X', res, -one(T), zero(T))
    grad[1] = grad[1] - rho*dist_res[1]
    for i in 1:length(dist_res)-1
        grad[i+1] = grad[i+1] + rho*(dist_res[i] - dist_res[i+1])
    end
    grad[end] = grad[end] + rho*dist_res[end]

    # Evaluate current state.
    loss = dot(res, res)
    distsq = dot(dist_res, dist_res)
    objective = 0.5 * (loss + rho*distsq)
    gradsq = dot(grad, grad)

    return (; loss=loss, objective=objective, distance=sqrt(distsq), gradient=sqrt(gradsq),)
end

function mm_step!(alg::MMPS, prob::LeastSquaresProblem, extras::FusedLasso, hparams)
    #
    @unpack rho = hparams
    X, beta, g = prob.design, prob.coefficients, prob.gradient
    evaluate(alg, prob, extras, hparams)

    normsqX = norm(X, 2)^2
    normsqD = 2*(length(beta)-1)
    t = 1 / (normsqX + rho*normsqD)
    axpy!(-t, g, beta)
    
    return nothing
end

function mm_step!(alg::SD, prob::LeastSquaresProblem, extras::FusedLasso, hparams)
    #
    T = float_type(prob)
    @unpack rho = hparams
    X, beta, r, g = prob.design, prob.coefficients, prob.residuals, prob.gradient
    d = extras.differences
    evaluate(alg, prob, extras, hparams)

    mul!(r, X, g)
    for i in eachindex(d)
        d[i] = g[i] - g[i+1]
    end
    a, b, c = dot(g, g), dot(r, r), dot(d, d)
    t = ifelse(iszero(a) && iszero(b) && iszero(c), zero(T), a/(b + rho*c))

    axpy!(-t, g, beta)

    return nothing
end
