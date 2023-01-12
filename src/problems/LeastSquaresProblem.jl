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

struct SparseRegression{T,projT}
    coefficients::Vector{T}
    projected::Vector{T}
    residuals::Vector{T}
    projection::projT
end

function SparseRegression{T}(::Type{projT}, n::Int, p::Int) where {T,projT}
    return SparseRegression(zeros(T, p), zeros(T, p), zeros(T, p), projT(p))
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
    F = StatsBase.fit(ZScoreTransform, X, dims=1)
    StatsBase.transform!(F, X)
    b0 = zeros(p)
    b0[1:k] .= 1
    shuffle!(rng, b0)
    y = X * b0
    return y, X, b0
end

function rho_sensitivity(prob::LeastSquaresProblem, extras::SparseRegression, rho, hparams)
    X, RHS, beta = prob.design, extras.residuals, extras.projected
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
    ldiv!(dbeta, lu!(A), RHS)

    return prob.coefficients, dbeta
end

struct FusedLasso{T}
    coefficients::Vector{T}
    projected1::Vector{T}
    projected2::Vector{T}
    differences::Vector{T}
    residuals1::Vector{T}
    residuals2::Vector{T}
    projection1::L1BallProjection{T}
    projection2::L1BallProjection{T}
end

function FusedLasso{T}(n::Int, p::Int) where T
    return FusedLasso{T}(
        zeros(T, p),
        zeros(T, p), zeros(T, p-1),
        zeros(T, p-1), 
        zeros(T, p), zeros(T, p-1),
        L1BallProjection{T}(p), L1BallProjection{T}(p-1)
    )
end

function evaluate(::AbstractMMAlg, prob::LeastSquaresProblem, extras::FusedLasso, hparams)
    #
    @unpack radius1, radius2, rho = hparams
    T = float_type(prob)
    y, X = prob.response, prob.design
    beta, res, grad = prob.coefficients, prob.residuals, prob.gradient
    proj1, proj2 = extras.projected1, extras.projected2
    dres1, dres2 = extras.residuals1, extras.residuals2
    P1, P2 = extras.projection1, extras.projection2
    diff = extras.differences

    # Project β onto the L1 ball with the given radius, P(β)
    copyto!(proj1, beta)
    P1(proj1, radius1)

    # Project Dβ onto the L1 ball with the given radius, P(Dβ).
    fd!(diff, beta)
    copyto!(proj2, diff)
    P2(proj2, radius2)
    
    # Compute residuals, y - X β, P(β) - β, and P(Dβ) - Dβ.
    copyto!(res, y);        mul!(res, X, beta, -one(T), one(T))
    copyto!(dres1, proj1);  axpy!(-one(T), beta, dres1)
    copyto!(dres2, proj2);  axpy!(-one(T), diff, dres2)

    # Evaluate gradient, -Xᵀ(y - X β) - ρ[P(β) - β] - ρDᵀ[P(Dβ) - Dβ]
    fill!(grad, zero(T))
    mul!(grad, X', res, -one(T), zero(T))
    axpy!(-rho, dres1, grad)
    fdt_axpy!(-rho, dres2, grad)

    # Evaluate current state.
    loss = dot(res, res)
    distsq = dot(dres1, dres1) + dot(dres2, dres2)
    objective = 0.5 * (loss + rho*distsq)
    gradsq = dot(grad, grad)

    return (; loss=loss, objective=objective, distance=sqrt(distsq), gradient=sqrt(gradsq),)
end

function mm_step!(alg::MMPS, prob::LeastSquaresProblem, extras::FusedLasso, hparams)
    #
    @unpack rho = hparams
    X, beta, g = prob.design, prob.coefficients, prob.gradient
    evaluate(alg, prob, extras, hparams)

    p = length(beta)
    normsqX = norm(X, 2)^2
    normsqD = 2*(p-1)
    t = inv(normsqX + rho*(p + normsqD))
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
    fd!(d, g)
    a, b, c = dot(g, g), dot(r, r), dot(d, d)
    t = ifelse(iszero(a) && iszero(b) && iszero(c), zero(T), a/(b + rho*(a + c)))

    axpy!(-t, g, beta)

    return nothing
end

function rho_sensitivity(prob::LeastSquaresProblem, extras::FusedLasso, rho, hparams)
    X, beta, RHS = prob.design, prob.coefficients, prob.gradient
    dres1, dres2 = extras.residuals1, extras.residuals2
    Dbeta = extras.differences

    function P1(x::AbstractVector{T}) where T
        y = copy(x)
        L1BallProjection{T}(length(y))(y, T(hparams.radius1))
        return y
    end
    function P2(x::AbstractVector{T}) where T
        y = copy(x)
        L1BallProjection{T}(length(y))(y, T(hparams.radius2))
        return y
    end
    T = float_type(prob)
    p = length(beta)
    dbeta = zeros(T, p)
    A = zeros(T, p, p)
    tmp = zeros(T, p-1, p)

    # Compute the differentials dP(β) and dP(Dβ).
    dP1 = ForwardDiff.jacobian(P1, beta)
    dP1 .= I - dP1
    fd!(Dbeta, beta)
    dP2 = ForwardDiff.jacobian(P2, Dbeta)
    dP2 .= I - dP2

    # Compute LHS: ∇²f(β) + ρ[I - dP(β)] + ρDᵀ[I - dP(Dβ)]D.
    for j in axes(tmp, 1) # compute V = I-dP(Dβ) * D
        dst = view(tmp, j, :)
        src = view(dP2, :, j)
        fdt!(dst, src)
    end
    for j in axes(A, 2) # compute DᵀV
        dst = view(A, :, j)
        src = view(tmp, :, j)
        fdt!(dst, src)
    end
    axpy!(one(T), dP1, A)
    mul!(A, transpose(X), X, one(T), T(rho))

    # Compute RHS: [P(β) - β] + Dᵀ[P(Dβ) - Dβ]
    fdt!(RHS, dres2)
    axpy!(one(T), dres1, RHS)

    # Solve for dbeta.
    ldiv!(dbeta, lu!(A), RHS)

    return prob.coefficients, dbeta
end
