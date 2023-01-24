"""
Represents a portfolio optimization problem.
"""
struct PortfolioProblem{T<:AbstractFloat, S} <: AbstractProblem
    C::Vector{Matrix{T}}
    R::Matrix{T}
    W::Matrix{T}
    n_assets::Int
    n_periods::Int
    gradient::Matrix{T}
    extras::S
    
    function PortfolioProblem(C, R, extras::S) where S
        # Sanity checks
        if any(Cj -> size(Cj, 1) != size(Cj, 2) || size(Cj) != size(first(C)), C)
            error("Covariance matrices must be square have the same dimensions.")
        end
        if length(C) != size(R, 2)
            error("Number of covariance matrices ($(length(C))) does not match the number of asset return periods ($(size(R, 1)).")
        end
        if size(first(C), 1) != size(R, 1)
            error("Inferred number of assets in covariances ($(size(first(C), 1))) does not match the number of assets ($(size(R, 1)).")
        end

        n_assets, n_periods = size(R)
        T = eltype(first(C))
        W = 1/n_assets * ones(n_assets, n_periods)
        gradient = fill!(similar(W), zero(T))
        new{T,S}(C, R, W, n_assets, n_periods, gradient, extras)
    end
end

"""
Returns a tuple `(n_assets, n_periods)`.
"""
probdims(prob::PortfolioProblem) = (prob.n_assets, prob.n_periods)
float_type(prob::PortfolioProblem) = eltype(prob.W)

function nesterov_acceleration!(prob::PortfolioProblem, nesterov_iter, needs_reset)
    x, y = prob.W, prob.extras.W
    nesterov_acceleration!(x, y, nesterov_iter, needs_reset)
end

function save_for_warm_start!(prob::PortfolioProblem)
    copyto!(prob.extras.W, prob.W)
    return nothing
end

struct PortfolioQuadratic{T,APT<:AffineProjection,LHS,RHS,CHOL}
    W::Matrix{T}
    L::Matrix{T}
    W_differences::Matrix{T}         # Lw
    residual1::Matrix{T}             # P₁(w) - w
    residual2::Matrix{T}             # P₂(Lw) - Lw
    residual3::Matrix{T}             # P₃(w) - w
    projected1::Matrix{T}            # P₁(w)
    projected2::Matrix{T}            # P₂(Lw)
    projected3::Matrix{T}            # P₃(w)
    projection1::L1BallProjection{T} # P₁
    projection2::L1BallProjection{T} # P₂
    projection3::APT                 # P₃
    lhs::LHS
    rhs::RHS
    chol::CHOL
end

function PortfolioQuadratic{T}(alg::AbstractMMAlg, R, wealth_init, wealth_term) where T<:Real
    n_assets, n_periods = size(R)

    W = zeros(n_assets, n_periods)
    L = wealth_difference_matrix(n_assets, n_periods)
    W_differences = zeros(n_assets, n_periods-1)

    residual1 = fill!(similar(W), zero(T))
    residual2 = fill!(similar(W_differences), zero(T))
    residual3 = fill!(similar(W), zero(T))

    projected1 = fill!(similar(W), zero(T))
    projected2 = fill!(similar(W_differences), zero(T))
    projected3 = fill!(similar(W), zero(T))

    A = portfolio_constraint_matrix(R)
    b = sparsevec([1, n_periods+1], [wealth_init, wealth_term])
    projection1 = L1BallProjection{T}(n_periods*n_assets)
    projection2 = L1BallProjection{T}((n_periods-1)*n_assets)
    projection3 = AffineProjection(A, b)

    lhs, rhs, chol = allocate_linear_solver(alg, T, n_assets, n_periods)
    APT, LHS, RHS, CHOL = typeof(projection3), typeof(lhs), typeof(rhs), typeof(chol)

    return PortfolioQuadratic{T,APT,LHS,RHS,CHOL}(
        W, L, W_differences,
        residual1, residual2, residual3, 
        projected1, projected2, projected3,
        projection1, projection2, projection3,
        lhs, rhs, chol
    )
end

function evaluate(::AbstractMMAlg, prob::PortfolioProblem, extras::PortfolioQuadratic, hparams)
    #
    @unpack tau, rho, alpha = hparams
    @unpack C, W, gradient = prob
    @unpack L, W_differences, residual1, residual2, residual3 = extras
    W1, W2, W3 = extras.projected1, extras.projected2, extras.projected3
    P1, P2, P3 = extras.projection1, extras.projection2, extras.projection3
    T = float_type(prob)

    # Project W to to ℓ₁ ball of raidus τ₁.
    copyto!(W1, W)
    P1(vec(W1), tau[1])

    # Project L vec(W) to to ℓ₁ ball of raidus τ₂.
    mul!(vec(W_differences), L, vec(W))
    copyto!(W2, W_differences)
    P2(vec(W2), tau[2])

    # Project W to the affine space {X : A vec(X) = b}.
    copyto!(W3, W)
    P3(vec(W3))

    # Partially evaluate gradient, C vec(W), and evaluate loss ∑ⱼ wⱼᵀ Cⱼ wⱼ.
    loss = zero(T)
    for j in eachindex(C)
        g = view(gradient, :, j)
        w = view(W, :, j)
        mul!(g, C[j], w)
        loss += dot(g, w)
    end

    # Evaluate distance residual, P₁(W) - W.
    copyto!(residual1, W1)
    axpy!(-one(T), W, residual1)

    # Evaluate distance residual, P₂(L vec(W)) - L vec(W).
    copyto!(residual2, W2)
    axpy!(-one(T), W_differences, residual2)

    # Evaluate distance residual, P₃(W) - W.
    copyto!(residual3, W3)
    axpy!(-one(T), W, residual3)

    # Complete the gradient, C vec(W) - ρ ∑ₖ Dₖᵀ[Pₖ(DₖW) - Dₖ].
    axpy!(-rho*alpha[1], residual1, gradient)
    mul!(vec(gradient), transpose(L), vec(residual2), -rho*alpha[2], one(T))
    axpy!(-rho*alpha[3], residual3, gradient)

    d1 = dot(residual1, residual1)
    d2 = dot(residual2, residual2)
    d3 = dot(residual3, residual3)

    # Evaluate current state.
    loss = loss
    distsq = alpha[1]*d1 + alpha[2]*d2 + alpha[3]*d3
    objective = 0.5 * (loss + rho*distsq)
    gradsq = dot(gradient, gradient)

    return (; loss=loss, objective=objective, distance=sqrt(distsq), gradient=sqrt(gradsq),)
end

function allocate_linear_solver(::SD, T, n_assets, n_periods)
    return nothing, nothing, nothing
end

function mm_step!(alg::SD, prob::PortfolioProblem, extras::PortfolioQuadratic, hparams)
    #
    T = float_type(prob)
    @unpack rho, alpha = hparams
    @unpack W, C = prob
    G = prob.gradient
    V = extras.W_differences
    L = extras.L

    evaluate(alg, prob, extras, hparams)

    mul!(vec(V), L, vec(G))

    a = dot(G, G)
    b = dot(V, V)
    c = zero(T)
    for j in eachindex(C)
        g = view(G, :, j)
        c += dot(g, C[j], g)
    end
    t = a / (c + rho*(alpha[1]*a + alpha[2]*a + alpha[3]*b))

    axpy!(-t, G, W)

    return nothing
end

function allocate_linear_solver(::MML, T, n_assets, n_periods)
    m = n_assets*n_periods
    lhs = Matrix(one(T)*I(m))
    rhs = zeros(m)
    
    # @info "Initial Cholesky allocation"
    # @time begin
        chol = cholesky(Symmetric(lhs))
    # end

    return lhs, rhs, chol
end

function update_datastructures!(::MML, prob::PortfolioProblem, extras::PortfolioQuadratic, hparams)
    @unpack C, n_assets = prob
    @unpack L, lhs, chol = extras
    @unpack rho, alpha = hparams
    T = float_type(prob)

    a = rho * (alpha[1] + alpha[3])
    b = rho * alpha[2]

    # Reconstruct Hessian.
    # @info "Resetting entries"
    # @time begin
        fill!(lhs, zero(T))
    # end

    # @info "Reconstructing Hessian"
    # @time begin
        for j in eachindex(C)
        idx = n_assets*(j-1)+1 : n_assets*j 
        H = view(lhs, idx, idx)
        copyto!(H, C[j])
        H .= H + a*I
        end
        mul!(lhs, transpose(L), L, b, one(T))
    # end

    # Update its Cholesky decomposition.
    # @info "Updating Cholesky decomposition"
    # @time begin
        F = cholesky!(Symmetric(lhs))
        copyto!(chol.factors, F.factors)
    # end

    return nothing
end

function mm_step!(alg::MML, prob::PortfolioProblem, extras::PortfolioQuadratic, hparams)
    @unpack rho, alpha = hparams
    @unpack W, C = prob
    @unpack L, projected1, projected2, projected3, rhs, chol = extras

    evaluate(alg, prob, extras, hparams)

    # Form the RHS of linear system.
    copyto!(rhs, projected1)
    mul!(rhs, transpose(L), vec(projected2), rho*alpha[2], rho*alpha[1])
    axpy!(rho*alpha[3], vec(projected3), rhs)

    # Solve using the cached Cholesky decomposition.
    ldiv!(vec(W), chol, rhs)
    # vec(W) .= chol \ rhs

    return nothing
end

# function rho_sensitivity(prob::PortfolioProblem, extras::PortfolioQuadratic, rho, hparams)
#     X, RHS, beta = prob.design, extras.residuals, extras.projected
#     dbeta = similar(beta)
#     p = length(beta)
#     T = float_type(prob)

#     # negE = I - dP(x); negation of sparsity pattern
#     negE = zeros(T, p, p)
#     for j in axes(negE, 2)
#         negE[j,j] = ifelse(iszero(beta[j]), one(T), zero(T))
#     end

#     # solve for dbeta
#     A = transpose(X)*X + rho*negE
#     ldiv!(dbeta, lu!(A), RHS)

#     return prob.coefficients, dbeta
# end

function portfolio_constraint_matrix(R)
    n_assets, n_periods = size(R)
    A = spzeros(n_periods+1, n_periods*n_assets)
    for k in 1:n_periods
        Ak = view(A, (k-1)+1:k+1, (k-1)*n_assets+1:k*n_assets)
        Ak[1, :] .= 1
        Ak[2, :] .= (1 .+ view(R, :, k))
        if k < n_periods
            Ak[2, :] .*= -1
        end
    end
    return A
end

function wealth_difference_matrix(n_assets, n_periods)
    D = spzeros((n_periods-1)*n_assets, n_periods*n_assets)
    for k in 1:n_periods-1
        Dk = view(D, (k-1)*n_assets+1:k*n_assets, (k-1)*n_assets+1:(k+1)*n_assets)
        for j in 1:n_assets
            Dk[j,j] = -1
            Dk[j,n_assets+j] = 1
        end
    end
    return D
end
