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

struct PortfolioQuadratic{T,APT<:AffineProjection,LS}
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
    linear_solve::LS
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

    linear_solve = portfolio_pq_allocate_linear_solver(alg, T, n_assets, n_periods)
    APT, LS = typeof(projection3), typeof(linear_solve)

    return PortfolioQuadratic{T,APT,LS}(
        W, L, W_differences,
        residual1, residual2, residual3, 
        projected1, projected2, projected3,
        projection1, projection2, projection3,
        linear_solve
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

function portfolio_pq_allocate_linear_solver(::SD, T, n_assets, n_periods)
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

function portfolio_pq_allocate_linear_solver(::MML, T, n_assets, n_periods)
    m = n_assets*n_periods
    lhs = Matrix(one(T)*I(m))
    rhs = zeros(m)
    
    chol = cholesky(Symmetric(lhs))

    return (;lhs=lhs, rhs=rhs, chol=chol)
end

function update_datastructures!(::MML, prob::PortfolioProblem, extras::PortfolioQuadratic, hparams)
    @unpack C, n_assets = prob
    @unpack L, linear_solve = extras
    @unpack lhs, rhs, chol = linear_solve
    @unpack rho, alpha = hparams
    T = float_type(prob)

    a = rho * (alpha[1] + alpha[3])
    b = rho * alpha[2]

    fill!(lhs, zero(T))

    for j in eachindex(C)
        idx = n_assets*(j-1)+1 : n_assets*j 
        H = view(lhs, idx, idx)
        copyto!(H, C[j])
        H .= H + a*I
    end
    mul!(lhs, transpose(L), L, b, one(T))

    F = cholesky!(Symmetric(lhs))
    copyto!(chol.factors, F.factors)

    return nothing
end

function mm_step!(alg::MML, prob::PortfolioProblem, extras::PortfolioQuadratic, hparams)
    @unpack rho, alpha = hparams
    @unpack W, C = prob
    @unpack L, projected1, projected2, projected3, linear_solve = extras
    @unpack rhs, chol = linear_solve

    evaluate(alg, prob, extras, hparams)

    # Form the RHS of linear system.
    copyto!(rhs, vec(projected1))
    mul!(rhs, transpose(L), vec(projected2), rho*alpha[2], rho*alpha[1])
    axpy!(rho*alpha[3], vec(projected3), rhs)

    # Solve using the cached Cholesky decomposition.
    ldiv!(vec(W), chol, rhs)

    return nothing
end

function portfolio_pq_allocate_linear_solver(::MMAL, T, n_assets, n_periods)
    m = n_assets * n_periods
    cholC = [zeros(T, n_assets, n_assets) for _ in 1:n_periods]
    M = zeros(T, m, m)
    S = zeros(T, m)
    Psi = Diagonal(zeros(T, m))
    buffer = zeros(T, m)
    rhs = zeros(m)
    (;cholC=cholC, M=M, S=S, Psi=Psi, buffer=buffer, rhs=rhs)
end

function update_datastructures!(::MMAL, prob::PortfolioProblem, extras::PortfolioQuadratic, hparams)
    function extract_cholesky!(dst, src)
        copyto!(dst, src)
        cholesky!(Symmetric(dst, :L))
    end
    @unpack C, n_assets, n_periods = prob
    @unpack L, linear_solve = extras
    @unpack Psi, S, cholC, M = linear_solve
    @unpack rho, alpha = hparams
    T = float_type(prob)

    needs_initialization = all(isequal(0), S)

    if needs_initialization
        # Compute Z = L⁻¹ Dᵀ
        m = n_assets * n_periods
        n = n_assets * (n_periods-1)
        Z = zeros(T, m, 2*m+n)

        Z[:, 1:m] .= sqrt(alpha[1]) * I(m)
        Z[:, m+1:m+n] .= sqrt(alpha[2]) * transpose(L)
        Z[:, m+n+1:2*m+n] .= sqrt(alpha[3]) * I(m)

        for j in eachindex(C)
            F = extract_cholesky!(cholC[j], C[j])
            rows = n_assets*(j-1)+1 : n_assets*j
            blk = view(Z, rows, :)
            ldiv!(F.L, blk)
        end
        # SVD of Z = U * S * Vᵀ
        F = svd!(Z, full=false, alg=LinearAlgebra.DivideAndConquer())
        copyto!(M, F.U)
        copyto!(S, F.S)

        # Compute M = (Lᵀ)⁻¹ U
        for j in eachindex(C)
            rows = n_assets*(j-1)+1 : n_assets*j
            L_j = LowerTriangular(cholC[j])
            ldiv!(transpose(L_j), view(M, rows, :))
        end

        @assert all(!isequal(0), S)
    end

    for i in eachindex(S)
        Psi.diag[i] = 1 / (S[i]^2 + inv(rho))
    end

    return nothing
end

function mm_step!(alg::MMAL, prob::PortfolioProblem, extras::PortfolioQuadratic, hparams)
    @unpack rho, alpha = hparams
    @unpack W, C, n_assets, n_periods = prob
    @unpack L, projected1, projected2, projected3, linear_solve = extras
    @unpack cholC, M, Psi, buffer, rhs = linear_solve
    T = float_type(prob)

    evaluate(alg, prob, extras, hparams)

    # Form the RHS of linear system.
    copyto!(rhs, vec(projected1))
    mul!(rhs, transpose(L), vec(projected2), alpha[2], alpha[1])
    axpy!(alpha[3], vec(projected3), rhs)

    # Solve by computing: vec(W) = M Ψ Mᵀ vec(RHS)
    mul!(buffer, transpose(M), rhs)
    lmul!(Psi, buffer)
    mul!(vec(W), M, buffer)

    return nothing
end

function rho_sensitivity(prob::PortfolioProblem, extras::PortfolioQuadratic, rho, hparams)
    @unpack alpha, tau = hparams
    @unpack W, C, n_assets, n_periods = prob
    @unpack W_differences, L, residual1, residual2, residual3, linear_solve = extras
    @unpack A, AAt, solver, r, y = extras.projection3
    @unpack rhs = linear_solve
    T = float_type(prob)
    m = n_assets * n_periods

    P1 = BallProjectionWrapper(L1BallProjection, tau[1])
    P2 = BallProjectionWrapper(L1BallProjection, tau[2])

    # Compute the differentials.
    dP1 = ForwardDiff.jacobian(P1, vec(W))
    dP1 .= I - dP1

    dP2 = transpose(L) * (I - ForwardDiff.jacobian(P2, vec(W_differences))) * L

    dP3 = zeros(T, m, m)
    for j in axes(A, 2)
        # Solve for column j of dP3: Aᵀ(AAᵀ)⁻¹A[:,j]
        # Object y is aliased to the solution in the CG step.
        colA_j = view(A, :, j)
        coldP3_j = view(dP3, :, j)
        copyto!(r, colA_j)
        cg!(solver, AAt, r)
        mul!(coldP3_j, transpose(A), y)
    end

    # Form LHS
    lhs = zeros(T, m, m)
    for j in eachindex(C)
        idx = n_assets*(j-1)+1 : n_assets*j 
        H = view(lhs, idx, idx)
        copyto!(H, C[j])
    end
    axpy!(rho*alpha[1], dP1, lhs)
    axpy!(rho*alpha[2], dP2, lhs)
    axpy!(rho*alpha[3], dP3, lhs)
    
    # Form RHS
    copyto!(rhs, vec(residual1))
    mul!(rhs, transpose(L), vec(residual2), alpha[2], alpha[1])
    axpy!(alpha[3], vec(residual3), rhs)

    # solve for dw
    dW = similar(W)
    ldiv!(vec(dW), cholesky!(Symmetric(lhs)), rhs)

    return prob.W, dW
end

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
