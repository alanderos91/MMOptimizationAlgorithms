"""
Represents a graph learning problem, `∑ wᵢⱼ dist(xᵢ,xⱼ) - α ∑ log[∑ wᵢⱼ]`.
"""
struct GraphLearningProblem{T<:AbstractFloat,M,S} <: AbstractProblem
    m::Int  # number of nodes
    p::Int  # number of parameters to fit
    d::Int  # number of dimensions in node signal
    node_data::Vector{Vector{T}}    # node signals
    dist_data::Vector{T}            # cache for distance between node signals
    metric::M                       # distance metric on node signals
    weights::Vector{T}              # model paramters; stored in column-major order
    gradient::Vector{T}
    extras::S

    function GraphLearningProblem(node_data::Vector{Vector{T}}, metric::M, extras::S) where {T,M,S}
        m = length(node_data)
        d = length(node_data[1])
        p = binomial(m, 2)
        dist_data = zeros(T, p)
        #
        # Model paramters are upper triangular part of weighted adjacency matrix, W.
        # They are stored in a vector organized in column-major order.
        #
        #   | 1  2  3  4  5
        # -----------------
        # 1 | X  1  2  4  7
        # 2 | 1  X  3  5  8
        # 3 | 2  3  X  6  9
        # 4 | 4  5  6  X 10
        # 5 | 7  8  9 10  X
        #
        #
        idx = 1
        for j in 1:m, i in 1:j-1
            dist_data[idx] = metric(node_data[i], node_data[j])
            idx += 1
        end
        new{T,M,S}(m, p, d, node_data, dist_data, metric, ones(T, p), zeros(T, p), extras)
    end
end

probdims(prob::GraphLearningProblem) = (prob.m, prob.p, prob.d)
float_type(::GraphLearningProblem{T}) where T = T

function nesterov_acceleration!(prob::GraphLearningProblem, nesterov_iter, needs_reset)
    # x, y = prob.weights, prob.extras.weights
    # nesterov_acceleration!(x, y, nesterov_iter, needs_reset)
    return 0
end

function save_for_warm_start!(prob::GraphLearningProblem)
    # copyto!(prob.extras.weights, prob.weights)
    return nothing
end

function graph_learning_wsum(m, w, i)
    #
    wsum = zero(eltype(w))
    # sum along a row (left to right) in the upper triangular portion
    idx = binomial(i+1, 2)
    for j in i+1:m
        wsum += w[idx]
        idx += j-1
    end
    # sum along a row (right to left) in the lower triangular portion
    idx = binomial(i, 2)
    for j in i-1:-1:1
        wsum += w[idx]
        idx -= 1
    end
    return wsum
end

function graph_learning_loss(m, w, d, alpha)
    log_barrier_penalty = zero(alpha)
    for i in 1:m
        wsum = graph_learning_wsum(m, w, i)
        log_barrier_penalty += log(wsum)
    end
    return 2*dot(w, d) - alpha*log_barrier_penalty
end

function graph_learning_gradient!(grad, m, w, d, alpha)
    # Fill in gradient by moving along rows of upper triangular matrix.
    # This allows us to compute the row sum once and only once at the cost of
    # iterating through w in a nonstrided fashion.
    for i in 1:m
        wsum_i = graph_learning_wsum(m, w, i)
        idx = binomial(i+1, 2)
        for j in i+1:m
            wsum_j = graph_learning_wsum(m, w, j)
            grad[idx] = 2*d[idx] - alpha * (1/wsum_i + 1/wsum_j)
            idx += j-1
        end
    end
    return grad
end

####
#### Node Smoothing: Fatima, Arora, Babu, Stoica (2022)
####

struct NodeSmoothing{T}
    weights::Vector{T}
    buffer::Vector{T}
end

function NodeSmoothing(node_data::Vector{Vector{T}}) where T
    m = length(node_data)
    p = binomial(m, 2)
    return NodeSmoothing{T}(ones(T, p), zeros(T, p))
end

function evaluate(::AbstractMMAlg, prob::GraphLearningProblem, extras::NodeSmoothing, hparams)
    #
    @unpack alpha, beta = hparams
    m, w, d, grad = prob.m, prob.weights, prob.dist_data, prob.gradient

    # Evaluate unpenalized loss.
    loss = graph_learning_loss(m, w, d, alpha)

    # Evaluate ridge penalty on loss.
    penalty = beta * dot(w, w)

    # Evaluate the full gradient.
    graph_learning_gradient!(grad, m, w, d, alpha)
    axpy!(2*beta, w, grad)

    # Evaluate the current state.
    objective = loss + penalty
    gradsq = dot(grad, grad)

    return (; loss=loss, objective=objective, distance=zero(loss), gradient=sqrt(gradsq),)
end

function unsafe_quadratic(a, b, c)
    d = b^2 - 4*a*c
    if b > 0
        r1 = (-b - sqrt(d)) / (2*a)
    else
        r1 = (-b + sqrt(d)) / (2*a)
    end
    r2 = c / (r1 * a)
    return (r1, r2)
end

function mm_step!(alg::MMPS, prob::GraphLearningProblem, extras::NodeSmoothing, hparams)
    #
    @unpack alpha, beta = hparams
    T = float_type(prob)
    m, w, d = prob.m, prob.weights, prob.dist_data
    epsilon = sqrt(eps())

    # Cache old weight estimates so we can update in parallel.
    wn = extras.buffer
    copyto!(wn, w)

    # Apply updates in parallel.
    @batch per=core for i in 1:m
        wsum_i = graph_learning_wsum(m, wn, i)
        idx = binomial(i+1, 2)
        for j in i+1:m
            wsum_j = graph_learning_wsum(m, wn, j)
            A = 2*beta
            B = 2*d[idx]
            C = -alpha * (1/wsum_i + 1/wsum_j) * wn[idx]
            wnew = maximum(unsafe_quadratic(A, B, C))
            w[idx] = ifelse(wnew > epsilon, wnew, 0)
            idx += j-1
        end
    end
    @assert all(!isnan, w)
    @assert all(>=(0), w)
    return nothing
end

####
#### Node Sparsity
####

struct NodeSparsity{RNG,T}
    weights::Vector{T}
    buffer::Vector{T}
    projected::Vector{T}
    residuals::Vector{T}
    projection::SparseSimplexProjection{RNG,T}
end

function NodeSparsity(node_data::Vector{Vector{T}}) where T
    m = length(node_data)
    p = binomial(m, 2)
    P = SparseSimplexProjection{T}(p)
    RNG = typeof(P.rng) # hacky
    return NodeSparsity{RNG,T}(ones(T, p), zeros(T, p), zeros(T, p), zeros(T, p), P)
end

function evaluate(::AbstractMMAlg, prob::GraphLearningProblem, extras::NodeSparsity, hparams)
    #
    @unpack alpha, k, rho = hparams
    T = float_type(prob)
    m, w, d, grad = prob.m, prob.weights, prob.dist_data, prob.gradient
    proj, dist_res, P = extras.projected, extras.residuals, extras.projection

    # Project w to sparse unit simplex.
    copyto!(proj, w)
    P(proj, k, one(T))

    # Evaluate unpenalized loss.
    loss = graph_learning_loss(m, w, d, alpha)

    # Evaluate distance penalty on loss.
    copyto!(dist_res, proj)
    axpy!(-one(T), w, dist_res)
    penalty = dot(dist_res, dist_res)

    # Evaluate the full gradient.
    graph_learning_gradient!(grad, m, w, d, alpha)
    axpy!(-2*rho, dist_res, grad)

    # Evaluate the current state.
    objective = loss + rho*penalty
    gradsq = dot(grad, grad)

    return (; loss=loss, objective=objective, distance=sqrt(penalty), gradient=sqrt(gradsq),)
end

function mm_step!(alg::MMPS, prob::GraphLearningProblem, extras::NodeSparsity, hparams)
    #
    @unpack alpha, k, rho = hparams
    T = float_type(prob)
    m, w, d = prob.m, prob.weights, prob.dist_data
    epsilon = sqrt(eps())

    # Cache old weight estimates so we can update in parallel.
    wn, proj, P = extras.buffer, extras.projected, extras.projection
    copyto!(wn, w)

    # Project w to sparse unit simplex.
    copyto!(proj, w)
    P(proj, k, one(T))

    # Apply updates in parallel.
    @batch per=core for i in 1:m
        wsum_i = graph_learning_wsum(m, wn, i)
        idx = binomial(i+1, 2)
        for j in i+1:m
            wsum_j = graph_learning_wsum(m, wn, j)
            A = 2*rho
            B = 2*(d[idx] - rho*proj[idx])
            C = -alpha * (1/wsum_i + 1/wsum_j) * wn[idx]
            wnew = maximum(unsafe_quadratic(A, B, C))
            w[idx] = ifelse(wnew > epsilon, wnew, 0)
            idx += j-1
        end
    end
    @assert all(!isnan, w)
    @assert all(>=(0), w)
    return nothing
end

"""
Simulate graph signals X for the given graph G, where each realization xᵢ ~ N(0, L† + σI)
follows a multivariate normal distribution.

Here L is the Laplacian of G and L† is its Moore-Penrose pseudoinverse. The noise level σ
can be modified by the keyword argument `sigma`.

Based on:

Dong, Thanou, Frossard, and Vandergheynst.
"Learning Laplacian matrix in smooth graph signal representations".
IEEE Transactions on Signal Processing (Volume: 64, Issue: 23, December 2016) 
"""
function simulate_graph_signals(G::Graph, nsamples; sigma=1.0)
#
    A = adjacency_matrix(G)
    L = Symmetric(laplacian_matrix(G))
    chol = cholesky!(Symmetric(pinv(Matrix(L)) + sigma*I)) # this is a bad approach; use ARPACK or KrylovKit to help

    (n, m) = (nsamples, size(A, 1)) # samples × nodes
    node_data = [zeros(n) for _ in 1:m]
    for i in 1:n
        x = chol.L*randn(m)
        for j in 1:m
            node_data[j][i] = x[j]
        end
    end
    return (node_data, A, L)
end

function simulate_erdos_renyi_instance(nnodes::Int, nsamples::Int; prob::Real=0.1, kwargs...)
    er_graph = erdos_renyi(nnodes, prob)
    return simulate_graph_signals(er_graph, nsamples; kwargs...)
end
