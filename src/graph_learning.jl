"""
```
node_smoothing(alg, node_data; [alpha], [beta], [metric], [options], [callback])
```

Estimate a sparse adjacency graph from node signals.
"""
function node_smoothing(alg::AbstractMMAlg, node_data::Vector{Vector{T}};
    alpha::Real=1.0,
    beta::Real=1.0,
    metric::SemiMetric=Euclidean(),
    kwargs...
) where T
    # Initialize problem object.
    prob = GraphLearningProblem(node_data, metric, NodeSmoothing(node_data))
    hparams = (; alpha=alpha, beta=beta,)

    # Solve the problem along the penalty path.
    state = proxdist!(alg, prob, hparams; kwargs...)

    # Reshape into a matrix.
    @unpack m, weights = prob
    W = zeros(m, m)
    idx = 1
    for j in 1:m, i in 1:j-1
        W[i,j] = weights[idx]
        idx += 1
    end

    return (;
        state...,
        matrix=Symmetric(W, :U),
    )
end

"""
```
node_sparsity(alg, node_data, k; [alpha], [metric], [options], [callback])
```

Estimate a sparse adjacency graph from node signals.
"""
function node_sparsity(alg::AbstractMMAlg, node_data::Vector{Vector{T}}, k::Int;
    alpha::Real=1.0,
    metric::SemiMetric=Euclidean(),
    kwargs...
) where T
    # Initialize problem object.
    extras = NodeSparsity(node_data)
    prob = GraphLearningProblem(node_data, metric, extras)
    hparams = (; alpha=alpha, k=k,)

    # Solve the problem along the penalty path.
    state = proxdist!(alg, prob, hparams; kwargs...)

    # Project the final estimate.
    weights, P = extras.projected, extras.projection
    copyto!(weights, prob.weights)
    P(weights, k, one(T))

    # Reshape into a matrix.
    @unpack m = prob
    W = zeros(m, m)
    idx = 1
    for j in 1:m, i in 1:j-1
        W[i,j] = weights[idx]
        idx += 1
    end

    return (;
        state...,
        matrix=Symmetric(W, :U),
    )
end
