function portfolio_optimization(alg::AbstractMMAlg, C, R, wealth_init, wealth_term;
    tau=(1.0, 1.0),
    alpha=(1e-2, 1e-2, 1-2e-2),
    kwargs...
    )
    # Sanity checks.
    if any(<=(0), tau)
        error("Each penalty parameter tau must be positive. Got: $(tau).")
    end
    if any(<=(0), alpha)
        error("Each penalty parameter alpha must be positive. Got: $(alpha).")
    end
    if sum(alpha) != 1
        error("Alpha values must sum to 1. Got $(alpha) with sum equal to $(sum(alpha)).")
    end

    # Initialize problem object.
    extras = PortfolioQuadratic{Float64}(alg, R, wealth_init, wealth_term)
    prob = PortfolioProblem(C, R, extras)
    hparams = (; tau=tau, alpha=alpha)

    # Solve the problem along the penalty path.
    state = proxdist!(alg, prob, hparams; kwargs...)

    # extras.projection3(vec(extras.projected1))

    mul!(vec(extras.W_differences), extras.L, vec(extras.projected1))
    A, b = extras.projection3.A, extras.projection3.b
    r = copy(b)
    mul!(r, A, vec(extras.projected1), 1.0, -1.0)
    return (;
        state...,
        W=prob.W,
        projected=extras.projected1,
        differences=extras.W_differences,
        residual=r,
    )
end
