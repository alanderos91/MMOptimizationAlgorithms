module MMOptimizationAlgorithms

using Printf, UnPack
using Random, StatsBase
using LinearAlgebra, Polyester

import Base: show

export AlgOptions, set_options,
    MML, MMPS, SD,
    VerboseCallback, HistoryCallback,
    sparse_regression,
    fused_lasso

abstract type AbstractMMAlg end

"""
MM w/ linearization.

Majorize an objective with a quadratic surrogate and minimize it.
"""
struct MML <: AbstractMMAlg end

"""
MM w/ parameter separation.

Majorize an objective with a surrogate that separates parameters.
"""
struct MMPS <: AbstractMMAlg end

"""
Steepest descent.

Majorize an objective with a surrogate that admits an exact line search in steepest descent.
"""
struct SD <: AbstractMMAlg end

"""
Abstract type for representing different kinds of problems. Must provide the field `extras`.
"""
abstract type AbstractProblem end

evaluate(alg, prob::AbstractProblem, hparams) = evaluate(alg, prob, prob.extras, hparams)
mm_step!(alg, prob::AbstractProblem, hparams) = mm_step!(alg, prob, prob.extras, hparams)
save_for_warm_start!(prob::AbstractProblem) = save_for_warm_start!(prob, prob.extras)

include(joinpath("projections", "L0Projection.jl"))
include(joinpath("projections", "SimplexProjection.jl"))
include(joinpath("projections", "L1BallProjection.jl"))
include(joinpath("projections", "SparseSimplexProjection.jl"))

include("callbacks.jl")
include("utilities.jl")

include(joinpath("problems", "LeastSquaresProblem.jl"))

"""
Placeholder for callbacks in main functions.
"""
__do_nothing_callback__((iter, state), problem, hyperparams) = nothing

const DEFAULT_CALLBACK = __do_nothing_callback__

function proxdist!(algorithm::AbstractMMAlg, problem::AbstractProblem, init_hyperparams;
    options::AlgOptions{G}=default_options(algorithm),
    callback::F=DEFAULT_CALLBACK,
) where {F,G}
    # Get algorithm options.
    @unpack maxrhov, gtol, dtol, rtol, rho_init, rho_max, rhof = options

    # Initialize ρ and iteration count.
    rho = rho_init
    iters = 0
    hyperparams = (; init_hyperparams..., rho=rho_init,)

    # Update data structures due to hyperparameters.

    # Check initial values for loss, objective, distance, and norm of gradient.
    state = evaluate(algorithm, problem, hyperparams)
    callback((-1, state), problem, hyperparams)
    old = state.distance

    is_proximate = old < dtol
    is_stationary = state.gradient < gtol

    for iter in 1:maxrhov
        # Solve minimization problem for fixed rho.
        (inner_iters, is_stationary, state) = solve!(algorithm, problem, hyperparams;
            options=options,
            callback=callback,
        )

        # Update total iteration count.
        iters += inner_iters

        # Check for convergence to constrained solution.
        dist = state.distance
        is_proximate = dist < dtol || abs(dist - old) < rtol * (1 + old)
        if is_proximate
            break
        else
            old = dist
        end

        # Update according to annealing schedule.
        rho = ifelse(iter < maxrhov, rhof(rho, iter, rho_max), rho)
        hyperparams = (; hyperparams..., rho=rho,)
    end

    # Evaluate objective at the final solution estimate.
    state = evaluate(algorithm, problem, hyperparams)

    return (;
        options=options,
        iters=iters,
        is_proximate=is_proximate,
        is_stationary=is_stationary,
        hyperparams...,
        state...,
    )
end

function solve!(algorithm::AbstractMMAlg, problem::AbstractProblem, hyperparams;
    options::AlgOptions=default_options(algorithm),
    callback::F=DEFAULT_CALLBACK,
) where F
    # Get algorithm options.
    @unpack maxiter, gtol, nesterov = options

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1

    # Check initial values for loss, objective, distance, and norm of gradient.
    state = evaluate(algorithm, problem, hyperparams)
    callback((0, state), problem, hyperparams)
    old = state.objective

    is_stationary = false

    for iter in 1:maxiter
        iters += 1

        # Apply an algorithm map to update estimates.
        mm_step!(algorithm, problem, hyperparams)

        # Update loss, objective, distance, and gradient.
        state = evaluate(algorithm, problem, hyperparams)
        callback((iter, state), problem, hyperparams)

        # Assess convergence.
        is_stationary = state.gradient < gtol
        if is_stationary
            break
        elseif iter < maxiter
            has_increased = state.objective > old
            needs_reset = iter < nesterov || has_increased
            nesterov_iter = nesterov_acceleration!(problem, nesterov_iter, needs_reset)
            old = state.objective
        end
    end

    # Save parameter estimates in case of warm start.
    save_for_warm_start!(problem)

    return (iters, is_stationary, state)
end

include("constrained_least_squares.jl")

end # module
