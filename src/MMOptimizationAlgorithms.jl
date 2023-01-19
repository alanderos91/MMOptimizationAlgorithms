module MMOptimizationAlgorithms

using Printf, UnPack
using Random, StatsBase
using Distances, Graphs, LinearAlgebra, ForwardDiff, Polyester

import Base: show
import ForwardDiff: Dual

export AlgOptions, set_options,
    MML, MMPS, SD, TRNewton,
    VerboseCallback, HistoryCallback,
    trnewton,
    newton,
    sparse_regression,
    fused_lasso,
    node_smoothing,
    node_sparsity,
    naive_update,
    linear_update,
    exponential_update,
    geometric_progression

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
Trust region Newton method.
"""
struct TRNewton <: AbstractMMAlg end

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
include(joinpath("problems", "GraphLearningProblem.jl"))

"""
Placeholder for callbacks in main functions.
"""
__do_nothing_callback__((iter, state), problem, hyperparams) = nothing

const DEFAULT_CALLBACK = __do_nothing_callback__

__evalone__(evalf, x, xold, iter, r) = 1.0

const DEFAULT_LIPSCHITZ = __evalone__

function proxdist!(algorithm::AbstractMMAlg, problem::AbstractProblem, init_hyperparams;
    options::AlgOptions{G}=default_options(algorithm),
    callback::F=DEFAULT_CALLBACK,
    pathf::H=naive_update,
) where {F,G,H}
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
        # rho = ifelse(iter < maxrhov, rhof(rho, iter, rho_max), rho)
        rho = if iter < maxrhov
            rho_new = rhof(rho, iter, rho_max)
            pathf(problem, rho, rho_new, hyperparams)
        else
            rho
        end
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
    @unpack maxiter, gtol, rtol, nesterov = options

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
        is_stationary = state.gradient < gtol || abs(state.objective - old) < rtol * (1 + old)
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

function trnewton(f::F, x0;
    options::AlgOptions=default_options(nothing),
    callback::G=DEFAULT_CALLBACK,
    estimatef::LFUN=DEFAULT_LIPSCHITZ,
    chunks::Int=1) where {F,G,LFUN}
    # Sanity checks.
    @unpack maxiter, gtol = options
    
    # Initialization.
    x, xold, v = copy(x0), copy(x0), similar(x0)
    result = DiffResults.HessianResult(x)
    cfg = ForwardDiff.HessianConfig(f, result, x, ForwardDiff.Chunk(chunks))

    evaluatef = let f=f, result=result, cfg=cfg
        function (x)
            r = ForwardDiff.hessian!(result, f, x, cfg)
            fx = DiffResults.value(r)
            dfx = DiffResults.gradient(r)
            d2fx = DiffResults.hessian(r)
            return r, fx, dfx, d2fx
        end
    end
    result, fold, grad, hess = evaluatef(x)
    H = similar(hess)

    state = (;
        objective=fold,
        gradient=norm(grad),
        trust_region_r=Inf,
        residual=Inf,
        lipschitz_L=Inf,
    )
    callback((-1, state), nothing, nothing)

    for iter in 1:maxiter
        # Find a valid positive definite approximation to the Hessian.
        lambda = eigmin(Symmetric(hess))
        if lambda <= 0
            lambda -= 1e-6
        elseif lambda > 0
            lambda = zero(lambda)
        end

        # Find a valid trust region radius.
        L, c = estimatef(x, grad, lambda)
        g = sqrt(norm(grad))
        r = c*g
        d = max(L*c/3 + lambda, 1/c)

        H .= Symmetric(hess) + (-lambda + d*g) * I

        # Solve for Newton increment.
        cholH = cholesky!(H)
        ldiv!(v, cholH, -grad)

        # Update and evaluate loss.
        x .= x + v 
        result, fnew, grad, hess = evaluatef(x) # re-alias
        state = (;
            objective=fnew,
            gradient=norm(grad),
            trust_region_r=r,
            residual=norm(x - xold),
            lipschitz_L=L,
        )
        callback((iter, state), nothing, nothing)

        # Check for descent and convergence.
        if fold < fnew
            @warn "Descent condition not satisfied at iteration $(iter)." new=fnew old=fold diff=fold-fnew
        elseif fold == fnew
            @warn "Reached a no-progress point." new=fnew old=fold diff=fold-fnew
            break
        elseif state.gradient < gtol
            break
        end
        fold = fnew
        xold .= x
    end
    return x
end

function newton(f::F, x0;
    options::AlgOptions=default_options(nothing),
    callback::G=DEFAULT_CALLBACK,
    nhalf::Int=3,
    chunks::Int=1) where {F,G,LFUN}
    # Sanity checks.
    @unpack maxiter, gtol = options
    
    # Initialization.
    x, xold, v = copy(x0), copy(x0), similar(x0)
    result = DiffResults.HessianResult(x)
    cfg = ForwardDiff.HessianConfig(f, result, x, ForwardDiff.Chunk(chunks))

    evaluatef = let f=f, result=result, cfg=cfg
        function (x)
            r = ForwardDiff.hessian!(result, f, x, cfg)
            fx = DiffResults.value(r)
            dfx = DiffResults.gradient(r)
            d2fx = DiffResults.hessian(r)
            return r, fx, dfx, d2fx
        end
    end
    result, fold, grad, hess = evaluatef(x)
    H = similar(hess)

    state = (;
        objective=fold,
        gradient=norm(grad),
        trust_region_r=Inf,
        residual=Inf,
        lipschitz_L=Inf,
    )
    callback((-1, state), nothing, nothing)

    for iter in 1:maxiter
        # Find a valid positive definite approximation to the Hessian.
        lambda = eigmin(Symmetric(hess))
        if lambda <= 0
            lambda -= 1e-6
        elseif lambda > 0
            lambda = zero(lambda)
        end    
        H .= Symmetric(hess) - lambda*I

        # Solve for Newton increment.
        cholH = cholesky!(H)
        ldiv!(v, cholH, -grad)
        
        # Update and evaluate loss.
        fnew = Inf
        for step in 0:nhalf
            s = 2.0 ^ -step
            axpy!(s, v, x)
            result, fnew, grad, hess = evaluatef(x) # re-alias
            if fnew < fold
                if step > 0 && callback isa VerboseCallback
                    println("\t\t $(step) step-halving operation(s)")
                end
                break
            else
                if step == nhalf && callback isa VerboseCallback
                    println("\t\t $(step) step-halving operation(s)")
                else
                    axpy!(-s, v, x)
                end
            end
        end

        state = (;
            objective=fnew,
            gradient=norm(grad),
            trust_region_r=Inf,
            residual=norm(x - xold),
            lipschitz_L=Inf,
        )
        callback((iter, state), nothing, nothing)

        # Check for descent and convergence.
        if fold < fnew
            @warn "Descent condition not satisfied at iteration $(iter)." new=fnew old=fold diff=fold-fnew
        elseif fold == fnew
            @warn "Reached a no-progress point." new=fnew old=fold diff=fold-fnew
            break
        elseif state.gradient < gtol
            break
        end
        fold = fnew
        xold .= x
    end
    return x
end

include("constrained_least_squares.jl")
include("graph_learning.jl")

end # module
