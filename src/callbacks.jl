# define valid fields and helper functions for storing convergence data
const VALID_FIELDS = [
    :rho, :k,               # hyperparameters: lambda, rho, number of active variables
    :iters,                 # total number of iterations
    :loss, :objective,      # loss metrics
    :gradient, :distance    # convergence quality metrics
]

# destructuring of complex tuple object returned by evaluate
_get_statistic_(::Any, ::AbstractProblem, hparams, ::Val{:rho}) = hparams.rho
_get_statistic_(::Any, ::AbstractProblem, hparams, ::Val{:k}) = hparams.k

_get_statistic_(statistics, ::AbstractProblem, ::Any, ::Val{:iters}) = first(statistics)
_get_statistic_(statistics, ::AbstractProblem, ::Any, ::Val{:loss}) = last(statistics).loss
_get_statistic_(statistics, ::AbstractProblem, ::Any, ::Val{:objective}) = last(statistics).objective
_get_statistic_(statistics, ::AbstractProblem, ::Any, ::Val{:distance}) = last(statistics).distance
_get_statistic_(statistics, ::AbstractProblem, ::Any, ::Val{:gradient}) = last(statistics).gradient

###
### VerboseCallback
###
"""
VerboseCallback(every::Int)

Prints convergence information after `every` number of inner iterations.
"""
struct VerboseCallback
    every::Int
end

"""
VerboseCallback()

Prints convergene information every inner iteration.
"""
VerboseCallback() = VerboseCallback(1)

function (F::VerboseCallback)((iter, state), problem, hyperparams)
    if iter == -1
        @printf("\n%-5s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s\n", "iter", "rho", "loss", "objective", "|gradient|", "distance")
        iter = 0
    end
    if iter % F.every == 0
        @printf("%4d\t%4.3e\t%4.3e\t%4.3e\t%4.3e\t%4.3e\n", iter, hyperparams.rho, state.loss, state.objective, state.gradient, state.distance)
    end
    return nothing
end

function (F::VerboseCallback)((iter, state), ::Nothing, ::Nothing)
    if iter == -1
        @printf(
            "\n%-5s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s\n",
            "iter",
            "objective",
            "|gradient|",
            "rₖ",
            "|xₖ-xₖ₋₁|",
            "Lₖ",
        )
        iter = 0
    end
    if iter % F.every == 0
        @printf(
            "%4d\t%4.3e\t%4.3e\t%4.3e\t%4.3e\t%4.3e\n",
            iter,
            state.objective,
            state.gradient,
            state.trust_region_r,
            state.residual,
            state.lipschitz_L,
        )
    end
    return nothing
end

###
### HistoryCallback
###
"""
HistoryCallback(every::Int)

Record convergence information `every` inner iterations.

Specify which metrics should be recorded by calling `add_field!(callback, fields...)`.
"""
struct HistoryCallback{T}
    data::Dict{Symbol,Vector{T}}
    every::Int

    function HistoryCallback{T}(every::Int) where T
        data = Dict{Symbol,Vector{T}}()
        new{T}(data, every)
    end
end

"""
HistoryCallback()

Record convergence information every inner iteration.
"""
HistoryCallback() = HistoryCallback{Float64}(1) # default to Float64 eltype

"""
add_field!(cb::HistoryCallback, field::Symbol)

Add a field to store convergence data for the specified `field`.
"""
function add_field!(cb::HistoryCallback, field::Symbol)
    global VALID_FIELDS
    if !(field in VALID_FIELDS)
        error("Unknown metric $(field).")
    end
    data = cb.data
    T = valtype(data)
    data[field] = T[]
    return data
end

"""
add_field!(cb::HistoryCallback, fields::Vararg{Symbol,N})

Add multiple `fields` to the callback.
"""
function add_field!(cb::HistoryCallback, fields::Vararg{Symbol,N}) where N
    for field in fields
        add_field!(cb, field)
    end
    return cb.data
end

function (F::HistoryCallback)((iter, state), problem::AbstractProblem, hyperparams)
    @unpack data, every = F
    if iter == -1
        foreach(empty!, values(data))
        iter = 0
    end
    if iter % every == 0
        for (field, arr) in data
            push!(arr, _get_statistic_((iter, state), problem, hyperparams, Val(field)))
        end
    end
    return nothing
end
