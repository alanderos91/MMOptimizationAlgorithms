struct AlgOptions{G}
    maxrhov::Int    # maximum number of rho values to test (outer iterations)
    maxiter::Int    # maximum number of inner iterations
    
    dtol::Float64   # control parameter; dist(alpha, S) < dtol
    rtol::Float64   # control parameter; |dist_old - dist_new| < rtol * (1 + dist_old)
    gtol::Float64   # control parameter; |gradient| < gtol
    
    rho_init::Float64   # initial value for rho
    rho_max::Float64    # maximum value for rho
    rhof::G             # function that generates a sequence for rho

    nesterov::Int   # number of iterations before Nesterov acceleration applies

    function AlgOptions(maxrhov, maxiter, dtol, rtol, gtol, rho_init, rho_max, rhof::G, nesterov) where G
        # Validate options.
        error_msg = ""
        has_errors = false
        if maxrhov < 0 || maxiter < 0
            has_errors = true
            error_msg = string(error_msg, "\nMaximum terations must be a nonnegative integer (maxrhov=$(maxrhov), maxiter=$(maxiter)).")
        end
        if dtol < 0 || rtol < 0 || gtol < 0
            has_errors = true
            error_msg = string(error_msg, "\nControl parameters must be nonnegative (dtol=$(dtol), rtol=$(rtol), gtol=$(gtol)).")
        end
        if rho_init < 0 || rho_max < 0 || rho_init > rho_max
            has_errors = true
            error_msg = string(error_msg, "\nCheck rho parameters for negative values (rho_init=$(rho_init), rho_max=$(rho_max), rho_init < rho_max? $(rho_init < rho_max)).")
        end
        if nesterov < 0
            has_errors = true
            error_msg = string(error_msg, "\nNesterov delay should be a nonnegative interger. (nesterov=$(nesterov)).")
        end
        has_errors && error(error_msg)
        
        new{G}(maxrhov, maxiter, dtol, rtol, gtol, rho_init, rho_max, rhof, nesterov)
    end
end

function Base.show(io::IO, options::AlgOptions)
    print(io, "Algorithm Options")
    print(io, "\n")
    print(io, "\n  max. outer iterations:  $(options.maxrhov)")
    print(io, "\n  max. inner iterations:  $(options.maxiter)")
    print(io, "\n  distance tolerance: $(options.dtol)")
    print(io, "\n  relative tolerance: $(options.rtol)")
    print(io, "\n  gradient tolerance: $(options.gtol)")
    print(io, "\n  initial rho:    $(options.rho_init)")
    print(io, "\n  maximum rho:    $(options.rho_max)")
    print(io, "\n  rho sequence:   $(options.rhof)")
    print(io, "\n  Nesterov delay: $(options.nesterov)")

    return nothing
end

function default_options(::Nothing)
    return AlgOptions(
        10^2, 10^2,
        1e-2, 1e-6, 1e-2,
        1.0, 1e8, geometric_progression(1.2),
        10
    )
end

default_options(::AbstractMMAlg) = default_options(nothing)

function set_options(options::AlgOptions;
        maxrhov::Int=options.maxrhov,
        maxiter::Int=options.maxiter,
        dtol::Real=options.dtol,
        rtol::Real=options.rtol,
        gtol::Real=options.gtol,
        rho_init::Real=options.rho_init,
        rho_max::Real=options.rho_max,
        rhof::G=options.rhof,
        nesterov::Int=options.nesterov,
    ) where G
#
    return AlgOptions(maxrhov, maxiter, dtol, rtol, gtol, rho_init, rho_max, rhof, nesterov)
end

set_options(algorithm::AbstractMMAlg; kwargs...) = set_options(default_options(algorithm); kwargs...)
set_options(; kwargs...) = set_options(default_options(nothing); kwargs...)

struct GeometricProression
    multiplier::Float64
end

function (f::GeometricProression)(rho, iter, rho_max)
    convert(typeof(rho), min(rho_max, rho * f.multiplier))
end

"""
Define a geometric progression recursively by the rule
```
    rho_new = min(rho_max, rho * multiplier)
```
The result is guaranteed to have type `typeof(rho)`.
"""
function geometric_progression(multiplier::Real=1.2)
    return GeometricProression(multiplier)
end

"""
Apply acceleration to the current iterate `x` based on the previous iterate `y`
according to Nesterov's method with parameter `r=3` (default).
"""
function nesterov_acceleration!(x::T, y::T, iter::Integer, needs_reset::Bool, r::Int=3) where T <: AbstractArray
    if needs_reset # Reset acceleration scheme
        copyto!(y, x)
        iter = 1
    else # Nesterov acceleration 
        c = (iter - 1) / (iter + r - 1)
        for i in eachindex(x)
            xi, yi = x[i], y[i]
            zi = xi + c * (xi - yi)
            x[i], y[i] = zi, xi
        end
        iter += 1
    end

    return iter
end

#
# Add a safe default in case a problem type has not implemented acceleration.
#
function nesterov_acceleration!(prob::AbstractProblem, ni, nr)
    @warn "Nesterov acceleration not implemented for $(typeof(prob))."
    return nothing
end

# Default: Update rho only
naive_update(prob::AbstractProblem, rho, rho_new, hparams) = rho_new

# x <- x + dx * dρ
function linear_update(prob::AbstractProblem, rho, rho_new, hparams)
    drho = rho_new - rho
    x, dx = rho_sensitivity(prob, prob.extras, rho, hparams)
    axpy!(drho, dx, x)
    return rho_new
end

# x <- x + dx * dη
function exponential_update(prob::AbstractProblem, rho, rho_new, hparams)
    deta = log(rho_new/rho)
    x, dx = rho_sensitivity(prob, prob.extras, rho, hparams)
    axpy!(rho*deta, dx, x)
    return rho_new
end

#
# Forward difference operator
#
# D = [
#      -1   1   0  0
#       0  -1   1  0
#       0   0  -1  1
#     ]
#
"Overwrite `y` with `alpha*D*x + beta*y`, where `D` is the compatible forward difference operator on `x`."
function fd_axpby!(alpha::T, x, beta::T, y) where T <: Real
    if length(y) != length(x)-1
        error("""
        Output $(length(y)) and input $(length(x)) dimensions are incompatible for computing forward differences.
        """)
    end
    for i in eachindex(y)
        y[i] = alpha*(x[i+1] - x[i]) + beta*y[i]
    end
    return y
end

"Overwrite `y` with `alpha*D*x + y`, where `D` is the compatible forward difference operator on `x`."
function fd_axpy!(alpha, x, y)
    T = eltype(y)
    fd_axpby!(T(alpha), x, one(T), y)
end

"Compute `y = D*x` in-place, where `D` is the compatible forward difference operator on `x`."
function fd!(y, x)
    T = eltype(y)
    fill!(y, zero(T))
    fd_axpby!(one(T), x, zero(T), y)
end

#
# Forward difference operator tranpose
#
# Dᵀ = [
#       -1   0   0
#        1  -1   0
#        0   1  -1
#        0   0   1
#      ]
#
"Overwrite `y` with `alpha*Dᵀ*x + beta*y`, where `Dᵀ` is the transpose of the compatible forward difference operator on `x`."
function fdt_axpby!(alpha::T, x, beta::T, y) where T <: Real
    if length(y) != length(x)+1
        error("""
        Output $(length(y)) and input $(length(x)) dimensions are incompatible for computing forward differences.
        """)
    end
    n = length(y)
    y[1] = -alpha*x[1] + beta*y[1]
    n > 2 && fd_axpby!(-alpha, x, beta, view(y, 2:n-1))
    y[end] = alpha*x[end] + beta*y[end]
    return y
end

"Overwrite `y` with `alpha*Dᵀ*x + y`, where `Dᵀ` is the transpose of the compatible forward difference operator on `x`."
function fdt_axpy!(alpha, x, y)
    T = eltype(y)
    fdt_axpby!(T(alpha), x, one(T), y)
end

"Compute `y = Dᵀ*x` in-place, where `Dᵀ` is the transpose of the compatible forward difference operator on `x`."
function fdt!(y, x)
    T = eltype(y)
    fill!(y, zero(T))
    fdt_axpby!(one(T), x, zero(T), y)
end

function simulate_constant_correlation(n, p, k;
    rng::AbstractRNG=Xoshiro(),
    ngroup::Vector{Int}=equal_size_groups(p, k),
    delta::Real=1e-2,
    epsilon::Real=1e-2,
    rho::Vector{T}=0.05*ones(k),
    noisedim::Int=1,
) where T
#
    rhomin = minimum(rho)
    rhomax = maximum(rho)

    rhomin < 0 && error("Need minimum(rho) >= 0")
    rhomax >= 1 && error("Need maximum(rho) < 1")
    delta >= rhomin && error("Need delta < minimum(rho)")
    epsilon >= 1-rhomax && error("Need epsilon < 1 - maximum(rho)")
    noisedim < 1 && error("Noise dimension must be >= 1")
    length(ngroup) != k && error("Group partitions ($(length(ngroup))) must match k ($(k))")

    X = zeros(n, p)
    S = zeros(p, p)
    groups = []
    index = 1
    for i in eachindex(ngroup)
        start = index
        stop = start + ngroup[i] - 1
        push!(groups, start:stop)
        index = stop + 1
    end

    if noisedim == 1
        u = [rand(rng, (-1.0, 1.0)) for _ in 1:p]
    else
        u = [randn(rng, noisedim) for _ in 1:p]
        u .= u ./ norm.(u)
    end
    for (A, groupA) in enumerate(groups), (B, groupB) in enumerate(groups)
        for j in groupB, i in groupA
            if i == j
                S[i,j] = 1
            elseif A == B
                S[i,j] = rho[A] + epsilon*dot(u[i], u[j])
            else
                S[i,j] = delta + epsilon*dot(u[i], u[j])
            end
        end
    end
    randmvn!(rng, X, Symmetric(S))
    return X
end

function equal_size_groups(p, k)
    k > p && error("Need number of groups k <= p.")
    #
    x = zeros(Int, k)
    q, r = divrem(p, k)
    x .= q
    r > 0 && (x[end] += r)
    return x
end

function randmvn!(rng, X, S)
    cholS = cholesky(S)
    z = zeros(size(X, 2))
    for x in eachrow(X)
        randn!(rng, z)
        mul!(x, cholS.L, z)
    end
    return X
end
