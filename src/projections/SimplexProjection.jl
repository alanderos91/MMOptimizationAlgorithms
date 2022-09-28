"""
Data structure for projecting vectors onto a simplex."
"""
struct SimplexProjection{T}
    buffer::Vector{T}
end

"""
Initialize `SimplexProjection` for vectors with `n` components.
"""
SimplexProjection{T}(n::Int) where T = SimplexProjection(zeros(T, n))

function (P::SimplexProjection)(x, r)
    project_simplex!(x, r, P.buffer)
end

"""Projects the point y onto the simplex {x | x >= 0, sum(x) = r}."""
function project_simplex!(x, r, z)
    # Compute the Lagrange multiplier, lambda.
    T = eltype(x)
    copyto!(z, x)
    sort!(z, rev=true)
    (s, lambda) = (zero(T), zero(T))
    for i in eachindex(z)
        s = s + z[i]
        lambda = (s - r) / i
        if i < length(z) && lambda < z[i] && lambda >= z[i+1]
            break
        end
    end
    # Threshold the values according to lambda.
    T = eltype(x)
    for i in eachindex(x)
        x[i] = max(zero(T), x[i] - lambda)
    end
    return x
end
