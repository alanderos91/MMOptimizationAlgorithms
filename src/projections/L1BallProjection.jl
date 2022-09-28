"""
Data structure for projecting vectors onto an L1 ball.
"""
struct L1BallProjection{T}
    buffer1::Vector{T}
    buffer2::Vector{T}
end

"""
Initialize `L1BallProjection` for vectors with `n` components.
"""
L1BallProjection{T}(n::Int) where T = L1BallProjection(zeros(T, n), zeros(T, n))

function (P::L1BallProjection)(x, r)
    project_l1_ball!(x, r, P.buffer1, P.buffer2)
end

"""
Projects the point `x` onto the L1 ball centered at the origin and with radius `r`.
"""
function project_l1_ball!(x, r, y, buffer)
    #
    map!(abs, y, x)
    if norm(y, 1) > r
        project_simplex!(y, r, buffer)
        for i in eachindex(x)
            x[i] = sign(x[i]) * y[i]
        end
    end
    return x
end
