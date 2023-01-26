"""
Projection operator onto the affine set {x | Ax = b}.
"""
struct AffineProjection{T,matT,vecT,solverT}
    A::matT
    b::vecT
    AAt::matT
    r::Vector{T}
    y::Vector{T}
    solver::solverT
end

function AffineProjection(LHS::AbstractMatrix, RHS::AbstractVector)
    AAt = LHS*transpose(LHS)
    r = zeros(size(LHS, 1))
    solver = CgSolver(AAt, r)
    y = solver.x

    return AffineProjection(LHS, RHS, AAt, r, y, solver)
end

function (P::AffineProjection)(x)
    # Unpack objects. The vector y is aliased to the solution of the CG step.
    @unpack A, AAt, b, r, y, solver = P
    T = eltype(x)

    # Compute r = Ax - b.
    copyto!(r, b)
    mul!(r, A, x, one(T), -one(T))

    # Solve AAᵀ y = r.
    cg!(solver, AAt, r)

    # Compute x - Aᵀy.
    mul!(x, transpose(A), y, -one(T), one(T))

    return x
end
