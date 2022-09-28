"""
Data structure for projecting vectors onto unstructured sparsity sets.
"""
struct L0Projection{VT,RNG}
    idx::VT     # stores a permutation of indices [1, 2, ..., n]
    buffer::VT  # used to break ties
    rng::RNG    # used to break ties
end

"""
Initialize `L0Projection` for vectors with `n` components.
"""
function L0Projection(n::Integer, rng::AbstractRNG=Xoshiro())
    idx = collect(1:n)
    return L0Projection(idx, similar(idx), rng)
end

"""
Project `x` onto the set of sparse vectors with at most `k` nonzero components.

**Note**: Ties are broken randomly.
"""
function (P::L0Projection)(x, k)
    project_l0_ball!(P.rng, x, k, P.idx, P.buffer)
end

"""
Project `x` onto sparsity set with `k` non-zero elements.
Assumes `idx` enters as a vector of indices into `x`.
"""
function project_l0_ball!(rng, x, k, idx, buffer)
    #
    is_equal_magnitude(x, y) = abs(x) == abs(y)
    is_equal_magnitude(y) = Base.Fix2(is_equal_magnitude, y)
    #
    n = length(x)
    # Do nothing if k >= length(x), fill with zeros if k ≤ 0.
    if k >= n return x end
    if k <= 0 return fill!(x, 0) end
    
    # Find pivot that determines the threshold. The array `idx` MUST contain [1,2,...,n] or
    # a permutation of the indices.
    #
    # Based on https://github.com/JuliaLang/julia/blob/788b2c77c10c2160f4794a4d4b6b81a95a90940c/base/sort.jl#L863
    # This eliminates a mysterious allocation of ~48 bytes per call for
    #   sortperm!(idx, x, alg=algorithm, lt=isless, by=abs, rev=true, initialized=false)
    # where algorithm = PartialQuickSort(lo:hi)
    # Savings are small in terms of performance but add up for CV code.
    #
    (lo, hi) = (k, k+1)
    (lt, by, rev, o) = (isless, abs, true, Base.Order.Forward)
    order = Base.Order.Perm(Base.Sort.ord(lt, by, rev, o), x)
    Base.Sort.Float.fpsort!(idx, PartialQuickSort(lo:hi), order)
    pivot = x[idx[k]]

    # Preserve the top k elements. Keep track of the number of nonzero components.
    nonzero_count = 0
    @inbounds for i in eachindex(x)
        if x[i] == 0 continue end
        if abs(x[i]) < abs(pivot)
            x[i] = 0
        else
            nonzero_count += 1
        end
    end

    # Resolve ties with randomization.
    if nonzero_count > k
        number_to_drop = nonzero_count - k
        idx_duplicates = findall(is_equal_magnitude(pivot), x)
        idx_threshold = view(buffer, 1:number_to_drop)
        StatsBase.sample!(rng, idx_duplicates, idx_threshold, replace=false)
        @inbounds for i in idx_threshold
            x[i] = 0
        end
    end

    return x
end
