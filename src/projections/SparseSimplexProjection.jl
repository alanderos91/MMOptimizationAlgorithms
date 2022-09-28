struct SparseSimplexProjection{RNG,T}
    idx::Vector{Int}
    idx_buffer::Vector{Int}
    data_buffer::Vector{T}
    rng::RNG
end

function SparseSimplexProjection{T}(n::Integer, rng::RNG=Xoshiro()) where {RNG,T}
    idx = collect(1:n)
    idx_buffer = similar(idx)
    data_buffer = zeros(T, n)
    return SparseSimplexProjection{RNG,T}(idx, idx_buffer, data_buffer, rng)
end

function (P::SparseSimplexProjection)(x, k, r)
    buffers = (P.idx, P.idx_buffer, P.data_buffer)
    project_sparse_simplex!(P.rng, x, k, r, buffers)
end

# use signs in projection
function project_l0_ball_no_abs!(rng, x, k, idx, buffer)
    #
    n = length(x)
    # Do nothing if k >= length(x), fill with zeros if k ≤ 0.
    if k >= n return x end
    if k <= 0 return fill!(x, 0) end
    
    # Find pivot that determines the threshold. The array `idx` MUST contain [1,2,...,n] or
    # a permutation of the indices.
    #
    # Based on https://github.com/JuliaLang/julia/blob/788b2c77c10c2160f4794a4d4b6b81a95a90940c/base/sort.jl#L863
    (lo, hi) = (k, k+1)
    (lt, by, rev, o) = (isless, identity, true, Base.Order.Forward)
    order = Base.Order.Perm(Base.Sort.ord(lt, by, rev, o), x)
    Base.Sort.Float.fpsort!(idx, PartialQuickSort(lo:hi), order)
    pivot = x[idx[k]]

    # Preserve the top k elements. Keep track of the number of nonzero components.
    nonzero_count = 0
    @inbounds for i in eachindex(x)
        if x[i] == 0 continue end
        if x[i] < pivot
            x[i] = 0
        else
            nonzero_count += 1
        end
    end

    # Resolve ties with randomization.
    if nonzero_count > k
        println("Ties detected!!!")
        number_to_drop = nonzero_count - k
        idx_duplicates = findall(isequal(pivot), x)
        idx_threshold = view(buffer, 1:number_to_drop)
        StatsBase.sample!(rng, idx_duplicates, idx_threshold, replace=false)
        @inbounds for i in idx_threshold
            x[i] = 0
        end
    end

    return x
end

function project_sparse_simplex!(rng, x, k, r, buffers)
    idx, idx_buffer, data_buffer = buffers
    project_l0_ball_no_abs!(rng, x, k, idx, idx_buffer) # x ∈ Rᵖ -> x ∈ Sₖ
    # Find the nonzero components in x.
    count = 0
    for i in eachindex(x)
        if count >= k break end
        if x[i] != 0
            count += 1
            idx_buffer[count] = i
        end
    end
    idx_nonzero = view(idx_buffer, 1:count)
    x_nonzero = view(x, idx_nonzero)
    _data_buffer = view(data_buffer, 1:count)
    project_simplex!(x_nonzero, r, _data_buffer) # x ∈ Sₖ -> x ∈ Sₖ ∩ Δₖ₋₁
    return x
end
