# Per-request pool of large host arrays for the condensed coupled solver.
#
# Large per-frequency host arrays come from, and go back to, a pool keyed by element type and size,
# so the next frequency writes pages it already owns. Filling a fresh 155 MB array costs ~57 ms of
# page faults, a reused one 1.5 ms; SAWMOD allocated ~1.1 GB per frequency (~76k faults, ~0.3 s of
# system time). An array goes back only once nothing references it; the driver empties the pool at
# the start and end of each request (`clear_condensed_host_pool!`).

const _HOST_POOL = Dict{Any,Vector{Any}}()
const _HOST_POOL_LOCK = ReentrantLock()

function _pool_take(::Type{T}, dims::Vararg{Int,N}) where {T,N}
    found = lock(_HOST_POOL_LOCK) do
        list = get(_HOST_POOL, (T, dims), nothing)
        isnothing(list) || isempty(list) ? nothing : pop!(list)
    end
    return isnothing(found) ? Array{T,N}(undef, dims) : found::Array{T,N}
end

function _pool_give!(arrays...)
    lock(_HOST_POOL_LOCK) do
        for array in arrays
            array isa Array || continue
            list = get!(Vector{Any}, _HOST_POOL, (eltype(array), size(array)))
            any(x -> x === array, list) || push!(list, array)
        end
    end
    return nothing
end

"""
    clear_condensed_host_pool!()

Drop every pooled host array, so none outlives the request that allocated it.
"""
clear_condensed_host_pool!() = lock(() -> empty!(_HOST_POOL), _HOST_POOL_LOCK)

# `zeros(T, m, n)` from the pool, zeroed by column on all threads.
function _pool_zeros(::Type{T}, m::Int, n::Int) where {T}
    array = _pool_take(T, m, n)
    Threads.@threads for column in 1:n
        @inbounds for row in 1:m
            array[row, column] = zero(T)
        end
    end
    return array
end

# `T.(source)` into a pooled array, by column on all threads.
function _pool_converted(::Type{T}, source::Matrix) where {T}
    array = _pool_take(T, size(source)...)
    Threads.@threads for column in axes(source, 2)
        @inbounds for row in axes(source, 1)
            array[row, column] = T(source[row, column])
        end
    end
    return array
end
