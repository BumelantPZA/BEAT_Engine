# Excerpt of the code relevant to the far-field Metal assembly kernel (BEAT Engine v0.2.0).
# Each part is copied verbatim; the header shows the source file and line range.
# Not compiled or included anywhere; it's here only for reading.

# ===== BeatEngineMetalCommon.jl lines 91-120 =====
struct MetalRegularAssemblyCache{T,C}
    host_cache::C
    face_vertices
    normals
    areas
    faces
    curls
    rule_points
    rule_weights
    element_rule_points   # face_count x rule_count x 3: every element's regular quadrature points
    vertex_offsets
    incident_elements
    incident_local_indices
    dp0_elements
    p1_dofs
    element_dp0_dofs
    color_elements
    color_offsets::Vector{Int}
    element_indices::Vector{Int}
    face_count::Int
    p1_dof_count::Int
    dp0_dof_count::Int
    rule_count::Int
    symmetry_mode::Symbol
    image_transforms::Vector{SymmetryTransform}
    image_singular_caches::Vector{MetalSingularCorrectionCache{T}}
    image_singular_pair_count::Int
    gather_tables::Ref{Any}   # MetalGatherTables, built lazily by the pair_gather kernel mode
    fused_gather_tables::Ref{Any}   # MetalFusedGatherTables, built lazily by the fused Burton-Miller path
end

# ===== BeatEngineMetalCommon.jl lines 206-210 =====
function _metal_launch(kernel, count::Integer, args...; groupsize::Integer=_metal_kernel_groupsize())
    count <= 0 && return nothing
    Metal.@metal threads=groupsize groups=cld(count, groupsize) kernel(args...)
    return nothing
end

# ===== BeatEngineMetalPairKernels.jl lines 19-32 =====
@inline function _metal_pair_is_skipped(
    faces,
    face_count,
    test_index,
    trial_index,
    pair_offsets,
    singular_trial_indices,
    skip_mode,
)
    return skip_mode == 0 ?
        _metal_faces_are_adjacent(faces, test_index, trial_index, face_count) :
        skip_mode == 1 &&
            _metal_find_pair(pair_offsets, singular_trial_indices, test_index, trial_index) != 0
end

# ===== BeatEngineMetalAtomicKernels.jl lines 1-229 =====
# Fused pair-atomic regular kernel: one thread per (test, trial) element pair
# on a 2-D thread grid, every Green's-function value evaluated once and used
# for all four operators, results scattered with Float32 atomics.
#
# This is the design hornlab-metal-bem measured as ~5x faster than its
# alternatives on Apple GPUs. Three things distinguish it from the colored
# pair-owned kernels: one dispatch instead of color_count^2, no second pass
# for the hypersingular operator, and no 64-bit integer division on the GPU
# (the pair is addressed by the 2-D grid position; the six trial quadrature
# points are hoisted out of the 36-point-pair loop). It trades determinism
# for that: results differ from the colored kernels by float32 summation
# order only.

using Metal: atomic_fetch_add_explicit, thread_position_in_grid_2d

@inline _metal_fast_cos(x::Float32) = Base.FastMath.cos_fast(x)
@inline _metal_fast_sin(x::Float32) = Base.FastMath.sin_fast(x)
@inline _metal_fast_rsqrt(x::Float32) = Metal.rsqrt_fast(x)

# Compile-time unrolled fold over the trial quadrature points:
# acc = term(...term(term(acc, 1), 2)..., N). The trial point, basis and
# weight are read from the (cached) rule and vertex arrays inside the term
# with constant indices instead of being hoisted into registers: this kernel
# is register-bound (the 3x3 double-layer and hypersingular accumulators
# alone are 36 floats), and hoisted trial data cost another ~42.
@inline _metal_trial_fold(acc, context, ::Val{0}, ::Val{R}) where {R} = acc
@inline function _metal_trial_fold(acc, context, ::Val{N}, ::Val{R}) where {N,R}
    acc = _metal_trial_fold(acc, context, Val(N - 1), Val(R))
    return _metal_trial_term(acc, context, Int32(N), Val(R))
end

@inline function _metal_trial_term(acc, context, trial_q::Int32, ::Val{R}) where {R}
    s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = acc
    x, y, z, test_weight_scale, k, inv_four_pi,
        test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        element_rule_points, rule_points, rule_weights, trial_index, face_count = context
    @inbounds xi = rule_points[trial_q]
    @inbounds eta = rule_points[trial_q + Int32(R)]
    rb1 = one(k) - xi - eta
    @inbounds trial_weight = rule_weights[trial_q]
    point_index = trial_index + face_count * (trial_q - Int32(1))
    @inbounds sx = element_rule_points[point_index]
    @inbounds sy = element_rule_points[point_index + face_count * Int32(R)]
    @inbounds sz = element_rule_points[point_index + face_count * Int32(2 * R)]
    Base.@fastmath begin
    dx = sx * trial_signs[1] - x
    dy = sy * trial_signs[2] - y
    dz = sz * trial_signs[3] - z
    radius2 = dx * dx + dy * dy + dz * dz
    if radius2 > zero(k)
        rb = SVector(rb1, xi, eta)
        # Fast-math intrinsics: this is what an Xcode-compiled Metal
        # shader gets by default, and what hornlab-metal-bem runs.
        inv_radius = _metal_fast_rsqrt(radius2)
        radius = radius2 * inv_radius
        phase = k * radius
        green_scale = inv_radius * inv_four_pi * (test_weight_scale * trial_weight)
        green_re = _metal_fast_cos(phase) * green_scale
        green_im = _metal_fast_sin(phase) * green_scale
        grad_re = -green_re * inv_radius - green_im * k
        grad_im = green_re * k - green_im * inv_radius
        test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
        trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
        s_re += green_re
        s_im += green_im
        a_re += grad_re * test_dot
        a_im += grad_im * test_dot
        d_re += rb * (grad_re * trial_dot)
        d_im += rb * (grad_im * trial_dot)
        h_re += rb * green_re
        h_im += rb * green_im
    end
    end
    return (s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im)
end

@inline function _metal_atomic_add_complex!(target_f32, complex_index, re, im)
    base = 2 * complex_index - 1
    atomic_fetch_add_explicit(pointer(target_f32, base), re)
    atomic_fetch_add_explicit(pointer(target_f32, base + 1), im)
    return nothing
end

# The pair arithmetic shared by the fused atomic kernel and the chunked
# gather kernel: every Green's-function value evaluated once for the four
# operators, accumulated in the rank-1 form. Returns the 3x1 single-layer and
# adjoint blocks and the 3x3 double-layer and hypersingular blocks
# (column-major, real and imaginary parts separately).
@inline function _metal_regular_pair_blocks(
    face_vertices,
    normals,
    areas,
    curls,
    rule_points,
    rule_weights,
    element_rule_points,
    test_index::Int32,
    trial_index::Int32,
    face_count::Int32,
    k,
    ::Val{R},
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
) where {R}
    @inbounds return _metal_regular_pair_blocks_inbounds(
        face_vertices, normals, areas, curls, rule_points, rule_weights, element_rule_points,
        test_index, trial_index, face_count, k, Val(R),
        trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    )
end

@inline function _metal_regular_pair_blocks_inbounds(
    face_vertices, normals, areas, curls, rule_points, rule_weights, element_rule_points,
    test_index::Int32, trial_index::Int32, face_count::Int32, k, ::Val{R},
    trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
) where {R}
    T = typeof(k)
    inv_four_pi = T(0.07957747154594767)
    k2 = k * k
    slp_re = zero(SVector{3,T})
    slp_im = zero(SVector{3,T})
    adj_re = zero(SVector{3,T})
    adj_im = zero(SVector{3,T})
    dlp_re = zero(SVector{9,T})
    dlp_im = zero(SVector{9,T})
    hyp_re = zero(SVector{9,T})
    hyp_im = zero(SVector{9,T})
    test_nx = normals[test_index]
    test_ny = normals[test_index + face_count]
    test_nz = normals[test_index + Int32(2) * face_count]
    trial_nx = trial_sign_x * normals[trial_index]
    trial_ny = trial_sign_y * normals[trial_index + face_count]
    trial_nz = trial_sign_z * normals[trial_index + Int32(2) * face_count]
    normal_product = test_nx * trial_nx + test_ny * trial_ny + test_nz * trial_nz
    jac_scale = T(4) * areas[test_index] * areas[trial_index]
    trial_signs = SVector(trial_sign_x, trial_sign_y, trial_sign_z)

    # Rank-1 structure: every 3x3 block a pair contributes is
    # sum_a sum_b (test basis at a) x (trial basis at b) * scalar(a, b), so the
    # inner loop over b only needs 3-vector (and scalar) accumulators, and the
    # outer products are applied once per test point. That is 16 FMAs per
    # point pair instead of 48, and the hypersingular curl term needs only the
    # summed Green's function.
    g_total_re = zero(k)
    g_total_im = zero(k)
    test_q = Int32(1)
    while test_q <= Int32(R)
        test_xi = rule_points[test_q]
        test_eta = rule_points[test_q + Int32(R)]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        test_basis = SVector(tb1, tb2, tb3)
        point_index = test_index + face_count * (test_q - Int32(1))
        x = element_rule_points[point_index]
        y = element_rule_points[point_index + face_count * Int32(R)]
        z = element_rule_points[point_index + face_count * Int32(2 * R)]
        test_weight = rule_weights[test_q]

        s_re = zero(k)          # sum_b g w                 -> single layer, G0
        s_im = zero(k)
        a_re = zero(k)          # sum_b grad*test_dot w     -> adjoint double layer
        a_im = zero(k)
        d_re = zero(SVector{3,T})  # sum_b rb grad*trial_dot w -> double layer
        d_im = zero(SVector{3,T})
        h_re = zero(SVector{3,T})  # sum_b rb g w              -> hypersingular basis term
        h_im = zero(SVector{3,T})
        # The trial loop is unrolled at compile time (Val recursion) so the
        # hoisted trial points, basis values and weights are indexed by
        # constants and stay in registers; a runtime-indexed tuple in a while
        # loop is materialised in thread-private memory instead. No closure:
        # a captured-and-reassigned variable would be boxed.
        context = (x, y, z, test_weight * jac_scale, k, inv_four_pi,
            test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
            element_rule_points, rule_points, rule_weights, trial_index, face_count)
        s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = _metal_trial_fold(
            (s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im),
            context, Val(R), Val(R),
        )
        slp_re += test_basis * s_re
        slp_im += test_basis * s_im
        adj_re += test_basis * a_re
        adj_im += test_basis * a_im
        g_total_re += s_re
        g_total_im += s_im
        # Outer products, column-major: entry (row i, col j) at index i + 3 (j - 1).
        dlp_re += SVector(
            tb1 * d_re[1], tb2 * d_re[1], tb3 * d_re[1],
            tb1 * d_re[2], tb2 * d_re[2], tb3 * d_re[2],
            tb1 * d_re[3], tb2 * d_re[3], tb3 * d_re[3],
        )
        dlp_im += SVector(
            tb1 * d_im[1], tb2 * d_im[1], tb3 * d_im[1],
            tb1 * d_im[2], tb2 * d_im[2], tb3 * d_im[2],
            tb1 * d_im[3], tb2 * d_im[3], tb3 * d_im[3],
        )
        hyp_re += SVector(
            tb1 * h_re[1], tb2 * h_re[1], tb3 * h_re[1],
            tb1 * h_re[2], tb2 * h_re[2], tb3 * h_re[2],
            tb1 * h_re[3], tb2 * h_re[3], tb3 * h_re[3],
        )
        hyp_im += SVector(
            tb1 * h_im[1], tb2 * h_im[1], tb3 * h_im[1],
            tb1 * h_im[2], tb2 * h_im[2], tb3 * h_im[2],
            tb1 * h_im[3], tb2 * h_im[3], tb3 * h_im[3],
        )
        test_q += Int32(1)
    end
    # hyp so far holds the basis-weighted Green's sums; the hypersingular
    # block is curl_products * G0 - k^2 * (n.n') * (basis-weighted sums).
    k2n = k2 * normal_product
    # Computed after the loop so its nine values are not live registers during it.
    curl_products = _metal_pair_curl_products(
        curls,
        test_index,
        trial_index,
        face_count,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
    )
    hyp_re = curl_products * g_total_re - hyp_re * k2n
    hyp_im = curl_products * g_total_im - hyp_im * k2n
    return slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im
end

# ===== BeatEngineMetalGatherKernels.jl lines 1-479 =====
# Chunked pair-gather regular kernel: zero atomics.
#
# The fused pair-atomic kernel is bound by atomic throughput, not arithmetic:
# every (test, trial) element pair scatters 48 Float32 atomics (3x1 single
# layer and adjoint blocks, 3x3 double layer and hypersingular blocks, real
# and imaginary), and on an Apple GPU that is ~10x the cost of evaluating the
# Green's function for the pair. This mode replaces the scatter with a plain
# store and a gather:
#
#   1. The trial elements are processed in chunks of `chunk_size` columns.
#      A 2-D pair kernel writes each pair's 48 block values to a device buffer
#      laid out [test position, trial local, component] so that a tile of
#      consecutive test positions writes consecutive addresses.
#   2. A gather kernel per operator entry sums the buffer: one thread per
#      (P1 row, trial element) for the single-layer and adjoint operators, one
#      per (P1 row, chunk node) for the double-layer and hypersingular
#      operators, walking the row's incident test elements and the node's
#      incident chunk elements. Each entry has exactly one owner per launch,
#      so the accumulation is a plain read-modify-write.
#
# Summation order is fixed, so unlike the atomic kernel this mode is
# bit-reproducible run to run. Memory traffic is 192 bytes written and read
# per pair; the chunk size is chosen from BLAB_METAL_GATHER_BUDGET_MB (512).

using Metal: thread_position_in_grid_2d

const _metal_gather_stage_timing = Dict{String,Float64}()

struct MetalGatherTables
    chunk_size::Int
    chunk_count::Int
    elements              # MtlArray{Int32}: assembly position -> global element
    element_positions     # MtlArray{Int32}: global element -> assembly position (0 if absent)
    chunk_node_offsets::Vector{Int}   # host: chunk -> first position in chunk_nodes
    chunk_nodes           # MtlArray{Int32}: chunk-node position -> global P1 dof
    inc_offsets           # MtlArray{Int32}: chunk-node position -> first entry in inc_packed
    inc_packed            # MtlArray{Int32}: (trial local - 1) * 4 + local column
    blocks                # MtlArray{Float32}: 48 * element_count * chunk_size
end

const _METAL_GATHER_COMPONENTS = 48

function _metal_gather_chunk_size(element_count::Int)
    element_count <= 0 && return 1
    budget_mb = parse(Float64, get(ENV, "BLAB_METAL_GATHER_BUDGET_MB", "512"))
    per_column_bytes = element_count * _METAL_GATHER_COMPONENTS * sizeof(Float32)
    chunk = clamp(floor(Int, budget_mb * 1e6 / per_column_bytes), 1, element_count)
    override = strip(get(ENV, "BLAB_METAL_GATHER_CHUNK", ""))
    isempty(override) || (chunk = clamp(parse(Int, override), 1, element_count))
    # Buffer indices are Int32 on the device.
    while element_count * chunk * _METAL_GATHER_COMPONENTS >= typemax(Int32) && chunk > 1
        chunk = chunk ÷ 2
    end
    return chunk
end

function _metal_gather_chunk_count(cache::MetalRegularAssemblyCache)
    element_count = length(cache.element_indices)
    element_count == 0 && return 0
    tables = cache.gather_tables[]
    tables === nothing && return cld(element_count, _metal_gather_chunk_size(element_count))
    return tables.chunk_count
end

function _metal_gather_tables(cache::MetalRegularAssemblyCache)
    tables = cache.gather_tables[]
    tables === nothing || return tables
    indices = cache.element_indices
    element_count = length(indices)
    chunk_size = _metal_gather_chunk_size(element_count)
    chunk_count = cld(element_count, chunk_size)
    p1_dofs = Array(cache.p1_dofs)   # face_count x 3
    element_positions = zeros(Int32, cache.face_count)
    for (position, element_index) in enumerate(indices)
        element_positions[element_index] = Int32(position)
    end
    chunk_node_offsets = Vector{Int}(undef, chunk_count + 1)
    chunk_node_offsets[1] = 1
    chunk_nodes = Int32[]
    inc_offsets = Int32[1]
    inc_packed = Int32[]
    for chunk in 1:chunk_count
        start = (chunk - 1) * chunk_size + 1
        stop = min(chunk * chunk_size, element_count)
        node_incidence = Dict{Int32,Vector{Int32}}()
        for position in start:stop
            element_index = indices[position]
            trial_local = position - start + 1
            for local_column in 1:3
                node = p1_dofs[element_index, local_column]
                push!(get!(node_incidence, node, Int32[]), Int32((trial_local - 1) * 4 + local_column))
            end
        end
        for node in sort!(collect(keys(node_incidence)))
            push!(chunk_nodes, node)
            append!(inc_packed, node_incidence[node])
            push!(inc_offsets, Int32(length(inc_packed) + 1))
        end
        chunk_node_offsets[chunk + 1] = length(chunk_nodes) + 1
    end
    blocks = MtlArray{Float32}(undef, _METAL_GATHER_COMPONENTS * element_count * chunk_size)
    tables = MetalGatherTables(
        chunk_size,
        chunk_count,
        MtlArray(Int32.(indices)),
        MtlArray(element_positions),
        chunk_node_offsets,
        MtlArray(chunk_nodes),
        MtlArray(inc_offsets),
        MtlArray(inc_packed),
        blocks,
    )
    cache.gather_tables[] = tables
    return tables
end

function _release_metal_gather_tables!(cache::MetalRegularAssemblyCache)
    tables = cache.gather_tables[]
    tables === nothing && return nothing
    Metal.unsafe_free!(tables.elements)
    Metal.unsafe_free!(tables.element_positions)
    Metal.unsafe_free!(tables.chunk_nodes)
    Metal.unsafe_free!(tables.inc_offsets)
    Metal.unsafe_free!(tables.inc_packed)
    Metal.unsafe_free!(tables.blocks)
    cache.gather_tables[] = nothing
    return nothing
end

@inline function _metal_store_block!(blocks, base::Int32, stride::Int32, offset::Int32, values::SVector{N,T}) where {N,T}
    i = 1
    while i <= N
        @inbounds blocks[base + (offset + Int32(i - 1)) * stride] = values[i]
        i += 1
    end
    return nothing
end

# Component layout per pair: 0-2 S re, 3-5 S im, 6-8 K' re, 9-11 K' im,
# 12-20 D re, 21-29 D im, 30-38 H re, 39-47 H im (3x3 blocks column-major).
function _metal_regular_pair_blocks_kernel!(
    blocks,
    face_vertices,
    normals,
    areas,
    faces,
    curls,
    rule_points,
    rule_weights,
    element_rule_points,
    elements,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    pair_stride::Int32,
    k,
    face_count::Int32,
    ::Val{R},
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
) where {R}
    position = thread_position_in_grid_2d()
    test_position = Int32(position.x)
    trial_local = Int32(position.y)
    (test_position > element_count || trial_local > chunk_count) && return nothing
    @inbounds test_index = Int32(elements[test_position])
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    base = test_position + element_count * (trial_local - Int32(1))
    if _metal_pair_is_skipped(
        faces,
        face_count,
        test_index,
        trial_index,
        pair_offsets,
        singular_trial_indices,
        skip_mode,
    )
        component = Int32(0)
        while component < Int32(_METAL_GATHER_COMPONENTS)
            @inbounds blocks[base + component * pair_stride] = zero(eltype(blocks))
            component += Int32(1)
        end
        return nothing
    end
    slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im = _metal_regular_pair_blocks(
        face_vertices,
        normals,
        areas,
        curls,
        rule_points,
        rule_weights,
        element_rule_points,
        test_index,
        trial_index,
        face_count,
        k,
        Val(R),
        trial_sign_x,
        trial_sign_y,
        trial_sign_z,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
    )
    _metal_store_block!(blocks, base, pair_stride, Int32(0), slp_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(3), slp_im)
    _metal_store_block!(blocks, base, pair_stride, Int32(6), adj_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(9), adj_im)
    _metal_store_block!(blocks, base, pair_stride, Int32(12), dlp_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(21), dlp_im)
    _metal_store_block!(blocks, base, pair_stride, Int32(30), hyp_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(39), hyp_im)
    return nothing
end

# One thread per (P1 row, trial element of the chunk): sums the 3x1 blocks of
# the row's incident test elements into S[row, dp0(trial)] and K'[row, dp0(trial)].
function _metal_gather_slp_adjoint_kernel!(
    single_layer,
    adjoint_double_layer,
    blocks,
    elements,
    element_positions,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    element_dp0_dofs,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    pair_stride::Int32,
    p1_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * chunk_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    trial_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    column_base = element_count * (trial_local - Int32(1))
    s_re = zero(eltype(blocks))
    s_im = zero(eltype(blocks))
    a_re = zero(eltype(blocks))
    a_im = zero(eltype(blocks))
    @inbounds incident_position = Int32(vertex_offsets[row])
    @inbounds incident_stop = Int32(vertex_offsets[row + Int32(1)]) - Int32(1)
    while incident_position <= incident_stop
        @inbounds test_position = Int32(element_positions[incident_elements[incident_position]])
        @inbounds local_row = Int32(incident_local_indices[incident_position])
        pair = test_position + column_base
        @inbounds s_re += blocks[pair + (local_row - Int32(1)) * pair_stride]
        @inbounds s_im += blocks[pair + (local_row + Int32(2)) * pair_stride]
        @inbounds a_re += blocks[pair + (local_row + Int32(5)) * pair_stride]
        @inbounds a_im += blocks[pair + (local_row + Int32(8)) * pair_stride]
        incident_position += Int32(1)
    end
    @inbounds dp0_column = Int32(element_dp0_dofs[trial_index])
    operator_index = row + (dp0_column - Int32(1)) * p1_count
    @inbounds single_layer[operator_index] += Complex(s_re, s_im)
    @inbounds adjoint_double_layer[operator_index] += Complex(a_re, a_im)
    return nothing
end

# One thread per (P1 row, P1 node touched by the chunk): sums the 3x3 block
# entries of every (incident test element, incident chunk element) pair into
# D[row, node] and H[row, node].
function _metal_gather_dlp_hyp_kernel!(
    double_layer,
    hypersingular,
    blocks,
    element_positions,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    chunk_nodes,
    inc_offsets,
    inc_packed,
    node_start::Int32,
    node_count::Int32,
    element_count::Int32,
    pair_stride::Int32,
    p1_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * node_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    node_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    node_position = node_start + node_local - Int32(1)
    @inbounds column = Int32(chunk_nodes[node_position])
    @inbounds chunk_first = Int32(inc_offsets[node_position])
    @inbounds chunk_stop = Int32(inc_offsets[node_position + Int32(1)]) - Int32(1)
    d_re = zero(eltype(blocks))
    d_im = zero(eltype(blocks))
    h_re = zero(eltype(blocks))
    h_im = zero(eltype(blocks))
    @inbounds incident_position = Int32(vertex_offsets[row])
    @inbounds incident_stop = Int32(vertex_offsets[row + Int32(1)]) - Int32(1)
    while incident_position <= incident_stop
        @inbounds test_position = Int32(element_positions[incident_elements[incident_position]])
        @inbounds local_row = Int32(incident_local_indices[incident_position])
        chunk_position = chunk_first
        while chunk_position <= chunk_stop
            @inbounds packed = Int32(inc_packed[chunk_position])
            trial_local = (packed >> 2) + Int32(1)
            local_column = packed & Int32(3)
            pair = test_position + element_count * (trial_local - Int32(1))
            component = local_row + Int32(3) * (local_column - Int32(1))   # 1..9
            @inbounds d_re += blocks[pair + (component + Int32(11)) * pair_stride]
            @inbounds d_im += blocks[pair + (component + Int32(20)) * pair_stride]
            @inbounds h_re += blocks[pair + (component + Int32(29)) * pair_stride]
            @inbounds h_im += blocks[pair + (component + Int32(38)) * pair_stride]
            chunk_position += Int32(1)
        end
        incident_position += Int32(1)
    end
    operator_index = row + (column - Int32(1)) * p1_count
    @inbounds double_layer[operator_index] += Complex(d_re, d_im)
    @inbounds hypersingular[operator_index] += Complex(h_re, h_im)
    return nothing
end

@inline function _metal_gather_stage!(name::String, timed::Bool, start::Float64)
    timed || return start
    Metal.synchronize()
    now = time()
    _metal_gather_stage_timing[name] = get(_metal_gather_stage_timing, name, 0.0) + (now - start)
    return now
end

function _launch_metal_gather_pair_kernels!(
    operators,
    cache::MetalRegularAssemblyCache,
    k,
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    element_count = length(cache.element_indices)
    element_count == 0 && return nothing
    rule_count = cache.rule_count
    rule_count in (1, 3, 6) || error("Metal gather assembly expects a 1-, 3-, or 6-point triangle rule; got $(rule_count).")
    tables = _metal_gather_tables(cache)
    chunk_size = tables.chunk_size
    pair_stride = Int32(element_count * chunk_size)
    tile_x, tile_y = _metal_atomic_tile()
    groupsize = _metal_kernel_groupsize()
    p1_count = Int32(cache.p1_dof_count)
    timed = get(ENV, "BLAB_METAL_GATHER_TIMING", "0") == "1"
    timed && Metal.synchronize()
    stamp = time()
    for chunk in 1:tables.chunk_count
        chunk_start = (chunk - 1) * chunk_size + 1
        chunk_count = min(chunk_size, element_count - chunk_start + 1)
        Metal.@metal threads=(tile_x, tile_y) groups=(cld(element_count, tile_x), cld(chunk_count, tile_y)) _metal_regular_pair_blocks_kernel!(
            tables.blocks,
            cache.face_vertices,
            cache.normals,
            cache.areas,
            cache.faces,
            cache.curls,
            cache.rule_points,
            cache.rule_weights,
            cache.element_rule_points,
            tables.elements,
            Int32(element_count),
            Int32(chunk_start),
            Int32(chunk_count),
            pair_stride,
            k,
            Int32(cache.face_count),
            Val(rule_count),
            pair_offsets,
            singular_trial_indices,
            skip_mode,
            trial_sign_x,
            trial_sign_y,
            trial_sign_z,
            trial_curl_sign_x,
            trial_curl_sign_y,
            trial_curl_sign_z,
        )
        stamp = _metal_gather_stage!("pairs", timed, stamp)
        _metal_launch(
            _metal_gather_slp_adjoint_kernel!,
            cache.p1_dof_count * chunk_count,
            operators.single_layer,
            operators.adjoint_double_layer,
            tables.blocks,
            tables.elements,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            cache.element_dp0_dofs,
            Int32(element_count),
            Int32(chunk_start),
            Int32(chunk_count),
            pair_stride,
            p1_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("slp_adjoint", timed, stamp)
        node_start = tables.chunk_node_offsets[chunk]
        node_count = tables.chunk_node_offsets[chunk + 1] - node_start
        _metal_launch(
            _metal_gather_dlp_hyp_kernel!,
            cache.p1_dof_count * node_count,
            operators.double_layer,
            operators.hypersingular,
            tables.blocks,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            tables.chunk_nodes,
            tables.inc_offsets,
            tables.inc_packed,
            Int32(node_start),
            Int32(node_count),
            Int32(element_count),
            pair_stride,
            p1_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("dlp_hyp", timed, stamp)
    end
    return nothing
end

function _launch_metal_regular_gather_kernels!(operators, cache::MetalRegularAssemblyCache, k)
    return _launch_metal_gather_pair_kernels!(
        operators,
        cache,
        k,
        cache.vertex_offsets,
        cache.incident_elements,
        Int32(0),
        one(k), one(k), one(k),
        one(k), one(k), one(k),
    )
end

function _launch_metal_symmetry_regular_gather_kernels!(
    operators,
    cache::MetalRegularAssemblyCache,
    image_cache::MetalSingularCorrectionCache,
    transform::SymmetryTransform,
    k;
    skip_image_singular::Bool,
)
    sx = typeof(k)(transform.signs[1])
    sy = typeof(k)(transform.signs[2])
    sz = typeof(k)(transform.signs[3])
    csx = typeof(k)(transform.determinant * transform.signs[1])
    csy = typeof(k)(transform.determinant * transform.signs[2])
    csz = typeof(k)(transform.determinant * transform.signs[3])
    return _launch_metal_gather_pair_kernels!(
        operators,
        cache,
        k,
        image_cache.pair_offsets,
        image_cache.trial_indices,
        skip_image_singular ? Int32(1) : Int32(2),
        sx, sy, sz,
        csx, csy, csz,
    )
end

# ===== BeatEngineMetalAssembly.jl lines 16-120 =====
function _assemble_regular_galerkin_operators_metal_native(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    rule::TriangleRule{T};
    skip_singular::Bool,
    singular_order::Int,
    element_indices,
    cache,
    timing,
    singular_cache,
    metal_singular_cache,
    symmetry_mode::Symbol,
) where {T<:AbstractFloat}
    normalized_mode = normalized_symmetry_mode(symmetry_mode)
    native_cache = cache === nothing ? build_metal_regular_assembly_cache(
        mesh,
        p1_space,
        dp0_space,
        rule;
        singular_order=singular_order,
        element_indices=element_indices,
        symmetry_mode=normalized_mode,
    ) : cache
    native_cache isa MetalRegularAssemblyCache || error("Native Metal assembly requires a MetalRegularAssemblyCache.")
    native_cache.symmetry_mode == normalized_mode ||
        error("Metal assembly cache symmetry mode $(native_cache.symmetry_mode) does not match requested $(normalized_mode).")
    singular_mode = _normalized_metal_singular_mode()

    operators = nothing
    storage = metal_operator_storage_mode()
    allocation_elapsed = @elapsed begin
        operators = (
            single_layer=Metal.zeros(Complex{T}, p1_space.global_dof_count, dp0_space.global_dof_count; storage=storage),
            double_layer=Metal.zeros(Complex{T}, p1_space.global_dof_count, p1_space.global_dof_count; storage=storage),
            adjoint_double_layer=Metal.zeros(Complex{T}, p1_space.global_dof_count, dp0_space.global_dof_count; storage=storage),
            hypersingular=Metal.zeros(Complex{T}, p1_space.global_dof_count, p1_space.global_dof_count; storage=storage),
        )
        Metal.synchronize()
    end
    timing !== nothing && (timing["metal_native_operator_alloc"] = allocation_elapsed)

    regular_kernel_mode = _normalized_metal_regular_kernel_mode()
    # In host singular mode the image regular kernels must integrate every
    # image pair with the regular rule (skip_mode 2), because the CPU image
    # correction is a Duffy-minus-regular delta. In native mode they skip the
    # image-singular pairs (skip_mode 1) and the gather kernel adds Duffy.
    skip_image_singular = !skip_singular && singular_mode == :native
    empty!(_metal_gather_stage_timing)
    kernel_elapsed = @elapsed begin
        if regular_kernel_mode == :pair_owned
            _launch_metal_regular_pair_kernels!(operators, native_cache, k)
        elseif regular_kernel_mode == :pair_atomic
            _launch_metal_regular_atomic_kernels!(operators, native_cache, k)
        elseif regular_kernel_mode == :pair_gather
            _launch_metal_regular_gather_kernels!(operators, native_cache, k)
        else
            _launch_metal_regular_entry_kernels!(operators, native_cache, k)
        end
        for (transform, image_cache) in zip(native_cache.image_transforms, native_cache.image_singular_caches)
            if regular_kernel_mode == :pair_owned
                _launch_metal_symmetry_regular_pair_kernels!(
                    operators,
                    native_cache,
                    image_cache,
                    transform,
                    k;
                    skip_image_singular=skip_image_singular,
                )
            elseif regular_kernel_mode == :pair_atomic
                _launch_metal_symmetry_regular_atomic_kernels!(
                    operators,
                    native_cache,
                    image_cache,
                    transform,
                    k;
                    skip_image_singular=skip_image_singular,
                )
            elseif regular_kernel_mode == :pair_gather
                _launch_metal_symmetry_regular_gather_kernels!(
                    operators,
                    native_cache,
                    image_cache,
                    transform,
                    k;
                    skip_image_singular=skip_image_singular,
                )
            else
                _launch_metal_symmetry_regular_entry_kernels!(
                    operators,
                    native_cache,
                    image_cache,
                    transform,
                    k;
                    skip_image_singular=skip_image_singular,
                )
            end
        end
        Metal.synchronize()
    end
    timing !== nothing && (timing["metal_native_regular_kernel"] = kernel_elapsed)
    if timing !== nothing
        for (stage, elapsed) in _metal_gather_stage_timing
            timing["metal_native_gather_" * stage] = elapsed
