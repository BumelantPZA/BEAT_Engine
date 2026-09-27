# Fused Burton-Miller exterior assembly for Metal.
#
# The four-operator path assembles S, K', D and H separately and combines them
# on the host (`burton_miller_neumann_matrices`):
#
#   lhs    = 0.5 I_p1p1 - D + (i/k) H                     (N x N)
#   rhs_op = -S - (i/k) (K' + 0.5 I_p1dp0)                (N x 2N)
#
# eta = i/k is known at assembly time, so both combinations can be formed
# inside the pair kernel instead. This file does that: one N x N system matrix
# and one right-hand-side vector per drive, never the four operators.
#
# What that is worth, measured on the ATH ladder (see the head-to-head doc):
#
# * Memory: N^2 complex instead of 6N^2 (S and K' are N x 2N and are two thirds
#   of the total). That is the certain win and it is what sets the dense
#   ceiling.
# * Time: the gather stage halves its pair-buffer reads and its operator
#   write-backs, worth ~2.4x on that stage. The pair kernel itself gains
#   nothing: it is 82-84% arithmetic-bound, and the arithmetic that forms the
#   combination costs back exactly what halving the stores saves.
#
# The four-operator path stays: the coupled FEM/LEM solver needs the operators
# separately, and so would any Calderon preconditioner. This is an
# exterior-only fast path chosen at assembly time, not a replacement.
#
# Multi-drive: every channel's Neumann vector is folded in the same pass, so
# one assembly serves the whole channel set at a frequency exactly as one
# factorization does. `q_neumann` is dp0_count x drive_count.
#
# Component layout per pair (24 floats, against 48 for the four operators):
#   0-8   lhs re (3x3, column-major)   9-17  lhs im
#   18-20 rhs coefficient re (3 rows)  21-23 rhs coefficient im

const _METAL_FUSED_COMPONENTS = 24

struct MetalFusedGatherTables
    chunk_size::Int
    chunk_count::Int
    elements              # MtlArray{Int32}: assembly position -> global element
    element_positions     # MtlArray{Int32}: global element -> assembly position (0 if absent)
    chunk_node_offsets::Vector{Int}
    chunk_nodes           # MtlArray{Int32}: chunk-node position -> global P1 dof
    inc_offsets           # MtlArray{Int32}
    inc_packed            # MtlArray{Int32}: (trial local - 1) * 4 + local column
    blocks                # MtlArray{Float32}: 24 * element_count * chunk_size
end

function _metal_fused_chunk_size(element_count::Int)
    element_count <= 0 && return 1
    budget_mb = parse(Float64, get(ENV, "BLAB_METAL_GATHER_BUDGET_MB", "512"))
    per_column_bytes = element_count * _METAL_FUSED_COMPONENTS * sizeof(Float32)
    chunk = clamp(floor(Int, budget_mb * 1e6 / per_column_bytes), 1, element_count)
    override = strip(get(ENV, "BLAB_METAL_GATHER_CHUNK", ""))
    isempty(override) || (chunk = clamp(parse(Int, override), 1, element_count))
    while element_count * chunk * _METAL_FUSED_COMPONENTS >= typemax(Int32) && chunk > 1
        chunk = chunk ÷ 2
    end
    return chunk
end

function _metal_fused_gather_tables(cache::MetalRegularAssemblyCache)
    tables = cache.fused_gather_tables[]
    tables === nothing || return tables
    indices = cache.element_indices
    element_count = length(indices)
    chunk_size = _metal_fused_chunk_size(element_count)
    chunk_count = cld(element_count, chunk_size)
    p1_dofs = Array(cache.p1_dofs)
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
    blocks = MtlArray{Float32}(undef, _METAL_FUSED_COMPONENTS * element_count * chunk_size)
    tables = MetalFusedGatherTables(
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
    cache.fused_gather_tables[] = tables
    return tables
end

function _release_metal_fused_gather_tables!(cache::MetalRegularAssemblyCache)
    tables = cache.fused_gather_tables[]
    tables === nothing && return nothing
    Metal.unsafe_free!(tables.elements)
    Metal.unsafe_free!(tables.element_positions)
    Metal.unsafe_free!(tables.chunk_nodes)
    Metal.unsafe_free!(tables.inc_offsets)
    Metal.unsafe_free!(tables.inc_packed)
    Metal.unsafe_free!(tables.blocks)
    cache.fused_gather_tables[] = nothing
    return nothing
end

# Both fused pair kernels form the same algebra as `burton_miller_neumann_matrices`,
# per pair and inside the accumulation:
#   lhs contribution of a pair: -D + (i/k) H, so
#     re = -D_re - H_im / k      im = -D_im + H_re / k
#   rhs coefficient of a pair: -S - (i/k) K', so
#     re = -S_re + K'_im / k     im = -S_im - K'_re / k
#
# The pair's Burton-Miller contribution, combined inside the accumulation
# rather than after it.
#
# The per-quadrature-point-pair arithmetic cannot drop: D and H carry different
# geometric prefactors per entry (D is basis_product * grad * trial_dot, H is
# curl - k^2 * basis_product * n.n' against the Green's value), so both terms
# are evaluated whatever they are accumulated into. Same for S against K'.
#
# The rank-1 *outer products* are a different matter. `_metal_regular_pair_blocks`
# accumulates four 3-vectors per test point and then expands each into its own
# 3x3 block, four expansions per test point. The Burton-Miller combination is
# linear, so it can be applied to the 3-vectors *before* the expansion, leaving
# one 3x3 expansion instead of two and one 3x1 instead of two. That halves the
# outer-product work and the live 3x3 accumulators, 48 floats to 24.
#
# The hypersingular curl term is not inside the test loop at all: H is
# curl_products * G0 - k^2 (n.n') * (basis-weighted sums), and only the second
# half accumulates per test point. The first half is added once after the loop.
@inline function _metal_regular_pair_fused_blocks(
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
    inverse_k,
    ::Val{R},
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
) where {R}
    T = typeof(k)
    inv_four_pi = T(0.07957747154594767)
    @inbounds begin
    lhs_re = zero(SVector{9,T})
    lhs_im = zero(SVector{9,T})
    rhs_re = zero(SVector{3,T})
    rhs_im = zero(SVector{3,T})
    test_nx = normals[test_index]
    test_ny = normals[test_index + face_count]
    test_nz = normals[test_index + Int32(2) * face_count]
    trial_nx = trial_sign_x * normals[trial_index]
    trial_ny = trial_sign_y * normals[trial_index + face_count]
    trial_nz = trial_sign_z * normals[trial_index + Int32(2) * face_count]
    normal_product = test_nx * trial_nx + test_ny * trial_ny + test_nz * trial_nz
    jac_scale = T(4) * areas[test_index] * areas[trial_index]
    trial_signs = SVector(trial_sign_x, trial_sign_y, trial_sign_z)
    # -(i/k) * k^2 (n.n'), folded into the per-test-point combination.
    curl_scale = inverse_k * k * k * normal_product

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

        s_re = zero(k)
        s_im = zero(k)
        a_re = zero(k)
        a_im = zero(k)
        d_re = zero(SVector{3,T})
        d_im = zero(SVector{3,T})
        h_re = zero(SVector{3,T})
        h_im = zero(SVector{3,T})
        context = (x, y, z, test_weight * jac_scale, k, inv_four_pi,
            test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
            element_rule_points, rule_points, rule_weights, trial_index, face_count)
        s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = _metal_trial_fold(
            (s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im),
            context, Val(R), Val(R),
        )
        # rhs coefficient: -S - (i/k) K'
        rhs_re += test_basis * (-s_re + inverse_k * a_im)
        rhs_im += test_basis * (-s_im - inverse_k * a_re)
        g_total_re += s_re
        g_total_im += s_im
        # lhs: -D + (i/k) H, less the curl term added after the loop.
        u_re = -d_re + curl_scale * h_im
        u_im = -d_im - curl_scale * h_re
        lhs_re += SVector(
            tb1 * u_re[1], tb2 * u_re[1], tb3 * u_re[1],
            tb1 * u_re[2], tb2 * u_re[2], tb3 * u_re[2],
            tb1 * u_re[3], tb2 * u_re[3], tb3 * u_re[3],
        )
        lhs_im += SVector(
            tb1 * u_im[1], tb2 * u_im[1], tb3 * u_im[1],
            tb1 * u_im[2], tb2 * u_im[2], tb3 * u_im[2],
            tb1 * u_im[3], tb2 * u_im[3], tb3 * u_im[3],
        )
        test_q += Int32(1)
    end
    # (i/k) * curl_products * G0, computed after the loop so its nine values are
    # not live registers during it.
    curl_products = _metal_pair_curl_products(
        curls,
        test_index,
        trial_index,
        face_count,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
    )
    lhs_re -= curl_products * (inverse_k * g_total_im)
    lhs_im += curl_products * (inverse_k * g_total_re)
    return lhs_re, lhs_im, rhs_re, rhs_im
    end
end

function _metal_fused_pair_blocks_kernel!(
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
    inverse_k,
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
    ::Val{PROBE}=Val(0),
    ::Val{ACC}=Val(false),
) where {R,PROBE,ACC}
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
        ACC && return nothing   # a later transform adds nothing to a skipped pair
        component = Int32(0)
        while component < Int32(_METAL_FUSED_COMPONENTS)
            @inbounds blocks[base + component * pair_stride] = zero(eltype(blocks))
            component += Int32(1)
        end
        return nothing
    end
    # Test probes (BLAB_TEST_FUSED_PROBE, timing only): 1 = no store, 2 = no maths.
    if PROBE == 2
        f = Float32(test_index) * 1.0f-4 + Float32(trial_index) * 1.0f-5
        lhs_re = SVector(f, f + 1, f + 2, f + 3, f + 4, f + 5, f + 6, f + 7, f + 8)
        _metal_store_block!(blocks, base, pair_stride, Int32(0), lhs_re)
        _metal_store_block!(blocks, base, pair_stride, Int32(9), lhs_re * 2.0f0)
        _metal_store_block!(blocks, base, pair_stride, Int32(18), SVector(f, f + 1, f + 2))
        _metal_store_block!(blocks, base, pair_stride, Int32(21), SVector(f, f + 2, f + 3))
        return nothing
    end
    lhs_re, lhs_im, rhs_re, rhs_im = _metal_regular_pair_fused_blocks(
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
        inverse_k,
        Val(R),
        trial_sign_x,
        trial_sign_y,
        trial_sign_z,
        trial_curl_sign_x,
        trial_curl_sign_y,
        trial_curl_sign_z,
    )
    if PROBE == 1
        total = sum(lhs_re) + sum(lhs_im) + sum(rhs_re) + sum(rhs_im)
        total == -1.2345f-30 && @inbounds blocks[base] = total
        return nothing
    end
    if ACC
        _test_add_block!(blocks, base, pair_stride, Int32(0), lhs_re)
        _test_add_block!(blocks, base, pair_stride, Int32(9), lhs_im)
        _test_add_block!(blocks, base, pair_stride, Int32(18), rhs_re)
        _test_add_block!(blocks, base, pair_stride, Int32(21), rhs_im)
        return nothing
    end
    _metal_store_block!(blocks, base, pair_stride, Int32(0), lhs_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(9), lhs_im)
    _metal_store_block!(blocks, base, pair_stride, Int32(18), rhs_re)
    _metal_store_block!(blocks, base, pair_stride, Int32(21), rhs_im)
    return nothing
end

# Test (BLAB_TEST_FUSED_PACKED=1): the fused pair kernel with the tile-reduce path's load diet:
# float4 points, normals and curls (`_metal_packed_pair_tables_for`), rule constants as a Val tuple and
# both quadrature loops unrolled. The Burton-Miller combination per test point is unchanged.
@inline function _test_fused_packed_test_point(acc, context, ::Val{Q}, rc, rv::Val{R}) where {Q,R}
    lhs_re, lhs_im, rhs_re, rhs_im, g_total_re, g_total_im = acc
    k, inv_four_pi, jac_scale, test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        points4, test_index, trial_index, inverse_k, curl_scale = context
    T = typeof(k)
    test_xi = _metal_rule_xi(rc, rv, Q)
    test_eta = _metal_rule_eta(rc, rv, Q)
    tb1 = one(k) - test_xi - test_eta
    tb2 = test_xi
    tb3 = test_eta
    test_basis = SVector(tb1, tb2, tb3)
    @inbounds p = points4[(test_index - Int32(1)) * Int32(R) + Int32(Q)]
    x = p[1].value
    y = p[2].value
    z = p[3].value
    test_weight = _metal_rule_w(rc, rv, Q)
    z3 = zero(SVector{3,T})
    trial_context = (x, y, z, test_weight * jac_scale, k, inv_four_pi,
        test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs, points4, trial_index, Val(0))
    s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = _metal_packed_trial_fold(
        (zero(k), zero(k), zero(k), zero(k), z3, z3, z3, z3), trial_context, Val(R), rc, rv)
    rhs_re += test_basis * (-s_re + inverse_k * a_im)
    rhs_im += test_basis * (-s_im - inverse_k * a_re)
    g_total_re += s_re
    g_total_im += s_im
    u_re = -d_re + curl_scale * h_im
    u_im = -d_im - curl_scale * h_re
    lhs_re += SVector(
        tb1 * u_re[1], tb2 * u_re[1], tb3 * u_re[1],
        tb1 * u_re[2], tb2 * u_re[2], tb3 * u_re[2],
        tb1 * u_re[3], tb2 * u_re[3], tb3 * u_re[3],
    )
    lhs_im += SVector(
        tb1 * u_im[1], tb2 * u_im[1], tb3 * u_im[1],
        tb1 * u_im[2], tb2 * u_im[2], tb3 * u_im[2],
        tb1 * u_im[3], tb2 * u_im[3], tb3 * u_im[3],
    )
    return (lhs_re, lhs_im, rhs_re, rhs_im, g_total_re, g_total_im)
end

# BLAB_TEST_FUSED_PACKED=2: the test-point loop stays a runtime loop (rule values from the device
# arrays, as the stock fused kernel does); only the trial fold is unrolled.
@inline function _test_fused_packed_test_loop(acc, context, rule_points, rule_weights, rc, rv::Val{R}) where {R}
    lhs_re, lhs_im, rhs_re, rhs_im, g_total_re, g_total_im = acc
    k, inv_four_pi, jac_scale, test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        points4, test_index, trial_index, inverse_k, curl_scale = context
    T = typeof(k)
    z3 = zero(SVector{3,T})
    test_q = Int32(1)
    while test_q <= Int32(R)
        @inbounds test_xi = rule_points[test_q]
        @inbounds test_eta = rule_points[test_q + Int32(R)]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        test_basis = SVector(tb1, tb2, tb3)
        @inbounds p = points4[(test_index - Int32(1)) * Int32(R) + test_q]
        @inbounds test_weight = rule_weights[test_q]
        trial_context = (p[1].value, p[2].value, p[3].value, test_weight * jac_scale, k, inv_four_pi,
            test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs, points4, trial_index, Val(0))
        s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = _metal_packed_trial_fold(
            (zero(k), zero(k), zero(k), zero(k), z3, z3, z3, z3), trial_context, Val(R), rc, rv)
        rhs_re += test_basis * (-s_re + inverse_k * a_im)
        rhs_im += test_basis * (-s_im - inverse_k * a_re)
        g_total_re += s_re
        g_total_im += s_im
        u_re = -d_re + curl_scale * h_im
        u_im = -d_im - curl_scale * h_re
        lhs_re += SVector(
            tb1 * u_re[1], tb2 * u_re[1], tb3 * u_re[1],
            tb1 * u_re[2], tb2 * u_re[2], tb3 * u_re[2],
            tb1 * u_re[3], tb2 * u_re[3], tb3 * u_re[3],
        )
        lhs_im += SVector(
            tb1 * u_im[1], tb2 * u_im[1], tb3 * u_im[1],
            tb1 * u_im[2], tb2 * u_im[2], tb3 * u_im[2],
            tb1 * u_im[3], tb2 * u_im[3], tb3 * u_im[3],
        )
        test_q += Int32(1)
    end
    return (lhs_re, lhs_im, rhs_re, rhs_im, g_total_re, g_total_im)
end

@inline _test_fused_packed_test_fold(acc, context, ::Val{0}, rc, rv) = acc
@inline function _test_fused_packed_test_fold(acc, context, ::Val{N}, rc, rv) where {N}
    acc = _test_fused_packed_test_fold(acc, context, Val(N - 1), rc, rv)
    return _test_fused_packed_test_point(acc, context, Val(N), rc, rv)
end

function _test_fused_pair_blocks_packed_kernel!(
    blocks, points4, normals4, areas, curls4, faces, elements, rule_points, rule_weights,
    element_count::Int32, chunk_start::Int32, chunk_count::Int32, pair_stride::Int32,
    k, inverse_k, face_count::Int32, rc, rv::Val{R},
    pair_offsets, singular_trial_indices, skip_mode,
    trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    ::Val{ACC},
    ::Val{LOOP},
    ::Val{FAR}=Val(0),
    centroids4=nothing,
    far_rho::Float32=0.0f0,
    far_kh::Float32=0.0f0,
) where {R,ACC,LOOP,FAR}
    position = thread_position_in_grid_2d()
    test_position = Int32(position.x)
    trial_local = Int32(position.y)
    (test_position > element_count || trial_local > chunk_count) && return nothing
    @inbounds test_index = Int32(elements[test_position])
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    base = test_position + element_count * (trial_local - Int32(1))
    # Test (BLAB_TEST_FAR_ORDER=rho/kh): FAR = 1 is the near pass (full rule, far pairs skipped), FAR = 2
    # the far pass (launched with the 3-point tables, adds far pairs only). Far: centroid distance at least
    # rho x (sum of the circumradii) and k x (sum of the circumradii) below kh.
    skip_far = false
    if FAR > 0
        @inbounds ct = centroids4[test_index]
        @inbounds cr = centroids4[trial_index]
        ex = cr[1].value * trial_sign_x - ct[1].value
        ey = cr[2].value * trial_sign_y - ct[2].value
        ez = cr[3].value * trial_sign_z - ct[3].value
        radii = ct[4].value + cr[4].value
        far = ex * ex + ey * ey + ez * ez >= far_rho * far_rho * radii * radii && k * radii < far_kh
        skip_far = FAR == 1 ? far : !far
    end
    if skip_far || _metal_pair_is_skipped(faces, face_count, test_index, trial_index, pair_offsets, singular_trial_indices, skip_mode)
        (ACC || FAR == 2) && return nothing
        component = Int32(0)
        while component < Int32(_METAL_FUSED_COMPONENTS)
            @inbounds blocks[base + component * pair_stride] = zero(eltype(blocks))
            component += Int32(1)
        end
        return nothing
    end
    T = typeof(k)
    inv_four_pi = T(0.07957747154594767)
    @inbounds tn = normals4[test_index]
    @inbounds rn = normals4[trial_index]
    test_nx = tn[1].value
    test_ny = tn[2].value
    test_nz = tn[3].value
    trial_nx = trial_sign_x * rn[1].value
    trial_ny = trial_sign_y * rn[2].value
    trial_nz = trial_sign_z * rn[3].value
    normal_product = test_nx * trial_nx + test_ny * trial_ny + test_nz * trial_nz
    @inbounds jac_scale = T(4) * areas[test_index] * areas[trial_index]
    trial_signs = SVector(trial_sign_x, trial_sign_y, trial_sign_z)
    curl_scale = inverse_k * k * k * normal_product
    context = (k, inv_four_pi, jac_scale, test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        points4, test_index, trial_index, inverse_k, curl_scale)
    acc = (zero(SVector{9,T}), zero(SVector{9,T}), zero(SVector{3,T}), zero(SVector{3,T}), zero(k), zero(k))
    lhs_re, lhs_im, rhs_re, rhs_im, g_total_re, g_total_im = LOOP ?
        _test_fused_packed_test_loop(acc, context, rule_points, rule_weights, rc, rv) :
        _test_fused_packed_test_fold(acc, context, rv, rc, rv)
    @inbounds t1 = curls4[(test_index - Int32(1)) * Int32(3) + Int32(1)]
    @inbounds t2 = curls4[(test_index - Int32(1)) * Int32(3) + Int32(2)]
    @inbounds t3 = curls4[(test_index - Int32(1)) * Int32(3) + Int32(3)]
    @inbounds q1 = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(1)]
    @inbounds q2 = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(2)]
    @inbounds q3 = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(3)]
    t11, t12, t13 = t1[1].value, t1[2].value, t1[3].value
    t21, t22, t23 = t2[1].value, t2[2].value, t2[3].value
    t31, t32, t33 = t3[1].value, t3[2].value, t3[3].value
    r11 = trial_curl_sign_x * q1[1].value
    r12 = trial_curl_sign_y * q1[2].value
    r13 = trial_curl_sign_z * q1[3].value
    r21 = trial_curl_sign_x * q2[1].value
    r22 = trial_curl_sign_y * q2[2].value
    r23 = trial_curl_sign_z * q2[3].value
    r31 = trial_curl_sign_x * q3[1].value
    r32 = trial_curl_sign_y * q3[2].value
    r33 = trial_curl_sign_z * q3[3].value
    curl_products = SVector(
        t11 * r11 + t12 * r12 + t13 * r13,
        t21 * r11 + t22 * r12 + t23 * r13,
        t31 * r11 + t32 * r12 + t33 * r13,
        t11 * r21 + t12 * r22 + t13 * r23,
        t21 * r21 + t22 * r22 + t23 * r23,
        t31 * r21 + t32 * r22 + t33 * r23,
        t11 * r31 + t12 * r32 + t13 * r33,
        t21 * r31 + t22 * r32 + t23 * r33,
        t31 * r31 + t32 * r32 + t33 * r33,
    )
    lhs_re -= curl_products * (inverse_k * g_total_im)
    lhs_im += curl_products * (inverse_k * g_total_re)
    if ACC
        _test_add_block!(blocks, base, pair_stride, Int32(0), lhs_re)
        _test_add_block!(blocks, base, pair_stride, Int32(9), lhs_im)
        _test_add_block!(blocks, base, pair_stride, Int32(18), rhs_re)
        _test_add_block!(blocks, base, pair_stride, Int32(21), rhs_im)
    else
        _metal_store_block!(blocks, base, pair_stride, Int32(0), lhs_re)
        _metal_store_block!(blocks, base, pair_stride, Int32(9), lhs_im)
        _metal_store_block!(blocks, base, pair_stride, Int32(18), rhs_re)
        _metal_store_block!(blocks, base, pair_stride, Int32(21), rhs_im)
    end
    return nothing
end

@inline function _test_add_block!(blocks, base::Int32, stride::Int32, offset::Int32, values::SVector{N,T}) where {N,T}
    i = 1
    while i <= N
        @inbounds blocks[base + (offset + Int32(i - 1)) * stride] += values[i]
        i += 1
    end
    return nothing
end

# One thread per (P1 row, P1 node touched by the chunk), exactly the ownership
# of `_metal_gather_dlp_hyp_kernel!` but reading two components instead of four
# and writing one matrix instead of two.
function _metal_fused_lhs_gather_kernel!(
    lhs,
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
    value_re = zero(eltype(blocks))
    value_im = zero(eltype(blocks))
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
            @inbounds value_re += blocks[pair + (component - Int32(1)) * pair_stride]
            @inbounds value_im += blocks[pair + (component + Int32(8)) * pair_stride]
            chunk_position += Int32(1)
        end
        incident_position += Int32(1)
    end
    @inbounds lhs[row + (column - Int32(1)) * p1_count] += Complex(value_re, value_im)
    return nothing
end

# One thread per (P1 row, trial element of the chunk, drive): the same
# ownership as `_metal_gather_slp_adjoint_kernel!`, but the result is a
# right-hand-side contribution rather than an operator column. Row `row` is
# touched by every trial element, so the sum over trial elements cannot happen
# here without a race; each (row, trial local) pair keeps its own partial and
# `_metal_fused_rhs_reduce_kernel!` sums them once at the end. The partial
# survives across chunks because (row, trial local) is the same owner in every
# chunk.
function _metal_fused_rhs_gather_kernel!(
    rhs_partial,
    blocks,
    elements,
    element_positions,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    element_dp0_dofs,
    q_neumann,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    chunk_size::Int32,
    pair_stride::Int32,
    p1_count::Int32,
    dp0_count::Int32,
    drive_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * chunk_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    trial_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    column_base = element_count * (trial_local - Int32(1))
    value_re = zero(eltype(blocks))
    value_im = zero(eltype(blocks))
    @inbounds incident_position = Int32(vertex_offsets[row])
    @inbounds incident_stop = Int32(vertex_offsets[row + Int32(1)]) - Int32(1)
    while incident_position <= incident_stop
        @inbounds test_position = Int32(element_positions[incident_elements[incident_position]])
        @inbounds local_row = Int32(incident_local_indices[incident_position])
        pair = test_position + column_base
        @inbounds value_re += blocks[pair + (local_row + Int32(17)) * pair_stride]
        @inbounds value_im += blocks[pair + (local_row + Int32(20)) * pair_stride]
        incident_position += Int32(1)
    end
    coefficient = Complex(value_re, value_im)
    @inbounds dp0_column = Int32(element_dp0_dofs[trial_index])
    partial_index = row + (trial_local - Int32(1)) * p1_count
    drive = Int32(1)
    while drive <= drive_count
        @inbounds rhs_partial[partial_index + (drive - Int32(1)) * p1_count * chunk_size] +=
            coefficient * q_neumann[dp0_column + (drive - Int32(1)) * dp0_count]
        drive += Int32(1)
    end
    return nothing
end

function _metal_fused_rhs_reduce_kernel!(
    rhs,
    rhs_partial,
    p1_count::Int32,
    chunk_size::Int32,
    drive_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * drive_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    drive = (index - Int32(1)) ÷ p1_count + Int32(1)
    base = (drive - Int32(1)) * p1_count * chunk_size
    total = zero(eltype(rhs))
    column = Int32(1)
    while column <= chunk_size
        @inbounds total += rhs_partial[base + row + (column - Int32(1)) * p1_count]
        column += Int32(1)
    end
    @inbounds rhs[index] = total
    return nothing
end

function _launch_metal_fused_pair_kernels!(
    lhs,
    rhs_partial,
    q_neumann,
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
    return _launch_metal_fused_pair_kernels!(lhs, rhs_partial, q_neumann, cache, k, Any[(
        pair_offsets, singular_trial_indices, skip_mode,
        trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z)])
end

# `transforms`: one tuple (pair_offsets, singular_trial_indices, skip_mode, 6 signs) per symmetry
# transform. Per chunk, the first transform writes the pair blocks and later ones add into them
# (BLAB_TEST_FUSED_IMAGE_ACC=1 passes all transforms at once), so the gathers run once per chunk.
function _launch_metal_fused_pair_kernels!(lhs, rhs_partial, q_neumann, cache::MetalRegularAssemblyCache, k, transforms)
    element_count = length(cache.element_indices)
    element_count == 0 && return nothing
    rule_count = cache.rule_count
    rule_count in (1, 3, 6) || error("Fused Metal assembly expects a 1-, 3-, or 6-point triangle rule; got $(rule_count).")
    tables = _metal_fused_gather_tables(cache)
    chunk_size = tables.chunk_size
    pair_stride = Int32(element_count * chunk_size)
    tile_x, tile_y = _metal_atomic_tile()
    groupsize = _metal_kernel_groupsize()
    p1_count = Int32(cache.p1_dof_count)
    dp0_count = Int32(cache.dp0_dof_count)
    drive_count = Int32(size(q_neumann, 2))
    timed = get(ENV, "BLAB_METAL_GATHER_TIMING", "0") == "1"
    packed_mode = parse(Int, get(ENV, "BLAB_TEST_FUSED_PACKED", "0"))
    packed = packed_mode > 0 ? _metal_packed_pair_tables_for(cache) : nothing
    far_setting = strip(get(ENV, "BLAB_TEST_FAR_ORDER", ""))
    far_tables = nothing
    far_rho = far_kh = 0.0f0
    if packed_mode == 2 && !isempty(far_setting) && cache.rule_count > 3
        far_rho, far_kh = parse.(Float32, split(far_setting, "/"))
        far_tables = _test_far_order_tables_for(cache)
    end
    timed && Metal.synchronize()
    stamp = time()
    for chunk in 1:tables.chunk_count
        chunk_start = (chunk - 1) * chunk_size + 1
        chunk_count = min(chunk_size, element_count - chunk_start + 1)
      for (transform_number, transform) in enumerate(transforms)
        (pair_offsets, singular_trial_indices, skip_mode, trial_sign_x, trial_sign_y, trial_sign_z,
         trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z) = transform
        pair_args = (
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
            inv(k),
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
            Val(parse(Int, get(ENV, "BLAB_TEST_FUSED_PROBE", "0"))),
            Val(transform_number > 1),
        )
        if packed !== nothing
            packed_args = (
                tables.blocks, packed.points4, packed.normals4, cache.areas, packed.curls4, cache.faces, tables.elements,
                cache.rule_points, cache.rule_weights,
                Int32(element_count), Int32(chunk_start), Int32(chunk_count), pair_stride,
                k, inv(k), Int32(cache.face_count), Val(packed.rule), Val(rule_count),
                pair_offsets, singular_trial_indices, skip_mode,
                trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
                Val(transform_number > 1), Val(packed_mode == 2),
            )
            if far_tables !== nothing
                near_args = (packed_args..., Val(1), far_tables.centroids4, far_rho, far_kh)
                @_test_cold_launch Metal.@metal threads=(tile_x, tile_y) groups=(cld(element_count, tile_x), cld(chunk_count, tile_y)) _test_fused_pair_blocks_packed_kernel!(near_args...)
                packed_args = (
                    tables.blocks, far_tables.points3, packed.normals4, cache.areas, packed.curls4, cache.faces, tables.elements,
                    far_tables.rule3_points, far_tables.rule3_weights,
                    Int32(element_count), Int32(chunk_start), Int32(chunk_count), pair_stride,
                    k, inv(k), Int32(cache.face_count), Val(_TEST_RULE3), Val(3),
                    pair_offsets, singular_trial_indices, skip_mode,
                    trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
                    Val(true), Val(true), Val(2), far_tables.centroids4, far_rho, far_kh,
                )
            end
            chunk == 1 && transform_number == 1 && _test_pipeinfo("fused_pair_packed", _test_fused_pair_blocks_packed_kernel!, packed_args...)
            @_test_cold_launch Metal.@metal threads=(tile_x, tile_y) groups=(cld(element_count, tile_x), cld(chunk_count, tile_y)) _test_fused_pair_blocks_packed_kernel!(packed_args...)
        else
        chunk == 1 && _test_pipeinfo("fused_pair", _metal_fused_pair_blocks_kernel!, pair_args...)
        @_test_cold_launch Metal.@metal threads=(tile_x, tile_y) groups=(cld(element_count, tile_x), cld(chunk_count, tile_y)) _metal_fused_pair_blocks_kernel!(pair_args...
        )
        end
        stamp = _metal_gather_stage!("fused_pairs", timed, stamp)
      end
        _metal_launch(
            _metal_fused_rhs_gather_kernel!,
            cache.p1_dof_count * chunk_count,
            rhs_partial,
            tables.blocks,
            tables.elements,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            cache.element_dp0_dofs,
            q_neumann,
            Int32(element_count),
            Int32(chunk_start),
            Int32(chunk_count),
            Int32(chunk_size),
            pair_stride,
            p1_count,
            dp0_count,
            drive_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("fused_rhs", timed, stamp)
        node_start = tables.chunk_node_offsets[chunk]
        node_count = tables.chunk_node_offsets[chunk + 1] - node_start
        _metal_launch(
            _metal_fused_lhs_gather_kernel!,
            cache.p1_dof_count * node_count,
            lhs,
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
        stamp = _metal_gather_stage!("fused_lhs", timed, stamp)
    end
    return nothing
end

# Singular corrections under fusion. The Duffy/Sauter-Schwab deltas are four
# per-operator blocks in the four-operator path; here they are combined with
# the same eta before the scatter. Getting this wrong is silent, so the
# equivalence gate compares a fused assembly against a four-operator one on the
# same mesh and frequency rather than trusting the validation tolerances.
#
# The combination is formed inside the Duffy quadrature loop rather than after
# it -- what `_metal_regular_pair_fused_blocks` already does for the regular
# kernel, and for the same two reasons.
#
# `_metal_singular_pair_blocks`, which the four-operator path still uses,
# carries 50 live accumulator floats through the loop (slp 3+3, adj 3+3,
# dlp 9+9, hb 9+9, g_total 1+1) and expands the rank-1 `outer` product four
# times per point pair. The Burton-Miller combination is linear, so it can be
# applied to the per-point scalars before the expansion: two 3x3 expansions
# instead of four, one 3x1 instead of two, and 26 live floats.
#
# The hypersingular curl term is loop-invariant (H is curl_products * G0 minus
# the basis-weighted part), so it is added once after the loop exactly as the
# regular kernel adds it. The remaining `g_total` pair is what carries it.
#
# Summation order therefore differs from `_metal_singular_pair_blocks` in
# Float32 -- which is what the fused-versus-four-operator equivalence gate
# measures, and why the four-operator path is left untouched as the reference.
# `scripts/validate_metal_singular_summation.jl` bounds that difference
# directly, per pair, against a Float64 evaluation of the same algebra.
#
# The per-pair algebra is `burton_miller_neumann_matrices` formed per pair:
#   lhs contribution: -D + (i/k) H, so re = -D_re - H_im / k, im = -D_im + H_re / k
#   rhs coefficient:  -S - (i/k) K', so re = -S_re + K'_im / k, im = -S_im - K'_re / k
@inline function _metal_singular_pair_fused_bm_blocks(
    linear_index::Int32,
    test_indices,
    trial_indices,
    rule_indices,
    jac_scales,
    normal_products,
    rule_offsets,
    rule_test_points,
    rule_trial_points,
    rule_weights,
    face_vertices,
    normals,
    curls,
    k,
    inverse_k,
    face_count::Int32,
    pair_count::Int32,
    rule_point_count::Int32,
    part_count::Int32,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    pair_position = (linear_index - Int32(1)) % pair_count + Int32(1)
    part = (linear_index - Int32(1)) ÷ pair_count + Int32(1)
    T = typeof(k)
    @inbounds begin
        test_index = Int32(test_indices[pair_position])
        trial_index = Int32(trial_indices[pair_position])
        rule_index = Int32(rule_indices[pair_position])
        q_first = Int32(rule_offsets[rule_index])
        q_last = Int32(rule_offsets[rule_index + Int32(1)]) - Int32(1)
        per_part = cld(q_last - q_first + Int32(1), part_count)
        q = q_first + (part - Int32(1)) * per_part
        q_stop = min(q + per_part - Int32(1), q_last)
        jac_scale = jac_scales[pair_position]
        normal_product = normal_products[pair_position]
        test_nx = normals[test_index]
        test_ny = normals[test_index + face_count]
        test_nz = normals[test_index + Int32(2) * face_count]
        trial_nx = trial_sign_x * normals[trial_index]
        trial_ny = trial_sign_y * normals[trial_index + face_count]
        trial_nz = trial_sign_z * normals[trial_index + Int32(2) * face_count]
    end
    inv_four_pi = T(0.07957747154594767)
    # -(i/k) * k^2 (n.n'), folded into the per-point combination.
    curl_scale = inverse_k * k * k * normal_product
    lhs_re = zero(SVector{9,T}); lhs_im = zero(SVector{9,T})
    rhs_re = zero(SVector{3,T}); rhs_im = zero(SVector{3,T})
    g_total_re = zero(T); g_total_im = zero(T)
    while q <= q_stop
        @inbounds begin
            test_xi = rule_test_points[q]
            test_eta = rule_test_points[q + rule_point_count]
            trial_xi = rule_trial_points[q]
            trial_eta = rule_trial_points[q + rule_point_count]
            weight = rule_weights[q] * jac_scale
        end
        tb1 = one(k) - test_xi - test_eta
        rb1 = one(k) - trial_xi - trial_eta
        x, y, z = _metal_face_point(face_vertices, test_index, face_count, tb1, test_xi, test_eta)
        sx, sy, sz = _metal_face_point(face_vertices, trial_index, face_count, rb1, trial_xi, trial_eta)
        Base.@fastmath begin
            dx = sx * trial_sign_x - x
            dy = sy * trial_sign_y - y
            dz = sz * trial_sign_z - z
            radius2 = dx * dx + dy * dy + dz * dz
            if radius2 > zero(k)
                inv_radius = _metal_fast_rsqrt(radius2)
                radius = radius2 * inv_radius
                phase = k * radius
                green_scale = inv_radius * inv_four_pi * weight
                green_re = _metal_fast_cos(phase) * green_scale
                green_im = _metal_fast_sin(phase) * green_scale
                grad_re = -green_re * inv_radius - green_im * k
                grad_im = green_re * k - green_im * inv_radius
                test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
                trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
                tb = SVector(tb1, test_xi, test_eta)
                outer = SVector(
                    tb1 * rb1, test_xi * rb1, test_eta * rb1,
                    tb1 * trial_xi, test_xi * trial_xi, test_eta * trial_xi,
                    tb1 * trial_eta, test_xi * trial_eta, test_eta * trial_eta,
                )
                # rhs coefficient: -S - (i/k) K'
                rhs_re += tb * (-green_re + inverse_k * (grad_im * test_dot))
                rhs_im += tb * (-green_im - inverse_k * (grad_re * test_dot))
                # lhs: -D + (i/k) H, less the loop-invariant curl term.
                u_re = -(grad_re * trial_dot) + curl_scale * green_im
                u_im = -(grad_im * trial_dot) - curl_scale * green_re
                lhs_re += outer * u_re
                lhs_im += outer * u_im
                g_total_re += green_re
                g_total_im += green_im
            end
        end
        q += Int32(1)
    end
    # (i/k) * curl_products * G0, added after the loop so its nine values are
    # not live registers during it.
    curl_products = _metal_pair_curl_products(
        curls, test_index, trial_index, face_count,
        trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    )
    lhs_re -= curl_products * (inverse_k * g_total_im)
    lhs_im += curl_products * (inverse_k * g_total_re)
    return lhs_re, lhs_im, rhs_re, rhs_im
end

function _metal_singular_fused_bm_blocks_kernel!(
    lhs_values,
    rhs_values,
    test_indices,
    trial_indices,
    rule_indices,
    jac_scales,
    normal_products,
    rule_offsets,
    rule_test_points,
    rule_trial_points,
    rule_weights,
    face_vertices,
    normals,
    curls,
    k,
    inverse_k,
    face_count::Int32,
    pair_count::Int32,
    rule_point_count::Int32,
    part_count::Int32,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
    ::Val{PROBE},
) where {PROBE}
    linear_index = Int32(thread_position_in_grid_1d())
    linear_index > pair_count * part_count && return nothing
    value_stride = pair_count * part_count
    # Timing probe (BLAB_TEST_SING_PROBE=2, wrong results): stores only, no quadrature.
    if PROBE == 2
        @inbounds begin
            i = 1
            while i <= 3
                rhs_values[linear_index + Int32(i - 1) * value_stride] = zero(eltype(rhs_values))
                i += 1
            end
            i = 1
            while i <= 9
                lhs_values[linear_index + Int32(i - 1) * value_stride] = zero(eltype(lhs_values))
                i += 1
            end
        end
        return nothing
    end
    lhs_re, lhs_im, rhs_re, rhs_im = _metal_singular_pair_fused_bm_blocks(
        linear_index,
        test_indices, trial_indices, rule_indices, jac_scales, normal_products,
        rule_offsets, rule_test_points, rule_trial_points, rule_weights,
        face_vertices, normals, curls, k, inverse_k,
        face_count, pair_count, rule_point_count, part_count,
        trial_sign_x, trial_sign_y, trial_sign_z,
        trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    )
    # Timing probe (BLAB_TEST_SING_PROBE=1, wrong results): full quadrature, the stores kept behind a
    # test that never holds so the compiler cannot drop the maths.
    PROBE == 1 && lhs_re[1] + rhs_im[3] != -1.2345f30 && return nothing
    @inbounds begin
        i = 1
        while i <= 3
            rhs_values[linear_index + Int32(i - 1) * value_stride] = Complex(rhs_re[i], rhs_im[i])
            i += 1
        end
        i = 1
        while i <= 9
            lhs_values[linear_index + Int32(i - 1) * value_stride] = Complex(lhs_re[i], lhs_im[i])
            i += 1
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Test (BLAB_TEST_SING_PACKED=1|2, Float32 only): load diet for the fused singular blocks. The maths and
# the summation order are those of `_metal_singular_pair_fused_bm_blocks`, but
#   - the pairs are grouped once per singular cache by rule length (coincident / edge / vertex rules
#     each have one point count), one launch per group with the point and part counts as Vals;
#   - each rule point is one float4 (test xi, test eta, trial xi, trial eta);
#   - face vertices, normals and curl rows are float4 rows (3 / 1 / 3 per face).
# 1 reloads both faces' vertices at every point, as stock does; 2 loads them once per thread; 3 is 2
# with the loop bound read at run time.
# Outputs keep stock's (pair, part) layout, so the gather after it is unchanged.

struct TestSingPackedTables
    groups::Vector{Tuple{Int,Any}}   # (rule point count, device Int32 pair positions)
    rule_points4
    vertices4
end

const _test_sing_packed_tables = WeakKeyDict{Any,TestSingPackedTables}()
_test_sing_packed_mode() = parse(Int, get(ENV, "BLAB_TEST_SING_PACKED", "0"))

function _test_sing_packed_tables_for(regular_cache, singular_cache)
    # Keyed by the singular cache's mutable gather_tables Ref, so it lives as long as that cache.
    get!(_test_sing_packed_tables, singular_cache.gather_tables) do
        F = regular_cache.face_count
        fv = Array(regular_cache.face_vertices)
        vertices4 = Vector{_MetalFloat4}(undef, 3F)
        for face in 1:F, v in 1:3
            b = face + 3 * (v - 1) * F
            vertices4[(face - 1) * 3 + v] = _metal_float4(Float32(fv[b]), Float32(fv[b + F]), Float32(fv[b + 2F]), 0.0f0)
        end
        tp = vec(Array(singular_cache.rule_test_points))
        rp = vec(Array(singular_cache.rule_trial_points))
        n = length(singular_cache.rule_weights)
        points4 = [_metal_float4(Float32(tp[q]), Float32(tp[q + n]), Float32(rp[q]), Float32(rp[q + n])) for q in 1:n]
        offsets = Array(singular_cache.rule_offsets)
        by_length = Dict{Int,Vector{Int32}}()
        for (position, rule) in enumerate(Array(singular_cache.rule_indices))
            push!(get!(by_length, Int(offsets[rule + 1] - offsets[rule]), Int32[]), Int32(position))
        end
        groups = Tuple{Int,Any}[(len, MtlArray(by_length[len])) for len in sort!(collect(keys(by_length)))]
        TestSingPackedTables(groups, MtlArray(points4), MtlArray(vertices4))
    end
end

@inline function _test_face_point4(v1, v2, v3, basis1, basis2, basis3)
    x = basis1 * v1[1].value + basis2 * v2[1].value + basis3 * v3[1].value
    y = basis1 * v1[2].value + basis2 * v2[2].value + basis3 * v3[2].value
    z = basis1 * v1[3].value + basis2 * v2[3].value + basis3 * v3[3].value
    return x, y, z
end

@inline function _test_pair_curl_products4(curls4, test_index, trial_index, csx, csy, csz)
    @inbounds begin
        ta = curls4[(test_index - Int32(1)) * Int32(3) + Int32(1)]
        tb = curls4[(test_index - Int32(1)) * Int32(3) + Int32(2)]
        tc = curls4[(test_index - Int32(1)) * Int32(3) + Int32(3)]
        ra = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(1)]
        rb = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(2)]
        rc = curls4[(trial_index - Int32(1)) * Int32(3) + Int32(3)]
    end
    t11 = ta[1].value; t12 = ta[2].value; t13 = ta[3].value
    t21 = tb[1].value; t22 = tb[2].value; t23 = tb[3].value
    t31 = tc[1].value; t32 = tc[2].value; t33 = tc[3].value
    r11 = csx * ra[1].value; r12 = csy * ra[2].value; r13 = csz * ra[3].value
    r21 = csx * rb[1].value; r22 = csy * rb[2].value; r23 = csz * rb[3].value
    r31 = csx * rc[1].value; r32 = csy * rc[2].value; r33 = csz * rc[3].value
    return SVector(
        t11 * r11 + t12 * r12 + t13 * r13,
        t21 * r11 + t22 * r12 + t23 * r13,
        t31 * r11 + t32 * r12 + t33 * r13,
        t11 * r21 + t12 * r22 + t13 * r23,
        t21 * r21 + t22 * r22 + t23 * r23,
        t31 * r21 + t32 * r22 + t33 * r23,
        t11 * r31 + t12 * r32 + t13 * r33,
        t21 * r31 + t22 * r32 + t23 * r33,
        t31 * r31 + t32 * r32 + t33 * r33,
    )
end

function _test_sing_packed_kernel!(
    lhs_values, rhs_values, positions,
    test_indices, trial_indices, rule_indices, jac_scales, normal_products, rule_offsets,
    points4, rule_weights, vertices4, normals4, curls4,
    k::Float32, inverse_k::Float32, group_count::Int32, pair_count::Int32,
    trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    ::Val{N}, ::Val{P}, ::Val{MODE},
) where {N,P,MODE}
    thread = Int32(thread_position_in_grid_1d())
    thread > group_count * Int32(P) && return nothing
    group_position = (thread - Int32(1)) % group_count + Int32(1)
    part = (thread - Int32(1)) ÷ group_count + Int32(1)
    T = Float32
    @inbounds begin
        pair_position = positions[group_position]
        test_index = Int32(test_indices[pair_position])
        trial_index = Int32(trial_indices[pair_position])
        rule_index = Int32(rule_indices[pair_position])
        q_first = Int32(rule_offsets[rule_index])
        # 3: the loop bound read at run time, as stock does (with 2's hoisted vertices).
        q_last = MODE == 3 ? Int32(rule_offsets[rule_index + Int32(1)]) - Int32(1) : q_first + Int32(N) - Int32(1)
        per_part = MODE == 3 ? cld(q_last - q_first + Int32(1), Int32(P)) : Int32(cld(N, P))
        q = q_first + (part - Int32(1)) * per_part
        q_stop = min(q + per_part - Int32(1), q_last)
        jac_scale = jac_scales[pair_position]
        normal_product = normal_products[pair_position]
        tn = normals4[test_index]
        rn = normals4[trial_index]
        test_nx = tn[1].value
        test_ny = tn[2].value
        test_nz = tn[3].value
        trial_nx = trial_sign_x * rn[1].value
        trial_ny = trial_sign_y * rn[2].value
        trial_nz = trial_sign_z * rn[3].value
        test_row = (test_index - Int32(1)) * Int32(3)
        trial_row = (trial_index - Int32(1)) * Int32(3)
        if MODE >= 2
            tv1 = vertices4[test_row + Int32(1)]; tv2 = vertices4[test_row + Int32(2)]; tv3 = vertices4[test_row + Int32(3)]
            sv1 = vertices4[trial_row + Int32(1)]; sv2 = vertices4[trial_row + Int32(2)]; sv3 = vertices4[trial_row + Int32(3)]
        end
    end
    inv_four_pi = T(0.07957747154594767)
    curl_scale = inverse_k * k * k * normal_product
    lhs_re = zero(SVector{9,T}); lhs_im = zero(SVector{9,T})
    rhs_re = zero(SVector{3,T}); rhs_im = zero(SVector{3,T})
    g_total_re = zero(T); g_total_im = zero(T)
    while q <= q_stop
        @inbounds begin
            rule_point = points4[q]
            weight = rule_weights[q] * jac_scale
            if MODE < 2
                tv1 = vertices4[test_row + Int32(1)]; tv2 = vertices4[test_row + Int32(2)]; tv3 = vertices4[test_row + Int32(3)]
                sv1 = vertices4[trial_row + Int32(1)]; sv2 = vertices4[trial_row + Int32(2)]; sv3 = vertices4[trial_row + Int32(3)]
            end
        end
        test_xi = rule_point[1].value
        test_eta = rule_point[2].value
        trial_xi = rule_point[3].value
        trial_eta = rule_point[4].value
        tb1 = one(k) - test_xi - test_eta
        rb1 = one(k) - trial_xi - trial_eta
        x, y, z = _test_face_point4(tv1, tv2, tv3, tb1, test_xi, test_eta)
        sx, sy, sz = _test_face_point4(sv1, sv2, sv3, rb1, trial_xi, trial_eta)
        Base.@fastmath begin
            dx = sx * trial_sign_x - x
            dy = sy * trial_sign_y - y
            dz = sz * trial_sign_z - z
            radius2 = dx * dx + dy * dy + dz * dz
            if radius2 > zero(k)
                inv_radius = _metal_fast_rsqrt(radius2)
                radius = radius2 * inv_radius
                phase = k * radius
                green_scale = inv_radius * inv_four_pi * weight
                green_re = _metal_fast_cos(phase) * green_scale
                green_im = _metal_fast_sin(phase) * green_scale
                grad_re = -green_re * inv_radius - green_im * k
                grad_im = green_re * k - green_im * inv_radius
                test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
                trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
                tb = SVector(tb1, test_xi, test_eta)
                outer = SVector(
                    tb1 * rb1, test_xi * rb1, test_eta * rb1,
                    tb1 * trial_xi, test_xi * trial_xi, test_eta * trial_xi,
                    tb1 * trial_eta, test_xi * trial_eta, test_eta * trial_eta,
                )
                rhs_re += tb * (-green_re + inverse_k * (grad_im * test_dot))
                rhs_im += tb * (-green_im - inverse_k * (grad_re * test_dot))
                u_re = -(grad_re * trial_dot) + curl_scale * green_im
                u_im = -(grad_im * trial_dot) - curl_scale * green_re
                lhs_re += outer * u_re
                lhs_im += outer * u_im
                g_total_re += green_re
                g_total_im += green_im
            end
        end
        q += Int32(1)
    end
    curl_products = _test_pair_curl_products4(
        curls4, test_index, trial_index, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    )
    lhs_re -= curl_products * (inverse_k * g_total_im)
    lhs_im += curl_products * (inverse_k * g_total_re)
    linear_index = pair_position + (part - Int32(1)) * pair_count
    value_stride = pair_count * Int32(P)
    @inbounds begin
        i = 1
        while i <= 3
            rhs_values[linear_index + Int32(i - 1) * value_stride] = Complex(rhs_re[i], rhs_im[i])
            i += 1
        end
        i = 1
        while i <= 9
            lhs_values[linear_index + Int32(i - 1) * value_stride] = Complex(lhs_re[i], lhs_im[i])
            i += 1
        end
    end
    return nothing
end

function _metal_singular_fused_bm_scatter_kernel!(
    lhs_f32,
    rhs_f32,
    lhs_values,
    rhs_values,
    q_neumann,
    test_indices,
    trial_indices,
    p1_dofs,
    element_dp0_dofs,
    pair_count,
    part_count,
    p1_dof_count,
    dp0_dof_count,
    drive_count,
    face_count,
)
    pair_position = _metal_global_linear_index()
    pair_position > pair_count && return nothing
    value_stride = pair_count * part_count
    test_index = Int(test_indices[pair_position])
    trial_index = Int(trial_indices[pair_position])
    dp0_column = Int(element_dp0_dofs[trial_index])
    local_row = 1
    while local_row <= 3
        row = Int(p1_dofs[test_index + (local_row - 1) * face_count])
        coefficient = zero(eltype(rhs_values))
        part = 1
        while part <= part_count
            coefficient += rhs_values[pair_position + (part - 1) * pair_count + (local_row - 1) * value_stride]
            part += 1
        end
        drive = 1
        while drive <= drive_count
            contribution = coefficient * q_neumann[dp0_column + (drive - 1) * dp0_dof_count]
            _metal_atomic_add_complex!(
                rhs_f32,
                row + (drive - 1) * p1_dof_count,
                real(contribution),
                imag(contribution),
            )
            drive += 1
        end
        local_row += 1
    end
    local_column = 1
    while local_column <= 3
        column = Int(p1_dofs[trial_index + (local_column - 1) * face_count])
        local_row = 1
        while local_row <= 3
            row = Int(p1_dofs[test_index + (local_row - 1) * face_count])
            component = (local_column - 1) * 3 + local_row - 1
            value = zero(eltype(lhs_values))
            part = 1
            while part <= part_count
                value += lhs_values[pair_position + (part - 1) * pair_count + component * value_stride]
                part += 1
            end
            _metal_atomic_add_complex!(
                lhs_f32,
                row + (column - 1) * p1_dof_count,
                real(value),
                imag(value),
            )
            local_row += 1
        end
        local_column += 1
    end
    return nothing
end

# Test (BLAB_TEST_FUSED_POOL=1): device scratch of the fused path (singular value buffers,
# right-hand-side partials) kept between calls instead of a fresh `Metal.zeros` per call. Taken and
# given back under a lock, so overlapping assemblies never share a buffer.
const _TEST_FUSED_POOL = Dict{Tuple{DataType,Tuple},Vector{Any}}()
const _TEST_FUSED_POOL_LOCK = ReentrantLock()
_test_fused_pool_on() = get(ENV, "BLAB_TEST_FUSED_POOL", "0") == "1"

function _test_fused_take(::Type{E}, dims...; zero_fill::Bool) where {E}
    _test_fused_pool_on() || return Metal.zeros(E, dims...)
    buffer = lock(_TEST_FUSED_POOL_LOCK) do
        list = get(_TEST_FUSED_POOL, (E, dims), nothing)
        list === nothing || isempty(list) ? nothing : pop!(list)
    end
    buffer === nothing && return Metal.zeros(E, dims...)
    zero_fill && fill!(buffer, zero(E))
    return buffer
end

function _test_fused_give(buffer)
    _test_fused_pool_on() || return Metal.unsafe_free!(buffer)
    lock(_TEST_FUSED_POOL_LOCK) do
        push!(get!(_TEST_FUSED_POOL, (eltype(buffer), size(buffer)), Any[]), buffer)
    end
    return nothing
end

function _launch_metal_fused_singular_kernels!(
    lhs,
    rhs,
    q_neumann,
    regular_cache::MetalRegularAssemblyCache,
    singular_cache::MetalSingularCorrectionCache,
    k,
    transform::SymmetryTransform=SymmetryTransform(:identity, SVector{3,Int}(1, 1, 1), 1),
)
    pair_count = singular_cache.pair_count
    pair_count == 0 && return nothing
    T = typeof(k)
    sx = T(transform.signs[1])
    sy = T(transform.signs[2])
    sz = T(transform.signs[3])
    csx = T(transform.determinant * transform.signs[1])
    csy = T(transform.determinant * transform.signs[2])
    csz = T(transform.determinant * transform.signs[3])
    rule_point_count = length(singular_cache.rule_weights)
    part_count = _metal_singular_part_count()
    # Same maps the four-operator path uses: the fused left-hand side lands on
    # the same P1-row/P1-column cells as the double layer and hypersingular.
    gather_tables = _normalized_metal_singular_writeback() == :gather ?
        _metal_singular_gather_tables(regular_cache, singular_cache, part_count) : nothing
    value_count = pair_count * part_count
    timed = get(ENV, "BLAB_METAL_GATHER_TIMING", "0") == "1"
    timed && Metal.synchronize()
    stamp = time()
    if timed && transform.label == :identity
        _metal_gather_stage_timing["sing_info_pairs"] = pair_count
        _metal_gather_stage_timing["sing_info_parts"] = part_count
        _metal_gather_stage_timing["sing_info_rule_points"] = rule_point_count
    end
    lhs_values = _test_fused_take(eltype(lhs), value_count, 9; zero_fill=false)   # every entry written below
    rhs_values = _test_fused_take(eltype(lhs), value_count, 3; zero_fill=false)
    stamp = _metal_gather_stage!("sing_alloc", timed, stamp)
    if T === Float32 && _test_sing_split_on(regular_cache, k)
        static = _test_singular_static(regular_cache, singular_cache, transform, part_count)
        _metal_launch(
            _test_sing_split_kernel!, value_count,
            lhs_values, rhs_values, static, singular_cache.test_indices, singular_cache.trial_indices,
            singular_cache.normal_products, regular_cache.element_rule_points, regular_cache.rule_points,
            regular_cache.rule_weights, regular_cache.areas, regular_cache.normals, regular_cache.curls,
            k, inv(k), Int32(regular_cache.face_count), Int32(pair_count), Int32(part_count), Val(regular_cache.rule_count),
            sx, sy, sz, csx, csy, csz,
        )
    elseif T === Float32 && _test_sing_packed_mode() != 0
        tables = _test_sing_packed_tables_for(regular_cache, singular_cache)
        packed = _metal_packed_pair_tables_for(regular_cache)
        for (index, (point_count, positions)) in enumerate(tables.groups)
            group_count = length(positions)
            packed_args = (
                lhs_values, rhs_values, positions,
                singular_cache.test_indices, singular_cache.trial_indices, singular_cache.rule_indices,
                singular_cache.jac_scales, singular_cache.normal_products, singular_cache.rule_offsets,
                tables.rule_points4, singular_cache.rule_weights, tables.vertices4, packed.normals4, packed.curls4,
                k, inv(k), Int32(group_count), Int32(pair_count),
                sx, sy, sz, csx, csy, csz,
                Val(point_count), Val(part_count), Val(_test_sing_packed_mode()),
            )
            transform.label == :identity && _test_pipeinfo("sing_packed_n$(point_count)", _test_sing_packed_kernel!, packed_args...)
            _metal_launch(_test_sing_packed_kernel!, group_count * part_count, packed_args...)
        end
    else
    sing_args = (
        lhs_values, rhs_values,
        singular_cache.test_indices, singular_cache.trial_indices, singular_cache.rule_indices,
        singular_cache.jac_scales, singular_cache.normal_products, singular_cache.rule_offsets,
        singular_cache.rule_test_points, singular_cache.rule_trial_points, singular_cache.rule_weights,
        regular_cache.face_vertices, regular_cache.normals, regular_cache.curls,
        k, inv(k), Int32(regular_cache.face_count), Int32(pair_count),
        Int32(rule_point_count), Int32(part_count),
        sx, sy, sz, csx, csy, csz,
        Val(parse(Int, get(ENV, "BLAB_TEST_SING_PROBE", "0"))),
    )
    transform.label == :identity && _test_pipeinfo("sing_fused_bm", _metal_singular_fused_bm_blocks_kernel!, sing_args...)
    _metal_launch(_metal_singular_fused_bm_blocks_kernel!, value_count, sing_args...)
    end
    stamp = _metal_gather_stage!("sing_blocks", timed, stamp)
    if gather_tables === nothing
        _metal_launch(
            _metal_singular_fused_bm_scatter_kernel!,
            pair_count,
            reinterpret(T, lhs),
            reinterpret(T, rhs),
            lhs_values,
            rhs_values,
            q_neumann,
            singular_cache.test_indices,
            singular_cache.trial_indices,
            regular_cache.p1_dofs,
            regular_cache.element_dp0_dofs,
            pair_count,
            part_count,
            regular_cache.p1_dof_count,
            regular_cache.dp0_dof_count,
            size(q_neumann, 2),
            regular_cache.face_count,
        )
    else
        # One thread per touched cell, no atomics, fixed summation order. This is
        # the path the fused Burton-Miller gate depends on for reproducibility.
        block_map = gather_tables.p1_p1
        _metal_launch(
            _metal_singular_entry_gather_kernel!,
            block_map.entry_count,
            lhs, lhs_values,
            block_map.entry_indices, block_map.contrib_offsets, block_map.contrib_values,
            block_map.entry_count, pair_count, part_count,
        )
        rhs_map = gather_tables.rhs
        _metal_launch(
            _metal_singular_rhs_gather_kernel!,
            rhs_map.entry_count,
            rhs, rhs_values, q_neumann,
            rhs_map.entry_indices, rhs_map.contrib_offsets,
            rhs_map.contrib_values, rhs_map.contrib_columns,
            rhs_map.entry_count, pair_count, part_count,
            regular_cache.p1_dof_count, regular_cache.dp0_dof_count, size(q_neumann, 2),
        )
    end
    Metal.synchronize()
    stamp = _metal_gather_stage!("sing_gather", timed, stamp)
    _test_fused_give(lhs_values)
    _test_fused_give(rhs_values)
    return nothing
end

"""
    build_metal_fused_identity_cache(identity_p1_p1, identity_p1_dp0, T)

Device-side form of the two L2 identity blocks for the fused path. Both are
assembled dense but are structurally sparse (a P1 mass matrix), and both are
frequency-independent, so a sweep builds this once. The blocks already carry
the p1 symmetry orbit weights from `assemble_l2_identity_matrix`, which is why
the fused assembly weights only the operator part before adding them.
"""
struct MetalFusedIdentityCache{S,M}
    p1_p1_scatter::S
    p1_dp0::M
end

function build_metal_fused_identity_cache(identity_p1_p1, identity_p1_dp0, ::Type{T}) where {T<:AbstractFloat}
    _require_metal!()
    return MetalFusedIdentityCache(
        build_metal_sparse_scatter_cache(sparse(Complex{T}.(identity_p1_p1))),
        sparse(Complex{T}.(identity_p1_dp0)),
    )
end

function release_metal_fused_identity_cache!(cache::MetalFusedIdentityCache)
    release_metal_sparse_scatter_cache!(cache.p1_p1_scatter)
    return nothing
end

"""
    assemble_burton_miller_neumann_system_metal(mesh, p1_space, dp0_space, q_neumann, k, rule; ...)

Assemble the Burton-Miller Neumann system directly, without ever forming S,
K', D or H. `q_neumann` is `dp0_dof_count x drive_count`; every drive's
right-hand side is accumulated in the same pass, so one assembly serves the
whole channel set at a frequency exactly as one factorization does.

Returns `(matrix, rhs, metadata)` with `matrix` `N x N` and `rhs` `N x drives`,
both Metal-resident. The caller owns and releases them.
"""
function assemble_burton_miller_neumann_system_metal(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    q_neumann,
    k::T,
    rule::TriangleRule{T};
    device_cache,
    singular_cache=nothing,
    device_singular_cache=nothing,
    identity_p1_p1=nothing,
    identity_p1_dp0=nothing,
    identity_cache=nothing,
    skip_singular::Bool=false,
    singular_order::Int=4,
    element_indices=eachindex(mesh.faces),
    symmetry_mode::Symbol=:off,
    timing=nothing,
) where {T<:AbstractFloat}
    _require_metal!()
    # The kernels combine -D + (i/k) H and -S - (i/k) K' with this k, so the
    # signed outgoing wavenumber carries the convention into the coupling too.
    k = outgoing_wavenumber(k)
    device_cache isa MetalRegularAssemblyCache ||
        error("Fused Metal Burton-Miller assembly requires a MetalRegularAssemblyCache.")
    normalized_mode = normalized_symmetry_mode(symmetry_mode)
    device_cache.symmetry_mode == normalized_mode ||
        error("Metal assembly cache symmetry mode $(device_cache.symmetry_mode) does not match requested $(normalized_mode).")
    q_host = q_neumann isa AbstractMatrix ? q_neumann : reshape(q_neumann, :, 1)
    size(q_host, 1) == dp0_space.global_dof_count ||
        error("Fused Metal Burton-Miller assembly needs one Neumann row per DP0 dof.")
    drive_count = size(q_host, 2)
    drive_count >= 1 || error("Fused Metal Burton-Miller assembly needs at least one drive.")
    p1_count = p1_space.global_dof_count

    owns_identity_cache = identity_cache === nothing
    if owns_identity_cache
        (identity_p1_p1 === nothing || identity_p1_dp0 === nothing) &&
            error("Fused Metal Burton-Miller assembly needs identity_cache or both identity blocks.")
        identity_cache = build_metal_fused_identity_cache(identity_p1_p1, identity_p1_dp0, T)
    end
    d_q = q_host isa MtlArray ? q_host : MtlArray(Complex{T}.(q_host))
    owns_q = !(q_host isa MtlArray)
    tables = _metal_fused_gather_tables(device_cache)
    lhs = rhs = rhs_partial = nothing
    succeeded = false
    empty!(_metal_gather_stage_timing)
    try
        storage = metal_operator_storage_mode()
        allocation_elapsed = @elapsed begin
            # The system matrix and right-hand side go to the same storage mode
            # as the four operators, so the host can wrap them in place instead
            # of blitting them through a staging buffer. The pair-block and
            # right-hand-side partials are device-only scratch and stay private.
            lhs = Metal.zeros(Complex{T}, p1_count, p1_count; storage=storage)
            rhs = Metal.zeros(Complex{T}, p1_count, drive_count; storage=storage)
            rhs_partial = _test_fused_take(Complex{T}, p1_count, tables.chunk_size, drive_count; zero_fill=true)
            Metal.synchronize()
        end
        timing !== nothing && (timing["metal_fused_alloc"] = allocation_elapsed)

        singular_mode = _normalized_metal_singular_mode()
        singular_mode == :native ||
            error("Fused Metal Burton-Miller assembly has no host singular mode; unset BLAB_METAL_SINGULAR_MODE.")
        skip_image_singular = !skip_singular
        one_t = one(T)
        kernel_elapsed = @elapsed if get(ENV, "BLAB_TEST_FUSED_IMAGE_ACC", "0") == "1"
            transforms = Any[(device_cache.vertex_offsets, device_cache.incident_elements, Int32(0),
                              one_t, one_t, one_t, one_t, one_t, one_t)]
            for (transform, image_cache) in zip(device_cache.image_transforms, device_cache.image_singular_caches)
                push!(transforms, (image_cache.pair_offsets, image_cache.trial_indices,
                    skip_image_singular ? Int32(1) : Int32(2),
                    T(transform.signs[1]), T(transform.signs[2]), T(transform.signs[3]),
                    T(transform.determinant * transform.signs[1]),
                    T(transform.determinant * transform.signs[2]),
                    T(transform.determinant * transform.signs[3])))
            end
            _launch_metal_fused_pair_kernels!(lhs, rhs_partial, d_q, device_cache, k, transforms)
            Metal.synchronize()
        else
            _launch_metal_fused_pair_kernels!(
                lhs, rhs_partial, d_q, device_cache, k,
                device_cache.vertex_offsets, device_cache.incident_elements, Int32(0),
                one_t, one_t, one_t, one_t, one_t, one_t,
            )
            for (transform, image_cache) in zip(device_cache.image_transforms, device_cache.image_singular_caches)
                _launch_metal_fused_pair_kernels!(
                    lhs, rhs_partial, d_q, device_cache, k,
                    image_cache.pair_offsets, image_cache.trial_indices,
                    skip_image_singular ? Int32(1) : Int32(2),
                    T(transform.signs[1]), T(transform.signs[2]), T(transform.signs[3]),
                    T(transform.determinant * transform.signs[1]),
                    T(transform.determinant * transform.signs[2]),
                    T(transform.determinant * transform.signs[3]),
                )
            end
            Metal.synchronize()
        end
        timing !== nothing && (timing["metal_fused_regular_kernel"] = kernel_elapsed)
        if timing !== nothing
            for (stage, elapsed) in _metal_gather_stage_timing
                timing["metal_fused_" * stage] = elapsed
            end
        end

        reduce_elapsed = @elapsed begin
            _metal_launch(
                _metal_fused_rhs_reduce_kernel!,
                p1_count * drive_count,
                rhs, rhs_partial,
                Int32(p1_count), Int32(tables.chunk_size), Int32(drive_count),
            )
            Metal.synchronize()
        end
        timing !== nothing && (timing["metal_fused_rhs_reduce"] = reduce_elapsed)

        indices = device_cache.element_indices
        correction_cache = singular_cache === nothing ?
            build_singular_correction_cache(mesh, singular_order, indices) : singular_cache
        singular_pairs = 0
        image_singular_pairs = 0
        if !skip_singular
            owns_device_singular_cache = device_singular_cache === nothing
            active_singular_cache = device_singular_cache === nothing ?
                build_metal_singular_correction_cache(correction_cache) : device_singular_cache
            singular_elapsed = @elapsed begin
                _launch_metal_fused_singular_kernels!(lhs, rhs, d_q, device_cache, active_singular_cache, k)
            end
            timing !== nothing && (timing["metal_fused_singular_kernel"] = singular_elapsed)
            singular_pairs = correction_cache.pair_count
            owns_device_singular_cache && release_metal_singular_correction_cache!(active_singular_cache)
            image_elapsed = @elapsed begin
                for (transform, image_cache) in zip(device_cache.image_transforms, device_cache.image_singular_caches)
                    image_cache.pair_count == 0 && continue
                    _launch_metal_fused_singular_kernels!(lhs, rhs, d_q, device_cache, image_cache, k, transform)
                end
                Metal.synchronize()
            end
            timing !== nothing && (timing["metal_fused_image_singular_kernel"] = image_elapsed)
            if timing !== nothing
                for (stage, elapsed) in _metal_gather_stage_timing
                    startswith(stage, "sing_") && (timing["metal_fused_" * stage] = elapsed)
                end
            end
            image_singular_pairs = device_cache.image_singular_pair_count
        end

        # Row weights scale the operator part only, exactly as the four-operator
        # path scales S, K', D and H before the host adds the identity blocks.
        weight_elapsed = @elapsed begin
            if normalized_mode != :off
                d_weights = MtlArray(Complex{T}.(p1_symmetry_orbit_weights(mesh, normalized_mode)))
                lhs .*= reshape(d_weights, :, 1)
                rhs .*= reshape(d_weights, :, 1)
                Metal.synchronize()
                Metal.unsafe_free!(d_weights)
            end
        end
        timing !== nothing && (timing["metal_fused_symmetry_row_weights"] = weight_elapsed)

        # lhs += 0.5 I_p1p1 ; rhs += -0.5 (i/k) I_p1dp0 q. The identity blocks
        # are sparse and the right-hand-side term is one sparse matvec per
        # drive, so both stay cheaper on the host than a kernel launch.
        identity_elapsed = @elapsed begin
            scatter_metal_sparse_to_dense!(lhs, identity_cache.p1_p1_scatter; alpha=Complex{T}(0.5), add=true)
            coupling = burton_miller_coupling(k)
            identity_rhs = (identity_cache.p1_dp0 * Complex{T}.(q_host)) .* (-Complex{T}(0.5) * coupling)
            d_identity_rhs = MtlArray(identity_rhs)
            try
                rhs .+= d_identity_rhs
                Metal.synchronize()
            finally
                Metal.unsafe_free!(d_identity_rhs)
            end
        end
        timing !== nothing && (timing["metal_fused_identity"] = identity_elapsed)

        total_pairs = length(indices) * length(indices)
        image_count = length(device_cache.image_transforms)
        succeeded = true
        return (
            matrix=lhs,
            rhs=rhs,
            regular_pairs=total_pairs - correction_cache.pair_count +
                image_count * total_pairs - image_singular_pairs,
            singular_pairs=singular_pairs,
            image_singular_pairs=image_singular_pairs,
            drive_count=drive_count,
            on_gpu=true,
            gpu_backend=:metal,
            assembly_mode=:metal_fused_burton_miller,
        )
    finally
        rhs_partial === nothing || _test_fused_give(rhs_partial)
        owns_q && Metal.unsafe_free!(d_q)
        owns_identity_cache && release_metal_fused_identity_cache!(identity_cache)
        if !succeeded
            lhs === nothing || Metal.unsafe_free!(lhs)
            rhs === nothing || Metal.unsafe_free!(rhs)
        end
    end
end

function release_metal_burton_miller_system!(system)
    backing = get(system, :metal_backing, nothing)
    if backing !== nothing
        Metal.unsafe_free!(backing.matrix)
        Metal.unsafe_free!(backing.rhs)
        return nothing
    end
    get(system, :on_gpu, false) || return nothing
    Metal.unsafe_free!(system.matrix)
    Metal.unsafe_free!(system.rhs)
    return nothing
end

"""
    metal_host_burton_miller_system(system)

Present the fused system to the host. Shared-storage buffers are wrapped in
place, so this costs nothing and the returned arrays alias device memory; the
returned tuple carries the device arrays under `metal_backing` and owns them,
exactly as `metal_host_operators` does for the four-operator path. Release
once, through whichever tuple you still hold.
"""
function metal_host_burton_miller_system(system)
    get(system, :on_gpu, false) || return system
    Metal.synchronize()
    shared = Metal.is_shared(system.matrix) && Metal.is_shared(system.rhs)
    host = shared ?
        (matrix=unsafe_wrap(Array, system.matrix), rhs=unsafe_wrap(Array, system.rhs)) :
        (matrix=Array(system.matrix), rhs=Array(system.rhs))
    backing = shared ? (matrix=system.matrix, rhs=system.rhs) : nothing
    extras = Base.structdiff(system, NamedTuple{(:matrix, :rhs, :on_gpu, :metal_backing)})
    # Private storage leaves the device tuple the caller's to free; shared
    # storage hands ownership of the wrapped buffers to the returned tuple.
    return merge(extras, host, (on_gpu=false, metal_backing=backing))
end

"""
    solve_metal_burton_miller_system_with_report(system; method=beat_dense_solve_method())

Solve the fused system on the host -- Metal.jl has no GPU LU, and the shared
storage the assembly uses means the host reads the device buffers in place
rather than copying them. Dense LU or diagonally preconditioned GMRES is
chosen by cost model over (dofs, drives); see `beat_solve_dense_system`.

Returns `(pressure, report)` with pressure `N x drives`. The system is left
allocated; the caller releases it.

The LU route factors once and solves every drive against that one
factorization, which is the property the CPU and Metal backends were built to
have. GMRES has no factorization to share and pays per drive, which is exactly
what the router weighs -- and why it is a cost comparison over both dimensions
rather than a dof threshold.
"""
function solve_metal_burton_miller_system_with_report(system; method::Symbol=beat_dense_solve_method())
    host = metal_host_burton_miller_system(system)
    # lu! would overwrite the shared buffer the caller still owns; GMRES reads
    # it and needs no copy at all.
    return beat_solve_dense_system(host.matrix, host.rhs; method=method, preserve_matrix=true)
end

function solve_metal_burton_miller_system(system; method::Symbol=beat_dense_solve_method())
    pressure, _ = solve_metal_burton_miller_system_with_report(system; method=method)
    return pressure
end

# Test (BLAB_TEST_SING_SPLIT=1; quadrature change, judge in dB): singular corrections of the fused
# exterior path as G = G0 + G1, G0 = 1/(4 pi r) (frequency-independent), G1 = (e^{ikr} - 1)/(4 pi r)
# (bounded). Once per singular cache and transform, the Sauter-Schwab rule integrates the G0 parts per
# (pair, part): S0 (3), K'0 (3), D0 (9), I0 = basis products x G0 (9), G0 total (1), 25 Float32.
# Per frequency a small kernel combines them with k and adds the G1 part of each pair on the regular
# R x R rule (part 1 only), so the gathers are unchanged.
const _TEST_SING_STATIC = IdDict{Any,Dict{Tuple{Symbol,Int},Any}}()   # keyed by identity; at most 16 caches kept
const _TEST_SING_STATIC_LOCK = ReentrantLock()
const _TEST_SING_COMPONENTS = 25

function _test_singular_static_kernel!(
    static, test_indices, trial_indices, rule_indices, jac_scales, rule_offsets,
    rule_test_points, rule_trial_points, rule_weights, face_vertices, normals,
    face_count::Int32, pair_count::Int32, rule_point_count::Int32, part_count::Int32,
    trial_sign_x, trial_sign_y, trial_sign_z,
)
    linear_index = Int32(thread_position_in_grid_1d())
    linear_index > pair_count * part_count && return nothing
    pair_position = (linear_index - Int32(1)) % pair_count + Int32(1)
    part = (linear_index - Int32(1)) ÷ pair_count + Int32(1)
    T = Float32
    @inbounds begin
        test_index = Int32(test_indices[pair_position])
        trial_index = Int32(trial_indices[pair_position])
        rule_index = Int32(rule_indices[pair_position])
        q_first = Int32(rule_offsets[rule_index])
        q_last = Int32(rule_offsets[rule_index + Int32(1)]) - Int32(1)
        per_part = cld(q_last - q_first + Int32(1), part_count)
        q = q_first + (part - Int32(1)) * per_part
        q_stop = min(q + per_part - Int32(1), q_last)
        jac_scale = jac_scales[pair_position]
        test_nx = normals[test_index]
        test_ny = normals[test_index + face_count]
        test_nz = normals[test_index + Int32(2) * face_count]
        trial_nx = trial_sign_x * normals[trial_index]
        trial_ny = trial_sign_y * normals[trial_index + face_count]
        trial_nz = trial_sign_z * normals[trial_index + Int32(2) * face_count]
    end
    inv_four_pi = T(0.07957747154594767)
    s0 = zero(SVector{3,T}); k0 = zero(SVector{3,T})
    d0 = zero(SVector{9,T}); i0 = zero(SVector{9,T}); g0 = zero(T)
    while q <= q_stop
        @inbounds begin
            test_xi = rule_test_points[q]
            test_eta = rule_test_points[q + rule_point_count]
            trial_xi = rule_trial_points[q]
            trial_eta = rule_trial_points[q + rule_point_count]
            weight = rule_weights[q] * jac_scale
        end
        tb1 = one(T) - test_xi - test_eta
        rb1 = one(T) - trial_xi - trial_eta
        x, y, z = _metal_face_point(face_vertices, test_index, face_count, tb1, test_xi, test_eta)
        sx, sy, sz = _metal_face_point(face_vertices, trial_index, face_count, rb1, trial_xi, trial_eta)
        dx = sx * trial_sign_x - x
        dy = sy * trial_sign_y - y
        dz = sz * trial_sign_z - z
        radius2 = dx * dx + dy * dy + dz * dz
        if radius2 > zero(T)
            inv_radius = one(T) / sqrt(radius2)
            green = inv_radius * inv_four_pi * weight
            grad = -green * inv_radius
            test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
            trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
            tb = SVector(tb1, test_xi, test_eta)
            outer = SVector(
                tb1 * rb1, test_xi * rb1, test_eta * rb1,
                tb1 * trial_xi, test_xi * trial_xi, test_eta * trial_xi,
                tb1 * trial_eta, test_xi * trial_eta, test_eta * trial_eta,
            )
            s0 += tb * green
            k0 += tb * (grad * test_dot)
            d0 += outer * (grad * trial_dot)
            i0 += outer * green
            g0 += green
        end
        q += Int32(1)
    end
    stride = pair_count * part_count
    @inbounds begin
        for i in 1:3
            static[linear_index + Int32(i - 1) * stride] = s0[i]
            static[linear_index + Int32(i + 2) * stride] = k0[i]
        end
        for i in 1:9
            static[linear_index + Int32(i + 5) * stride] = d0[i]
            static[linear_index + Int32(i + 14) * stride] = i0[i]
        end
        static[linear_index + Int32(24) * stride] = g0
    end
    return nothing
end

function _test_singular_static(regular_cache, singular_cache, transform, part_count)
    key = (transform.label, part_count)
    lock(_TEST_SING_STATIC_LOCK) do
        if !haskey(_TEST_SING_STATIC, singular_cache.test_indices) && length(_TEST_SING_STATIC) >= 16
            for (_, old) in _TEST_SING_STATIC, (_, buffer) in old
                Metal.unsafe_free!(buffer)
            end
            empty!(_TEST_SING_STATIC)
        end
        per_cache = get!(() -> Dict{Tuple{Symbol,Int},Any}(), _TEST_SING_STATIC, singular_cache.test_indices)
        get!(per_cache, key) do
            pair_count = singular_cache.pair_count
            static = Metal.zeros(Float32, pair_count * part_count * _TEST_SING_COMPONENTS)
            _metal_launch(
                _test_singular_static_kernel!, pair_count * part_count, static,
                singular_cache.test_indices, singular_cache.trial_indices, singular_cache.rule_indices,
                singular_cache.jac_scales, singular_cache.rule_offsets,
                singular_cache.rule_test_points, singular_cache.rule_trial_points, singular_cache.rule_weights,
                regular_cache.face_vertices, regular_cache.normals,
                Int32(regular_cache.face_count), Int32(pair_count), Int32(length(singular_cache.rule_weights)), Int32(part_count),
                Float32(transform.signs[1]), Float32(transform.signs[2]), Float32(transform.signs[3]),
            )
            Metal.synchronize()
            static
        end
    end
end

# G1 = (e^{ikr} - 1)/(4 pi r) and its r-derivative, both bounded at r = 0, without Float32 cancellation:
# cos(kr) - 1 = -2 sin^2(kr/2); kr - sin(kr) by its series for small kr.
@inline function _test_g1(radius, k, scale)
    phase = k * radius
    half = 0.5f0 * phase
    sh = sin(half)
    g1_re = -2.0f0 * sh * sh * scale                       # scale = weight / (4 pi r)
    g1_im = sin(phase) * scale
    # d/dr: ik (G1 + G0) - G1 / r
    #   re: -k g1_im - g1_re / r      im: k g1_re + (k r - sin(kr)) scale / r
    kms = phase < 0.05f0 ? phase * phase * phase * (1.0f0 / 6.0f0 - phase * phase * (1.0f0 / 120.0f0)) : phase - sin(phase)
    return g1_re, g1_im, kms
end

function _test_sing_split_kernel!(
    lhs_values, rhs_values, static, test_indices, trial_indices, normal_products,
    element_rule_points, rule_points, rule_weights, areas, normals, curls,
    k::Float32, inverse_k::Float32, face_count::Int32, pair_count::Int32, part_count::Int32, ::Val{R},
    trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
) where {R}
    linear_index = Int32(thread_position_in_grid_1d())
    linear_index > pair_count * part_count && return nothing
    pair_position = (linear_index - Int32(1)) % pair_count + Int32(1)
    part = (linear_index - Int32(1)) ÷ pair_count + Int32(1)
    stride = pair_count * part_count
    T = Float32
    inv_four_pi = T(0.07957747154594767)
    @inbounds begin
        test_index = Int32(test_indices[pair_position])
        trial_index = Int32(trial_indices[pair_position])
        normal_product = normal_products[pair_position]
        s0 = SVector{3,T}(ntuple(i -> static[linear_index + Int32(i - 1) * stride], Val(3)))
        k0 = SVector{3,T}(ntuple(i -> static[linear_index + Int32(i + 2) * stride], Val(3)))
        d0 = SVector{9,T}(ntuple(i -> static[linear_index + Int32(i + 5) * stride], Val(9)))
        i0 = SVector{9,T}(ntuple(i -> static[linear_index + Int32(i + 14) * stride], Val(9)))
        g0 = static[linear_index + Int32(24) * stride]
    end
    curl_scale = inverse_k * k * k * normal_product
    # G0 part: rhs = -S0 - (i/k) K'0; lhs = -D0 - i curl_scale I0 (+ curl term below).
    rhs_re = -s0
    rhs_im = -inverse_k * k0
    lhs_re = -d0
    lhs_im = -curl_scale * i0
    g_total_re = g0
    g_total_im = zero(T)
    if part == Int32(1)
        @inbounds begin
            test_nx = normals[test_index]
            test_ny = normals[test_index + face_count]
            test_nz = normals[test_index + Int32(2) * face_count]
            trial_nx = trial_sign_x * normals[trial_index]
            trial_ny = trial_sign_y * normals[trial_index + face_count]
            trial_nz = trial_sign_z * normals[trial_index + Int32(2) * face_count]
            jac_scale = T(4) * areas[test_index] * areas[trial_index]
        end
        tq = Int32(1)
        while tq <= Int32(R)
            @inbounds begin
                test_xi = rule_points[tq]
                test_eta = rule_points[tq + Int32(R)]
                tw = rule_weights[tq]
                pi_ = test_index + face_count * (tq - Int32(1))
                x = element_rule_points[pi_]
                y = element_rule_points[pi_ + face_count * Int32(R)]
                z = element_rule_points[pi_ + face_count * Int32(2 * R)]
            end
            tb1 = one(T) - test_xi - test_eta
            tb = SVector(tb1, test_xi, test_eta)
            rq = Int32(1)
            while rq <= Int32(R)
                @inbounds begin
                    trial_xi = rule_points[rq]
                    trial_eta = rule_points[rq + Int32(R)]
                    weight = tw * rule_weights[rq] * jac_scale
                    pj = trial_index + face_count * (rq - Int32(1))
                    sx = element_rule_points[pj] * trial_sign_x
                    sy = element_rule_points[pj + face_count * Int32(R)] * trial_sign_y
                    sz = element_rule_points[pj + face_count * Int32(2 * R)] * trial_sign_z
                end
                rb1 = one(T) - trial_xi - trial_eta
                outer = SVector(
                    tb1 * rb1, test_xi * rb1, test_eta * rb1,
                    tb1 * trial_xi, test_xi * trial_xi, test_eta * trial_xi,
                    tb1 * trial_eta, test_xi * trial_eta, test_eta * trial_eta,
                )
                dx = sx - x
                dy = sy - y
                dz = sz - z
                radius2 = dx * dx + dy * dy + dz * dz
                if radius2 > zero(T)
                    radius = sqrt(radius2)
                    inv_radius = one(T) / radius
                    scale = inv_radius * inv_four_pi * weight
                    g1_re, g1_im, kms = _test_g1(radius, k, scale)
                    grad_re = -k * g1_im - g1_re * inv_radius
                    grad_im = k * g1_re + kms * scale * inv_radius
                    test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
                    trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
                else
                    # r = 0: G1 = ik/(4 pi); the gradient terms carry n.(x - y)/r = 0 on a flat element.
                    g1_re = zero(T)
                    g1_im = k * inv_four_pi * weight
                    grad_re = zero(T); grad_im = zero(T); test_dot = zero(T); trial_dot = zero(T)
                end
                rhs_re += tb * (-g1_re + inverse_k * (grad_im * test_dot))
                rhs_im += tb * (-g1_im - inverse_k * (grad_re * test_dot))
                u_re = -(grad_re * trial_dot) + curl_scale * g1_im
                u_im = -(grad_im * trial_dot) - curl_scale * g1_re
                lhs_re += outer * u_re
                lhs_im += outer * u_im
                g_total_re += g1_re
                g_total_im += g1_im
                rq += Int32(1)
            end
            tq += Int32(1)
        end
    end
    curl_products = _metal_pair_curl_products(
        curls, test_index, trial_index, face_count, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    )
    lhs_re -= curl_products * (inverse_k * g_total_im)
    lhs_im += curl_products * (inverse_k * g_total_re)
    @inbounds begin
        i = 1
        while i <= 3
            rhs_values[linear_index + Int32(i - 1) * stride] = Complex(rhs_re[i], rhs_im[i])
            i += 1
        end
        i = 1
        while i <= 9
            lhs_values[linear_index + Int32(i - 1) * stride] = Complex(lhs_re[i], lhs_im[i])
            i += 1
        end
    end
    return nothing
end

# BLAB_TEST_SING_SPLIT=<kappa>: split only while k h_max < kappa (h_max: the largest element's
# equilateral edge from its area), where the regular rule still integrates G1 accurately; above it the
# stock Sauter-Schwab evaluation runs.
const _TEST_SING_HMAX = IdDict{Any,Float32}()
function _test_sing_split_on(regular_cache, k)
    raw = get(ENV, "BLAB_TEST_SING_SPLIT", "0")
    raw == "0" && return false
    kappa = parse(Float32, raw)
    hmax = lock(_TEST_SING_STATIC_LOCK) do
        get!(_TEST_SING_HMAX, regular_cache.areas) do
            Float32(sqrt(4 * maximum(Array(regular_cache.areas)) / sqrt(3)))
        end
    end
    return abs(k) * hmax < kappa
end

# BLAB_TEST_FAR_ORDER tables: the 3-point rule's points per face (float4, face-major), and per face the
# centroid and circumradius (largest vertex distance from the centroid).
const _TEST_RULE3 = (1.0f0 / 6, 2.0f0 / 3, 1.0f0 / 6, 1.0f0 / 6, 1.0f0 / 6, 2.0f0 / 3, 1.0f0 / 6, 1.0f0 / 6, 1.0f0 / 6)
struct TestFarOrderTables
    points3
    centroids4
    rule3_points
    rule3_weights
end
const _TEST_FAR_ORDER = IdDict{Any,TestFarOrderTables}()
function _test_far_order_tables_for(cache::MetalRegularAssemblyCache)
    lock(_TEST_SING_STATIC_LOCK) do
        get!(_TEST_FAR_ORDER, cache.face_vertices) do
            F = cache.face_count
            fv = Array(cache.face_vertices)
            vertex(face, v) = (Float64(fv[face + (3 * (v - 1)) * F]), Float64(fv[face + (3 * (v - 1) + 1) * F]),
                               Float64(fv[face + (3 * (v - 1) + 2) * F]))
            points3 = Vector{_MetalFloat4}(undef, 3 * F)
            centroids4 = Vector{_MetalFloat4}(undef, F)
            for face in 1:F
                a, b, c = vertex(face, 1), vertex(face, 2), vertex(face, 3)
                for q in 1:3
                    xi, eta = Float64(_TEST_RULE3[q]), Float64(_TEST_RULE3[3 + q])
                    b1 = 1 - xi - eta
                    p = ntuple(i -> b1 * a[i] + xi * b[i] + eta * c[i], 3)
                    points3[(face - 1) * 3 + q] = _metal_float4(Float32(p[1]), Float32(p[2]), Float32(p[3]), 0.0f0)
                end
                ce = ntuple(i -> (a[i] + b[i] + c[i]) / 3, 3)
                radius = maximum(sqrt(sum((v[i] - ce[i])^2 for i in 1:3)) for v in (a, b, c))
                centroids4[face] = _metal_float4(Float32(ce[1]), Float32(ce[2]), Float32(ce[3]), Float32(radius))
            end
            TestFarOrderTables(MtlArray(points3), MtlArray(centroids4),
                MtlArray(collect(_TEST_RULE3[1:6])), MtlArray(collect(_TEST_RULE3[7:9])))
        end
    end
end
