# pair_gather_v4: the pair_gather path with the 48 block components packed as
# 12 float4 groups, so a gather visit is one 16-byte load instead of four
# scalar loads to four different component planes.
#
#   group c - 1, c = 1..9 : (D re, D im, H re, H im) of 3x3 entry c (column-major)
#   group 8 + lr, lr = 1..3: (S re, S im, K' re, K' im) of local row lr
#
# Layout [group][trial local][test position] in float4 units, so consecutive
# test positions of a tile still store to consecutive addresses. Tables and
# summation order are those of pair_gather, so the results are bit-identical.

const _MetalFloat4 = NTuple{4,VecElement{Float32}}
const _METAL_GATHER_V4_GROUPS = 12

@inline _metal_float4(a, b, c, d) = (VecElement(a), VecElement(b), VecElement(c), VecElement(d))

function _metal_gather_v4_blocks(tables::MetalGatherTables)
    # Reuses the pair_gather buffer: 48 floats per pair either way.
    return reinterpret(_MetalFloat4, tables.blocks)
end

function _metal_regular_pair_blocks_v4_kernel!(
    blocks4,
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
    group_stride::Int32,
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
    base = test_position + element_count * (trial_local - Int32(1))   # 1-based, group 0
    if _metal_pair_is_skipped(
        faces,
        face_count,
        test_index,
        trial_index,
        pair_offsets,
        singular_trial_indices,
        skip_mode,
    )
        zero4 = _metal_float4(0.0f0, 0.0f0, 0.0f0, 0.0f0)
        group = Int32(0)
        while group < Int32(_METAL_GATHER_V4_GROUPS)
            @inbounds blocks4[base + group * group_stride] = zero4
            group += Int32(1)
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
    @inbounds for c in 1:9
        blocks4[base + Int32(c - 1) * group_stride] = _metal_float4(dlp_re[c], dlp_im[c], hyp_re[c], hyp_im[c])
    end
    @inbounds for lr in 1:3
        blocks4[base + Int32(8 + lr) * group_stride] = _metal_float4(slp_re[lr], slp_im[lr], adj_re[lr], adj_im[lr])
    end
    return nothing
end

function _metal_gather_slp_adjoint_v4_kernel!(
    single_layer,
    adjoint_double_layer,
    blocks4,
    elements,
    element_positions,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    element_dp0_dofs,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    group_stride::Int32,
    p1_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * chunk_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    trial_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    column_base = element_count * (trial_local - Int32(1))
    s_re = 0.0f0
    s_im = 0.0f0
    a_re = 0.0f0
    a_im = 0.0f0
    @inbounds incident_position = Int32(vertex_offsets[row])
    @inbounds incident_stop = Int32(vertex_offsets[row + Int32(1)]) - Int32(1)
    while incident_position <= incident_stop
        @inbounds test_position = Int32(element_positions[incident_elements[incident_position]])
        @inbounds local_row = Int32(incident_local_indices[incident_position])
        @inbounds v = blocks4[test_position + column_base + (local_row + Int32(8)) * group_stride]
        s_re += v[1].value
        s_im += v[2].value
        a_re += v[3].value
        a_im += v[4].value
        incident_position += Int32(1)
    end
    @inbounds dp0_column = Int32(element_dp0_dofs[trial_index])
    operator_index = row + (dp0_column - Int32(1)) * p1_count
    @inbounds single_layer[operator_index] += Complex(s_re, s_im)
    @inbounds adjoint_double_layer[operator_index] += Complex(a_re, a_im)
    return nothing
end

function _metal_gather_dlp_hyp_v4_kernel!(
    double_layer,
    hypersingular,
    blocks4,
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
    group_stride::Int32,
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
    d_re = 0.0f0
    d_im = 0.0f0
    h_re = 0.0f0
    h_im = 0.0f0
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
            group = local_row - Int32(1) + Int32(3) * (local_column - Int32(1))   # 0..8
            @inbounds v = blocks4[test_position + element_count * (trial_local - Int32(1)) + group * group_stride]
            d_re += v[1].value
            d_im += v[2].value
            h_re += v[3].value
            h_im += v[4].value
            chunk_position += Int32(1)
        end
        incident_position += Int32(1)
    end
    operator_index = row + (column - Int32(1)) * p1_count
    @inbounds double_layer[operator_index] += Complex(d_re, d_im)
    @inbounds hypersingular[operator_index] += Complex(h_re, h_im)
    return nothing
end

function _launch_metal_gather_v4_pair_kernels!(
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
    k isa Float32 || error("pair_gather_v4 supports Float32 only.")
    tables = _metal_gather_tables(cache)
    blocks4 = _metal_gather_v4_blocks(tables)
    _test_dump_gather_mesh(cache)
    packed = get(ENV, "BLAB_METAL_V4_PACKED", "0") == "1" ? _metal_packed_pair_tables_for(cache) : nothing
    chunk_size = tables.chunk_size
    group_stride = Int32(element_count * chunk_size)
    tile_x, tile_y = _metal_atomic_tile()
    groupsize = _metal_kernel_groupsize()
    p1_count = Int32(cache.p1_dof_count)
    timed = get(ENV, "BLAB_METAL_GATHER_TIMING", "0") == "1"
    timed && Metal.synchronize()
    stamp = time()
    for chunk in 1:tables.chunk_count
        chunk_start = (chunk - 1) * chunk_size + 1
        chunk_count = min(chunk_size, element_count - chunk_start + 1)
        if packed !== nothing
            Metal.@metal threads=(tile_x, tile_y) groups=(cld(element_count, tile_x), cld(chunk_count, tile_y)) _metal_regular_pair_blocks_v4p_kernel!(
                blocks4,
                packed.points4,
                packed.normals4,
                cache.areas,
                packed.curls4,
                cache.faces,
                tables.elements,
                Int32(element_count),
                Int32(chunk_start),
                Int32(chunk_count),
                group_stride,
                k,
                Int32(cache.face_count),
                Val(packed.rule),
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
        else
        Metal.@metal threads=(tile_x, tile_y) groups=(cld(element_count, tile_x), cld(chunk_count, tile_y)) _metal_regular_pair_blocks_v4_kernel!(
            blocks4,
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
            group_stride,
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
        end
        stamp = _metal_gather_stage!("pairs", timed, stamp)
        _metal_launch(
            _metal_gather_slp_adjoint_v4_kernel!,
            cache.p1_dof_count * chunk_count,
            operators.single_layer,
            operators.adjoint_double_layer,
            blocks4,
            tables.elements,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            cache.element_dp0_dofs,
            Int32(element_count),
            Int32(chunk_start),
            Int32(chunk_count),
            group_stride,
            p1_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("slp_adjoint", timed, stamp)
        node_start = tables.chunk_node_offsets[chunk]
        node_count = tables.chunk_node_offsets[chunk + 1] - node_start
        _metal_launch(
            _metal_gather_dlp_hyp_v4_kernel!,
            cache.p1_dof_count * node_count,
            operators.double_layer,
            operators.hypersingular,
            blocks4,
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
            group_stride,
            p1_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("dlp_hyp", timed, stamp)
    end
    return nothing
end

function _launch_metal_regular_gather_v4_kernels!(operators, cache::MetalRegularAssemblyCache, k)
    return _launch_metal_gather_v4_pair_kernels!(
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

function _launch_metal_symmetry_regular_gather_v4_kernels!(
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
    return _launch_metal_gather_v4_pair_kernels!(
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

# Test hook: BLAB_TEST_DUMP_GATHER_MESH=<file> writes the element order and the
# P1 dofs once (text: element_count, face_count, then per position "element d1 d2 d3").
function _test_dump_gather_mesh(cache::MetalRegularAssemblyCache)
    path = get(ENV, "BLAB_TEST_DUMP_GATHER_MESH", "")
    (isempty(path) || isfile(path)) && return nothing
    p1_dofs = Array(cache.p1_dofs)
    open(path, "w") do io
        println(io, length(cache.element_indices), " ", cache.face_count)
        for element in cache.element_indices
            println(io, element, " ", p1_dofs[element, 1], " ", p1_dofs[element, 2], " ", p1_dofs[element, 3])
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Load diet for the pair maths (BLAB_METAL_V4_PACKED=1, pair_gather_v4 only).
# The same arithmetic as `_metal_regular_pair_blocks_inbounds`, but the rule
# points, weights and basis values are compile-time constants (a Val tuple),
# and each quadrature point, normal and curl row is one float4 load:
#   points4[(face - 1) * R + q] = (x, y, z, 0)
#   normals4[face]              = (nx, ny, nz, 0)
#   curls4[(face - 1) * 3 + a]  = curl of basis a, (x, y, z, 0)
# Values are copied, not recomputed, so the results are bit-identical.

struct MetalPackedPairTables
    points4
    normals4
    curls4
    rule::Tuple   # (xi_1..xi_R, eta_1..eta_R, w_1..w_R), Float32
end

const _metal_packed_pair_tables = WeakKeyDict{Any,MetalPackedPairTables}()

function _metal_packed_pair_tables_for(cache::MetalRegularAssemblyCache)
    # Keyed by the cache's mutable gather_tables Ref, so it lives as long as the cache.
    get!(_metal_packed_pair_tables, cache.gather_tables) do
        R = cache.rule_count
        F = cache.face_count
        erp = Array(cache.element_rule_points)   # flat: face + F*(q-1) + F*R*(c-1)
        nrm = Array(cache.normals)
        crl = Array(cache.curls)
        points4 = Vector{_MetalFloat4}(undef, F * R)
        normals4 = Vector{_MetalFloat4}(undef, F)
        curls4 = Vector{_MetalFloat4}(undef, F * 3)
        for face in 1:F
            for q in 1:R
                p = face + F * (q - 1)
                points4[(face - 1) * R + q] = _metal_float4(erp[p], erp[p + F * R], erp[p + 2 * F * R], 0.0f0)
            end
            normals4[face] = _metal_float4(nrm[face], nrm[face + F], nrm[face + 2F], 0.0f0)
            for a in 1:3
                b = face + 3 * (a - 1) * F
                curls4[(face - 1) * 3 + a] = _metal_float4(crl[b], crl[b + F], crl[b + 2F], 0.0f0)
            end
        end
        rp = Array(cache.rule_points)
        rw = Array(cache.rule_weights)
        rule = (Float32.(rp[1:R])..., Float32.(rp[R+1:2R])..., Float32.(rw[1:R])...)
        MetalPackedPairTables(MtlArray(points4), MtlArray(normals4), MtlArray(curls4), rule)
    end
end

function _release_metal_packed_pair_tables!(cache::MetalRegularAssemblyCache)
    tables = pop!(_metal_packed_pair_tables, cache.gather_tables, nothing)
    tables === nothing && return nothing
    Metal.unsafe_free!(tables.points4)
    Metal.unsafe_free!(tables.normals4)
    Metal.unsafe_free!(tables.curls4)
    return nothing
end

@inline _metal_rule_xi(::Val{RC}, ::Val{R}, q) where {RC,R} = RC[q]
@inline _metal_rule_eta(::Val{RC}, ::Val{R}, q) where {RC,R} = RC[R + q]
@inline _metal_rule_w(::Val{RC}, ::Val{R}, q) where {RC,R} = RC[2R + q]

@inline _metal_packed_trial_fold(acc, context, ::Val{0}, rc, rv) = acc
@inline function _metal_packed_trial_fold(acc, context, ::Val{N}, rc, rv) where {N}
    acc = _metal_packed_trial_fold(acc, context, Val(N - 1), rc, rv)
    return _metal_packed_trial_term(acc, context, Val(N), rc, rv)
end

@inline function _metal_packed_trial_term(acc, context, ::Val{Q}, rc, rv::Val{R}) where {Q,R}
    s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = acc
    x, y, z, test_weight_scale, k, inv_four_pi,
        test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        points4, trial_index = context
    xi = _metal_rule_xi(rc, rv, Q)
    eta = _metal_rule_eta(rc, rv, Q)
    rb1 = one(k) - xi - eta
    trial_weight = _metal_rule_w(rc, rv, Q)
    @inbounds p = points4[(trial_index - Int32(1)) * Int32(R) + Int32(Q)]
    sx = p[1].value
    sy = p[2].value
    sz = p[3].value
    Base.@fastmath begin
    dx = sx * trial_signs[1] - x
    dy = sy * trial_signs[2] - y
    dz = sz * trial_signs[3] - z
    radius2 = dx * dx + dy * dy + dz * dz
    if radius2 > zero(k)
        rb = SVector(rb1, xi, eta)
        inv_radius = _metal_fast_rsqrt(radius2)
        radius = radius2 * inv_radius
        phase = k * radius
        green_scale = inv_radius * inv_four_pi * (test_weight_scale * trial_weight)
        if context[16] === Val(3)   # test probe (BLAB_TEST_TR_PROBE=3): no sin/cos, timing only
            green_re = (one(k) - phase * phase * 0.5f0) * green_scale
            green_im = phase * green_scale
        else
            green_re = _metal_fast_cos(phase) * green_scale
            green_im = _metal_fast_sin(phase) * green_scale
        end
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

@inline function _metal_packed_test_point(acc, context, ::Val{Q}, rc, rv::Val{R}) where {Q,R}
    slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im, g_total_re, g_total_im = acc
    k, inv_four_pi, jac_scale, test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        points4, test_index, trial_index = context
    probe = context[14]
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
    s_re = zero(k)
    s_im = zero(k)
    a_re = zero(k)
    a_im = zero(k)
    d_re = zero(SVector{3,T})
    d_im = zero(SVector{3,T})
    h_re = zero(SVector{3,T})
    h_im = zero(SVector{3,T})
    trial_context = (x, y, z, test_weight * jac_scale, k, inv_four_pi,
        test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs, points4, trial_index, probe)
    s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im = _metal_packed_trial_fold(
        (s_re, s_im, a_re, a_im, d_re, d_im, h_re, h_im), trial_context, Val(R), rc, rv)
    slp_re += test_basis * s_re
    slp_im += test_basis * s_im
    adj_re += test_basis * a_re
    adj_im += test_basis * a_im
    g_total_re += s_re
    g_total_im += s_im
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
    return (slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im, g_total_re, g_total_im)
end

@inline _metal_packed_test_fold(acc, context, ::Val{0}, rc, rv) = acc
@inline function _metal_packed_test_fold(acc, context, ::Val{N}, rc, rv) where {N}
    acc = _metal_packed_test_fold(acc, context, Val(N - 1), rc, rv)
    return _metal_packed_test_point(acc, context, Val(N), rc, rv)
end

@inline function _metal_regular_pair_blocks_packed(
    points4, normals4, areas, curls4,
    test_index::Int32, trial_index::Int32, k, rc, rv::Val{R},
    trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    probe::Val=Val(0),
) where {R}
    T = typeof(k)
    inv_four_pi = T(0.07957747154594767)
    k2 = k * k
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
    context = (k, inv_four_pi, jac_scale, test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        points4, test_index, trial_index, probe)
    acc = (zero(SVector{3,T}), zero(SVector{3,T}), zero(SVector{3,T}), zero(SVector{3,T}),
        zero(SVector{9,T}), zero(SVector{9,T}), zero(SVector{9,T}), zero(SVector{9,T}), zero(k), zero(k))
    slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im, g_total_re, g_total_im =
        _metal_packed_test_fold(acc, context, Val(R), rc, rv)
    k2n = k2 * normal_product
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
    hyp_re = curl_products * g_total_re - hyp_re * k2n
    hyp_im = curl_products * g_total_im - hyp_im * k2n
    return slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im
end

function _metal_regular_pair_blocks_v4p_kernel!(
    blocks4,
    points4,
    normals4,
    areas,
    curls4,
    faces,
    elements,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    group_stride::Int32,
    k,
    face_count::Int32,
    rc::Val{RC},
    rv::Val{R},
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
) where {RC,R}
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
        zero4 = _metal_float4(0.0f0, 0.0f0, 0.0f0, 0.0f0)
        group = Int32(0)
        while group < Int32(_METAL_GATHER_V4_GROUPS)
            @inbounds blocks4[base + group * group_stride] = zero4
            group += Int32(1)
        end
        return nothing
    end
    slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im = _metal_regular_pair_blocks_packed(
        points4, normals4, areas, curls4, test_index, trial_index, k, rc, rv,
        trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
    )
    @inbounds for c in 1:9
        blocks4[base + Int32(c - 1) * group_stride] = _metal_float4(dlp_re[c], dlp_im[c], hyp_re[c], hyp_im[c])
    end
    @inbounds for lr in 1:3
        blocks4[base + Int32(8 + lr) * group_stride] = _metal_float4(slp_re[lr], slp_im[lr], adj_re[lr], adj_im[lr])
    end
    return nothing
end
