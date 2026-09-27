# Packed pair tables shared by the pair_tilereduce kernels, the fused exterior
# Burton-Miller assembly and the fast field evaluation.
#
# The pair maths is the same arithmetic as `_metal_regular_pair_blocks_inbounds`,
# but the rule points, weights and basis values are compile-time constants (a
# Val tuple), and each quadrature point, normal and curl row is one float4 load:
#   points4[(face - 1) * R + q] = (x, y, z, 0)
#   normals4[face]              = (nx, ny, nz, 0)
#   curls4[(face - 1) * 3 + a]  = curl of basis a, (x, y, z, 0)
# Values are copied, not recomputed, so the results are bit-identical to the
# unpacked kernels.

const _MetalFloat4 = NTuple{4,VecElement{Float32}}

@inline _metal_float4(a, b, c, d) = (VecElement(a), VecElement(b), VecElement(c), VecElement(d))

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

@inline function _metal_packed_test_point(acc, context, ::Val{Q}, rc, rv::Val{R}) where {Q,R}
    slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im, g_total_re, g_total_im = acc
    k, inv_four_pi, jac_scale, test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        points4, test_index, trial_index = context
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
        test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs, points4, trial_index)
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
        points4, test_index, trial_index)
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
