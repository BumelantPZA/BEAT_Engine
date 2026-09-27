# Fast Metal field evaluation (BLAB_METAL_FIELD_FAST=1, Float32 only): the same sums as
# `evaluate_galerkin_field_metal`, with fast rsqrt/sin/cos (as the regular assembly kernels
# use), Int32 indexing, and the per-source data packed into float4s:
#   points4[s]  = (x, y, z, 0)       normals4[s] = (nx, ny, nz, 0)
#   weights4[s] = (p re, p im, q re, q im)  (weighted pressure and Neumann data, per call)

struct MetalFastFieldTables
    points4
    normals4
end

# Identity-keyed (an MtlArray key would be hashed by content); freed with the field cache.
const _metal_fast_field_tables = IdDict{Any,MetalFastFieldTables}()
const _metal_fast_field_lock = ReentrantLock()

function _release_metal_fast_field_tables!(cache::MetalFieldEvaluationCache)
    tables = lock(() -> pop!(_metal_fast_field_tables, cache.source_points, nothing), _metal_fast_field_lock)
    tables === nothing && return nothing
    Metal.unsafe_free!(tables.points4)
    Metal.unsafe_free!(tables.normals4)
    return nothing
end

function _metal_fast_field_tables_for(cache::MetalFieldEvaluationCache)
    lock(_metal_fast_field_lock) do
    get!(_metal_fast_field_tables, cache.source_points) do
        n = cache.source_count
        points = Array(cache.source_points)
        normals = Array(cache.source_normals)
        MetalFastFieldTables(
            MtlArray([_metal_float4(points[s, 1], points[s, 2], points[s, 3], 0.0f0) for s in 1:n]),
            MtlArray([_metal_float4(normals[s, 1], normals[s, 2], normals[s, 3], 0.0f0) for s in 1:n]),
        )
    end
    end
end

function _metal_fast_field_sources_kernel!(
    weights4, pressure, q_neumann, source_weights, source_faces, source_elements, basis_values, source_count::Int32,
)
    s = Int32(thread_position_in_grid_1d())
    s > source_count && return nothing
    @inbounds begin
        face1 = Int32(source_faces[s])
        face2 = Int32(source_faces[s + source_count])
        face3 = Int32(source_faces[s + Int32(2) * source_count])
        basis1 = basis_values[s]
        basis2 = basis_values[s + source_count]
        basis3 = basis_values[s + Int32(2) * source_count]
        weight = source_weights[s]
        p = (basis1 * pressure[face1] + basis2 * pressure[face2] + basis3 * pressure[face3]) * weight
        q = q_neumann[Int32(source_elements[s])] * weight
        weights4[s] = _metal_float4(real(p), imag(p), real(q), imag(q))
    end
    return nothing
end

# MODE 1: fast sin/cos on the raw phase. MODE 2: precise sin/cos. MODE 3: fast sin/cos after
# Cody-Waite reduction of the phase to [-pi, pi] (the fast intrinsics are accurate only there).
# MODE 4: MODE 3 with the distance from a precise sqrt. The phase k*r reaches ~1e3 rad at 20 kHz and
# 3 m, where the fast rsqrt's relative error becomes ~1e-3 rad per term (SAWMOD: 0.035 dB at 17-20 kHz
# against the stock field, over the 0.01 dB target).
@inline _metal_field_cis(phase, ::Val{4}) = _metal_field_cis(phase, Val(3))
@inline _metal_precise_sqrt(x::Float32) = sqrt(x)   # called from a @fastmath block: not rewritten
@inline _metal_field_cis(phase, ::Val{1}) = (_metal_fast_cos(phase), _metal_fast_sin(phase))
@inline _metal_field_cis(phase, ::Val{2}) = (cos(phase), sin(phase))
@inline function _metal_field_cis(phase, ::Val{3})
    turns = round(phase * 0.15915494f0)
    reduced = fma(-turns, 6.2831855f0, phase) + turns * 1.7484555f-7   # fma keeps turns * hi exact
    return (_metal_fast_cos(reduced), _metal_fast_sin(reduced))
end

function _metal_fast_field_kernel!(
    partials, eval_points, points4, normals4, weights4, k::Float32,
    source_count::Int32, point_count::Int32, chunk_length::Int32, chunk_count::Int32, ::Val{MODE},
) where {MODE}
    linear_index = Int32(thread_position_in_grid_1d())
    linear_index > point_count * chunk_count && return nothing
    point_index = (linear_index - Int32(1)) % point_count + Int32(1)
    chunk_index = (linear_index - Int32(1)) ÷ point_count
    source_start = chunk_index * chunk_length + Int32(1)
    source_stop = min(source_count, source_start + chunk_length - Int32(1))
    inv_four_pi = 0.07957747154594767f0
    @inbounds x1 = eval_points[point_index]
    @inbounds x2 = eval_points[point_index + point_count]
    @inbounds x3 = eval_points[point_index + Int32(2) * point_count]
    potential_re = 0.0f0
    potential_im = 0.0f0
    s = source_start
    while s <= source_stop
        @inbounds sp = points4[s]
        Base.@fastmath begin
        r1 = sp[1].value - x1
        r2 = sp[2].value - x2
        r3 = sp[3].value - x3
        radius2 = r1 * r1 + r2 * r2 + r3 * r3
        if radius2 > 0.0f0
            @inbounds sn = normals4[s]
            @inbounds w = weights4[s]
            if MODE == 4
                radius = _metal_precise_sqrt(radius2)
                inv_radius = 1.0f0 / radius
            else
                inv_radius = _metal_fast_rsqrt(radius2)
                radius = radius2 * inv_radius
            end
            phase = k * radius
            green_scale = inv_radius * inv_four_pi
            c, sn_ = _metal_field_cis(phase, Val(MODE))
            green_re = c * green_scale
            green_im = sn_ * green_scale
            normal_projection = (r1 * sn[1].value + r2 * sn[2].value + r3 * sn[3].value) * inv_radius
            double_re = (-green_re * inv_radius - green_im * k) * normal_projection
            double_im = (green_re * k - green_im * inv_radius) * normal_projection
            p_re = w[1].value
            p_im = w[2].value
            q_re = w[3].value
            q_im = w[4].value
            potential_re += double_re * p_re - double_im * p_im - green_re * q_re + green_im * q_im
            potential_im += double_re * p_im + double_im * p_re - green_re * q_im - green_im * q_re
        end
        end
        s += Int32(1)
    end
    @inbounds partials[linear_index] = Complex(potential_re, potential_im)
    return nothing
end

function _evaluate_galerkin_field_metal_fast(
    eval_points, pressure, q_neumann, k::Float32, cache::MetalFieldEvaluationCache; return_device::Bool=false,
)
    point_count = length(eval_points)
    # Test (BLAB_TEST_FIELD_INFO=<file>): sizes and wall of each call.
    info_path = get(ENV, "BLAB_TEST_FIELD_INFO", "")
    started = time_ns()
    tables = _metal_fast_field_tables_for(cache)
    d_eval_points = MtlArray(_metal_eval_point_arrays(eval_points, Float32))
    pressure_on_device = pressure isa MtlArray
    neumann_on_device = q_neumann isa MtlArray
    d_pressure = pressure_on_device ? pressure : MtlArray(ComplexF32.(pressure))
    d_neumann = neumann_on_device ? q_neumann : MtlArray(ComplexF32.(q_neumann))
    d_weights4 = MtlArray{_MetalFloat4}(undef, cache.source_count)
    d_potentials = Metal.zeros(ComplexF32, point_count)
    groupsize = 128
    _metal_launch(
        _metal_fast_field_sources_kernel!,
        cache.source_count,
        d_weights4, d_pressure, d_neumann, cache.source_weights, cache.source_faces, cache.source_elements,
        cache.basis_values, Int32(cache.source_count);
        groupsize=groupsize,
    )
    chunk_count = _metal_field_chunk_count(point_count, cache.source_count)
    chunk_length = cld(cache.source_count, chunk_count)
    d_partials = chunk_count == 1 ? d_potentials : Metal.zeros(ComplexF32, point_count * chunk_count)
    _metal_launch(
        _metal_fast_field_kernel!,
        point_count * chunk_count,
        d_partials, d_eval_points, tables.points4, tables.normals4, d_weights4, k,
        Int32(cache.source_count), Int32(point_count), Int32(chunk_length), Int32(chunk_count),
        Val(parse(Int, get(ENV, "BLAB_METAL_FIELD_FAST", "1")));
        groupsize=groupsize,
    )
    if chunk_count > 1
        _metal_launch(
            _metal_field_reduce_partials_kernel!,
            point_count,
            d_potentials, d_partials, point_count, chunk_count;
            groupsize=groupsize,
        )
    end
    Metal.synchronize()
    isempty(info_path) || open(io -> println(io, "points=", point_count, " sources=", cache.source_count, " chunks=", chunk_count,
        " drives=", size(pressure, 2), " wall=", (time_ns() - started) / 1e9), info_path, "a")
    result = return_device ? d_potentials : ComplexF32.(Array(d_potentials))
    Metal.unsafe_free!(d_eval_points)
    pressure_on_device || Metal.unsafe_free!(d_pressure)
    neumann_on_device || Metal.unsafe_free!(d_neumann)
    Metal.unsafe_free!(d_weights4)
    chunk_count == 1 || Metal.unsafe_free!(d_partials)
    return_device || Metal.unsafe_free!(d_potentials)
    return result
end

# Test (BLAB_TEST_FIELD_MULTI=1): every excitation's field in one pass. Distance, Green's value and
# normal projection are computed once per (point, source) and applied to ND weight sets (drive-major
# weights4); each drive sums in the same order as `_metal_fast_field_kernel!`.
@inline _test_mf_update(pre, pim, weights4, s, stride, dr, di, gr, gi, ::Val{0}) = (pre, pim)
@inline function _test_mf_update(pre, pim, weights4, s, stride, dr, di, gr, gi, ::Val{D}) where {D}
    pre, pim = _test_mf_update(pre, pim, weights4, s, stride, dr, di, gr, gi, Val(D - 1))
    @inbounds w = weights4[s + Int32(D - 1) * stride]
    Base.@fastmath begin
        re = pre[D] + (dr * w[1].value - di * w[2].value - gr * w[3].value + gi * w[4].value)
        im = pim[D] + (dr * w[2].value + di * w[1].value - gr * w[4].value - gi * w[3].value)
    end
    return Base.setindex(pre, re, D), Base.setindex(pim, im, D)
end

function _test_multi_field_kernel!(
    partials, eval_points, points4, normals4, weights4, k::Float32,
    source_count::Int32, point_count::Int32, chunk_length::Int32, chunk_count::Int32, ::Val{MODE}, ::Val{ND},
) where {MODE,ND}
    linear_index = Int32(thread_position_in_grid_1d())
    linear_index > point_count * chunk_count && return nothing
    point_index = (linear_index - Int32(1)) % point_count + Int32(1)
    chunk_index = (linear_index - Int32(1)) ÷ point_count
    source_start = chunk_index * chunk_length + Int32(1)
    source_stop = min(source_count, source_start + chunk_length - Int32(1))
    inv_four_pi = 0.07957747154594767f0
    @inbounds x1 = eval_points[point_index]
    @inbounds x2 = eval_points[point_index + point_count]
    @inbounds x3 = eval_points[point_index + Int32(2) * point_count]
    potential_re = ntuple(_ -> 0.0f0, Val(ND))
    potential_im = ntuple(_ -> 0.0f0, Val(ND))
    s = source_start
    while s <= source_stop
        @inbounds sp = points4[s]
        Base.@fastmath begin
        r1 = sp[1].value - x1
        r2 = sp[2].value - x2
        r3 = sp[3].value - x3
        radius2 = r1 * r1 + r2 * r2 + r3 * r3
        if radius2 > 0.0f0
            @inbounds sn = normals4[s]
            if MODE == 4
                radius = _metal_precise_sqrt(radius2)
                inv_radius = 1.0f0 / radius
            else
                inv_radius = _metal_fast_rsqrt(radius2)
                radius = radius2 * inv_radius
            end
            phase = k * radius
            green_scale = inv_radius * inv_four_pi
            c, sn_ = _metal_field_cis(phase, Val(MODE))
            green_re = c * green_scale
            green_im = sn_ * green_scale
            normal_projection = (r1 * sn[1].value + r2 * sn[2].value + r3 * sn[3].value) * inv_radius
            double_re = (-green_re * inv_radius - green_im * k) * normal_projection
            double_im = (green_re * k - green_im * inv_radius) * normal_projection
            potential_re, potential_im = _test_mf_update(potential_re, potential_im, weights4, s, source_count,
                double_re, double_im, green_re, green_im, Val(ND))
        end
        end
        s += Int32(1)
    end
    stride = point_count * chunk_count
    d = 1
    while d <= ND
        @inbounds partials[linear_index + Int32(d - 1) * stride] = Complex(potential_re[d], potential_im[d])
        d += 1
    end
    return nothing
end

function _test_multi_field_reduce_kernel!(potentials, partials, point_count::Int32, chunk_count::Int32, drive_count::Int32)
    index = Int32(thread_position_in_grid_1d())
    index > point_count * drive_count && return nothing
    point_index = (index - Int32(1)) % point_count + Int32(1)
    drive = (index - Int32(1)) ÷ point_count
    base = drive * point_count * chunk_count
    total_re = 0.0f0
    total_im = 0.0f0
    chunk_index = Int32(0)
    while chunk_index < chunk_count
        @inbounds value = partials[base + point_index + chunk_index * point_count]
        total_re += real(value)
        total_im += imag(value)
        chunk_index += Int32(1)
    end
    @inbounds potentials[index] = Complex(total_re, total_im)
    return nothing
end

_test_field_multi_on() = get(ENV, "BLAB_TEST_FIELD_MULTI", "0") == "1" && get(ENV, "BLAB_METAL_FIELD_FAST", "0") != "0" &&
    get(ENV, "BLAB_TEST_FIELD_F64", "0") != "1"

"""
Test (BLAB_TEST_FIELD_MULTI=1): `[evaluate_galerkin_field_metal(points, mesh, p, q, k, cache) for (p, q)]`
in one kernel pass (up to 8 drives per pass). Falls back to one call per drive when the switch or the
fast Float32 field is off.
"""
function evaluate_galerkin_field_metal_multi(eval_points, mesh::BoundaryMesh{T}, pressures, neumanns, k::T,
                                             cache::MetalFieldEvaluationCache{T}) where {T<:AbstractFloat}
    drive_count = length(pressures)
    if !(T === Float32 && _test_field_multi_on()) || drive_count <= 1 || isempty(eval_points)
        return [evaluate_galerkin_field_metal(eval_points, mesh, p, q, k, cache) for (p, q) in zip(pressures, neumanns)]
    end
    k = outgoing_wavenumber(k)
    _require_metal!()
    results = Vector{Vector{ComplexF32}}(undef, drive_count)
    first_drive = 1
    while first_drive <= drive_count
        last_drive = min(drive_count, first_drive + 7)
        results[first_drive:last_drive] = _test_multi_field_pass(eval_points, pressures[first_drive:last_drive],
                                                                 neumanns[first_drive:last_drive], Float32(k), cache)
        first_drive = last_drive + 1
    end
    return results
end

function _test_multi_field_pass(eval_points, pressures, neumanns, k::Float32, cache::MetalFieldEvaluationCache)
    nd = length(pressures)
    point_count = length(eval_points)
    source_count = cache.source_count
    tables = _metal_fast_field_tables_for(cache)
    d_eval_points = MtlArray(_metal_eval_point_arrays(eval_points, Float32))
    d_weights4 = MtlArray{_MetalFloat4}(undef, source_count * nd)
    groupsize = 128
    for d in 1:nd
        d_pressure = MtlArray(ComplexF32.(pressures[d]))
        d_neumann = MtlArray(ComplexF32.(neumanns[d]))
        _metal_launch(
            _metal_fast_field_sources_kernel!,
            source_count,
            view(d_weights4, (d - 1) * source_count + 1:d * source_count), d_pressure, d_neumann,
            cache.source_weights, cache.source_faces, cache.source_elements, cache.basis_values, Int32(source_count);
            groupsize=groupsize,
        )
        Metal.synchronize()
        Metal.unsafe_free!(d_pressure)
        Metal.unsafe_free!(d_neumann)
    end
    chunk_count = _metal_field_chunk_count(point_count, source_count)
    chunk_length = cld(source_count, chunk_count)
    d_partials = MtlArray{ComplexF32}(undef, point_count * chunk_count * nd)
    d_potentials = MtlArray{ComplexF32}(undef, point_count * nd)
    _metal_launch(
        _test_multi_field_kernel!,
        point_count * chunk_count,
        d_partials, d_eval_points, tables.points4, tables.normals4, d_weights4, k,
        Int32(source_count), Int32(point_count), Int32(chunk_length), Int32(chunk_count),
        Val(parse(Int, get(ENV, "BLAB_METAL_FIELD_FAST", "1"))), Val(nd);
        groupsize=groupsize,
    )
    _metal_launch(
        _test_multi_field_reduce_kernel!,
        point_count * nd,
        d_potentials, d_partials, Int32(point_count), Int32(chunk_count), Int32(nd);
        groupsize=groupsize,
    )
    Metal.synchronize()
    host = Array(d_potentials)
    Metal.unsafe_free!(d_eval_points)
    Metal.unsafe_free!(d_weights4)
    Metal.unsafe_free!(d_partials)
    Metal.unsafe_free!(d_potentials)
    return [host[(d - 1) * point_count + 1:d * point_count] for d in 1:nd]
end
