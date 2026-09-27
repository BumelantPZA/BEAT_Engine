# pair_tilereduce: the pair_gather path with the packed pair maths of
# BeatEngineMetalPackedPairs.jl and a threadgroup reduction along the test
# axis, so the intermediate buffer holds per-(node slot, trial) sums instead of
# per-(test element, trial) blocks.
#
# Test positions are sorted by their smallest P1 node, so a tile of 16
# consecutive test elements touches ~20-40 nodes ("slots"). A 16 x TY
# threadgroup evaluates 16 test x TY trial pairs (packed pair maths), and in
# four phases writes 12 of each pair's 48 values to threadgroup memory; the
# group then sums them per (slot, trial) in a fixed order and stores one
# float4 per (phase, slot, trial):
#   phase 0      : (S re, S im, K' re, K' im) for the slot's node as row
#   phase lc 1..3: (D re, D im, H re, H im) for row = slot node, local column lc
# The gathers then sum over the row node's slots (1-3 per node) instead of
# its incident test elements (~6). Summation order changes against
# pair_gather (float32 noise), but is fixed, so results are reproducible.

using Metal: thread_position_in_threadgroup, threadgroup_position_in_grid, thread_index_in_threadgroup,
    threadgroup_barrier, MemoryFlagThreadGroup, MtlThreadGroupArray

const _METAL_TILEREDUCE_TX = 16
# Trial rows per threadgroup. 2, 4 and 8 were measured no faster on an M1 Pro.
const _METAL_TILEREDUCE_TY = 16

struct MetalTileReduceTables
    tile_count::Int
    chunk_size::Int
    chunk_count::Int
    slot_total::Int
    max_slots::Int
    elements              # position (tile order) -> global element
    tile_slot_offsets     # tile -> first global slot, 0-based; tile_count + 1 entries
    slot_entry_offsets    # global slot -> first entry in slot_entries; slot_total + 1 entries
    slot_entries          # (local test element - 1) * 4 + local row
    row_slot_offsets      # P1 row -> first entry in row_slots; p1_count + 1 entries
    row_slots             # global slot ids (1-based)
    chunk_node_offsets::Vector{Int}
    chunk_nodes
    inc_offsets
    inc_packed
    blocks4               # MtlArray{_MetalFloat4}: 4 * slot_total * chunk_size
end

# Test-element order for the tiles: compact patches of `tile` elements, grown from a seed touching the previous patch (else the lowest
# unassigned min node), repeatedly add the unassigned element sharing the most
# nodes with the patch. That gives ~1.1 node slots per element on the test mesh
# against ~1.4 for plain sorting by smallest node.
function _metal_tilereduce_order(p1_dofs, elements, tile::Int)
    n = length(elements)
    min_node = [minimum(@view p1_dofs[e, :]) for e in elements]
    by_min_node = sortperm(1:n; by=i -> (min_node[i], i))
    node_elements = Dict{Int,Vector{Int}}()
    for i in 1:n, c in 1:3
        push!(get!(node_elements, Int(p1_dofs[elements[i], c]), Int[]), i)
    end
    assigned = falses(n)
    order = Int[]
    seed_pointer = 1
    previous_nodes = Int[]
    score = Dict{Int,Int}()
    patch_nodes = Set{Int}()
    while length(order) < n
        seed = 0
        for v in sort(previous_nodes), i in node_elements[v]
            if !assigned[i]
                seed = i
                break
            end
        end
        if seed == 0
            while assigned[by_min_node[seed_pointer]]
                seed_pointer += 1
            end
            seed = by_min_node[seed_pointer]
        end
        empty!(score)
        empty!(patch_nodes)
        count = 0
        next = seed
        while true
            assigned[next] = true
            push!(order, next)
            count += 1
            delete!(score, next)
            for c in 1:3
                v = Int(p1_dofs[elements[next], c])
                v in patch_nodes && continue
                push!(patch_nodes, v)
                for i in node_elements[v]
                    assigned[i] || (score[i] = get(score, i, 0) + 1)
                end
            end
            (count >= tile || length(order) == n) && break
            if isempty(score)
                # Enclosed patch: keep the tile full with the lowest unassigned element.
                while assigned[by_min_node[seed_pointer]]
                    seed_pointer += 1
                end
                next = by_min_node[seed_pointer]
                continue
            end
            best = 0
            best_key = (typemin(Int), typemin(Int), typemin(Int))
            for (i, sc) in score
                key = (sc, -min_node[i], -i)
                if key > best_key
                    best_key = key
                    best = i
                end
            end
            next = best
        end
        previous_nodes = collect(patch_nodes)
    end
    return order
end

const _metal_tilereduce_tables = WeakKeyDict{Any,MetalTileReduceTables}()

function _metal_tilereduce_tables_for(cache::MetalRegularAssemblyCache)
    haskey(_metal_tilereduce_tables, cache.gather_tables) && return _metal_tilereduce_tables[cache.gather_tables]
    p1_dofs = Array(cache.p1_dofs)   # face_count x 3
    original = cache.element_indices
    element_count = length(original)
    TX = _METAL_TILEREDUCE_TX
    order = _metal_tilereduce_order(p1_dofs, original, TX)
    indices = original[order]
    tile_count = cld(element_count, TX)
    p1_count = cache.p1_dof_count

    tile_slot_offsets = Int32[0]
    slot_entry_offsets = Int32[1]
    slot_entries = Int32[]
    row_lists = [Int32[] for _ in 1:p1_count]
    slot = 0
    max_slots = 0
    for tile in 1:tile_count
        first = (tile - 1) * TX + 1
        last = min(tile * TX, element_count)
        incidence = Dict{Int32,Vector{Int32}}()
        for position in first:last, local_row in 1:3
            node = Int32(p1_dofs[indices[position], local_row])
            push!(get!(incidence, node, Int32[]), Int32((position - first) * 4 + local_row))
        end
        nodes = sort!(collect(keys(incidence)))
        max_slots = max(max_slots, length(nodes))
        for node in nodes
            slot += 1
            append!(slot_entries, incidence[node])
            push!(slot_entry_offsets, Int32(length(slot_entries) + 1))
            push!(row_lists[node], Int32(slot))
        end
        push!(tile_slot_offsets, Int32(slot))
    end
    slot_total = slot
    row_slot_offsets = Int32[1]
    row_slots = Int32[]
    for row in 1:p1_count
        append!(row_slots, row_lists[row])
        push!(row_slot_offsets, Int32(length(row_slots) + 1))
    end

    budget_mb = parse(Float64, get(ENV, "BLAB_METAL_GATHER_BUDGET_MB", "512"))
    per_column_bytes = slot_total * 4 * sizeof(_MetalFloat4)
    chunk_size = clamp(floor(Int, budget_mb * 1e6 / per_column_bytes), 1, element_count)
    while 4 * slot_total * chunk_size >= typemax(Int32) && chunk_size > 1
        chunk_size ÷= 2
    end
    chunk_count = cld(element_count, chunk_size)

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
            trial_local = position - start + 1
            for local_column in 1:3
                node = Int32(p1_dofs[indices[position], local_column])
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

    tables = MetalTileReduceTables(
        tile_count,
        chunk_size,
        chunk_count,
        slot_total,
        max_slots,
        MtlArray(Int32.(indices)),
        MtlArray(tile_slot_offsets),
        MtlArray(slot_entry_offsets),
        MtlArray(slot_entries),
        MtlArray(row_slot_offsets),
        MtlArray(row_slots),
        chunk_node_offsets,
        MtlArray(chunk_nodes),
        MtlArray(inc_offsets),
        MtlArray(inc_packed),
        MtlArray{_MetalFloat4}(undef, 4 * slot_total * chunk_size),
    )
    _metal_tilereduce_tables[cache.gather_tables] = tables
    get(ENV, "BLAB_METAL_GATHER_TIMING", "0") == "1" &&
        @info "pair_tilereduce tables" tile_count slot_total max_slots chunk_size chunk_count
    return tables
end

function _release_metal_tilereduce_tables!(cache::MetalRegularAssemblyCache)
    tables = pop!(_metal_tilereduce_tables, cache.gather_tables, nothing)
    tables === nothing && return nothing
    for array in (tables.elements, tables.tile_slot_offsets, tables.slot_entry_offsets, tables.slot_entries,
            tables.row_slot_offsets, tables.row_slots, tables.chunk_nodes, tables.inc_offsets, tables.inc_packed,
            tables.blocks4)
        Metal.unsafe_free!(array)
    end
    return nothing
end

# Writes one phase's 12 values of this thread's pair: a[lr], b[lr], c[lr], d[lr] at rows lr, 3+lr, 6+lr, 9+lr.
@inline function _metal_tilereduce_put!(tg, tx, ty, a, b, c, d)
    @inbounds begin
        tg[tx, ty, 1] = a[1]; tg[tx, ty, 2] = a[2]; tg[tx, ty, 3] = a[3]
        tg[tx, ty, 4] = b[1]; tg[tx, ty, 5] = b[2]; tg[tx, ty, 6] = b[3]
        tg[tx, ty, 7] = c[1]; tg[tx, ty, 8] = c[2]; tg[tx, ty, 9] = c[3]
        tg[tx, ty, 10] = d[1]; tg[tx, ty, 11] = d[2]; tg[tx, ty, 12] = d[3]
    end
    return nothing
end

# Sums the phase's values per (slot, trial row) and stores float4s at group `phase`.
# Thread (tx, ty) owns slots tx, tx + 16, ... of its own trial row ty.
@inline function _metal_tilereduce_reduce!(
    blocks4, tg, slot_entry_offsets, slot_entries, slot_first::Int32, slot_count::Int32,
    tx::Int32, ty::Int32, trial_local::Int32, chunk_count::Int32, slot_total::Int32,
    phase_offset::Int32, accumulate::Int32,
)
    slot_local = tx - Int32(1)
    while slot_local < slot_count
        global_slot = slot_first + slot_local + Int32(1)
        @inbounds entry = Int32(slot_entry_offsets[global_slot])
        @inbounds entry_stop = Int32(slot_entry_offsets[global_slot + Int32(1)]) - Int32(1)
        a1 = 0.0f0
        a2 = 0.0f0
        a3 = 0.0f0
        a4 = 0.0f0
        while entry <= entry_stop
            @inbounds packed = Int32(slot_entries[entry])
            le = (packed >> 2) + Int32(1)
            lr = packed & Int32(3)
            @inbounds a1 += tg[le, ty, lr]
            @inbounds a2 += tg[le, ty, lr + Int32(3)]
            @inbounds a3 += tg[le, ty, lr + Int32(6)]
            @inbounds a4 += tg[le, ty, lr + Int32(9)]
            entry += Int32(1)
        end
        if trial_local <= chunk_count
            block_index = phase_offset + (trial_local - Int32(1)) * slot_total + global_slot
            if accumulate != Int32(0)
                # Later symmetry transforms add to the block; this thread is the only
                # writer of the entry.
                @inbounds old = blocks4[block_index]
                a1 += old[1].value
                a2 += old[2].value
                a3 += old[3].value
                a4 += old[4].value
            end
            @inbounds blocks4[block_index] = _metal_float4(a1, a2, a3, a4)
        end
        slot_local += Int32(16)
    end
    return nothing
end

function _metal_tilereduce_pair_kernel!(
    blocks4,
    points4,
    normals4,
    areas,
    curls4,
    faces,
    elements,
    tile_slot_offsets,
    slot_entry_offsets,
    slot_entries,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    slot_total::Int32,
    group_stride::Int32,
    k,
    face_count::Int32,
    rc::Val{RC},
    rv::Val{R},
    tyv::Val{TY},
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
    element_flux_mask,
    mask_on::Int32,
    accumulate::Int32,
    combv::Val{COMB},
    beta_re,
    beta_im,
) where {RC,R,TY,COMB}
    tg = MtlThreadGroupArray(Float32, (16, TY, 12))
    tpos = thread_position_in_threadgroup()
    gpos = threadgroup_position_in_grid()
    tx = Int32(tpos.x)
    ty = Int32(tpos.y)
    tile = Int32(gpos.x)
    trial_first = (Int32(gpos.y) - Int32(1)) * Int32(TY)
    test_position = (tile - Int32(1)) * Int32(16) + tx
    trial_local = trial_first + ty
    T = Float32
    slp_re = zero(SVector{3,T})
    slp_im = zero(SVector{3,T})
    adj_re = zero(SVector{3,T})
    adj_im = zero(SVector{3,T})
    dlp_re = zero(SVector{9,T})
    dlp_im = zero(SVector{9,T})
    hyp_re = zero(SVector{9,T})
    hyp_im = zero(SVector{9,T})
    # No early return: every thread must reach the barriers below.
    rigid_trial = false   # no flux on the trial face: its S/K' column is never gathered
    if test_position <= element_count && trial_local <= chunk_count
        @inbounds test_index = Int32(elements[test_position])
        @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
        if mask_on != Int32(0)
            @inbounds rigid_trial = element_flux_mask[trial_index] == Int32(0)
        end
        if !_metal_pair_is_skipped(
            faces,
            face_count,
            test_index,
            trial_index,
            pair_offsets,
            singular_trial_indices,
            skip_mode,
        )
            slp_re, slp_im, adj_re, adj_im, dlp_re, dlp_im, hyp_re, hyp_im = _metal_regular_pair_blocks_packed(
                points4, normals4, areas, curls4, test_index, trial_index, k, rc, rv,
                trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
            )
        end
    end
    @inbounds slot_first = Int32(tile_slot_offsets[tile])
    @inbounds slot_count = Int32(tile_slot_offsets[tile + Int32(1)]) - slot_first

    if COMB
        # Combined Burton-Miller assembly: C = -S - βK' (3 test rows) and A = -D + βH (3 x 3) in two passes.
        # Phase 0: (C re, C im, A re/im of trial basis 1); phase 1: A re/im of trial bases 2, 3.
        c_re = -slp_re - (beta_re * adj_re - beta_im * adj_im)
        c_im = -slp_im - (beta_re * adj_im + beta_im * adj_re)
        a_re = -dlp_re + (beta_re * hyp_re - beta_im * hyp_im)
        a_im = -dlp_im + (beta_re * hyp_im + beta_im * hyp_re)
        _metal_tilereduce_put!(tg, tx, ty, c_re, c_im,
            SVector(a_re[1], a_re[2], a_re[3]), SVector(a_im[1], a_im[2], a_im[3]))
        threadgroup_barrier(MemoryFlagThreadGroup)
        _metal_tilereduce_reduce!(blocks4, tg, slot_entry_offsets, slot_entries, slot_first, slot_count,
            tx, ty, trial_local, chunk_count, slot_total, Int32(0), accumulate)
        threadgroup_barrier(MemoryFlagThreadGroup)
        _metal_tilereduce_put!(tg, tx, ty,
            SVector(a_re[4], a_re[5], a_re[6]), SVector(a_im[4], a_im[5], a_im[6]),
            SVector(a_re[7], a_re[8], a_re[9]), SVector(a_im[7], a_im[8], a_im[9]))
        threadgroup_barrier(MemoryFlagThreadGroup)
        _metal_tilereduce_reduce!(blocks4, tg, slot_entry_offsets, slot_entries, slot_first, slot_count,
            tx, ty, trial_local, chunk_count, slot_total, group_stride, accumulate)
        return nothing
    end

    _metal_tilereduce_put!(tg, tx, ty, slp_re, slp_im, adj_re, adj_im)
    threadgroup_barrier(MemoryFlagThreadGroup)
    if !rigid_trial
        _metal_tilereduce_reduce!(blocks4, tg, slot_entry_offsets, slot_entries, slot_first, slot_count,
            tx, ty, trial_local, chunk_count, slot_total, Int32(0), accumulate)
    end
    threadgroup_barrier(MemoryFlagThreadGroup)

    _metal_tilereduce_put!(tg, tx, ty,
        SVector(dlp_re[1], dlp_re[2], dlp_re[3]), SVector(dlp_im[1], dlp_im[2], dlp_im[3]),
        SVector(hyp_re[1], hyp_re[2], hyp_re[3]), SVector(hyp_im[1], hyp_im[2], hyp_im[3]))
    threadgroup_barrier(MemoryFlagThreadGroup)
    _metal_tilereduce_reduce!(blocks4, tg, slot_entry_offsets, slot_entries, slot_first, slot_count,
        tx, ty, trial_local, chunk_count, slot_total, group_stride, accumulate)
    threadgroup_barrier(MemoryFlagThreadGroup)

    _metal_tilereduce_put!(tg, tx, ty,
        SVector(dlp_re[4], dlp_re[5], dlp_re[6]), SVector(dlp_im[4], dlp_im[5], dlp_im[6]),
        SVector(hyp_re[4], hyp_re[5], hyp_re[6]), SVector(hyp_im[4], hyp_im[5], hyp_im[6]))
    threadgroup_barrier(MemoryFlagThreadGroup)
    _metal_tilereduce_reduce!(blocks4, tg, slot_entry_offsets, slot_entries, slot_first, slot_count,
        tx, ty, trial_local, chunk_count, slot_total, Int32(2) * group_stride, accumulate)
    threadgroup_barrier(MemoryFlagThreadGroup)

    _metal_tilereduce_put!(tg, tx, ty,
        SVector(dlp_re[7], dlp_re[8], dlp_re[9]), SVector(dlp_im[7], dlp_im[8], dlp_im[9]),
        SVector(hyp_re[7], hyp_re[8], hyp_re[9]), SVector(hyp_im[7], hyp_im[8], hyp_im[9]))
    threadgroup_barrier(MemoryFlagThreadGroup)
    _metal_tilereduce_reduce!(blocks4, tg, slot_entry_offsets, slot_entries, slot_first, slot_count,
        tx, ty, trial_local, chunk_count, slot_total, Int32(3) * group_stride, accumulate)
    return nothing
end

# One thread per (P1 row, trial element of the chunk).
function _metal_tilereduce_slp_adjoint_kernel!(
    single_layer,
    adjoint_double_layer,
    blocks4,
    elements,
    row_slot_offsets,
    row_slots,
    element_dp0_dofs,
    chunk_start::Int32,
    chunk_count::Int32,
    slot_total::Int32,
    p1_count::Int32,
    flux_mask,
    mask_on::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * chunk_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    trial_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    if mask_on != Int32(0)
        @inbounds flux_mask[Int32(element_dp0_dofs[trial_index])] == Int32(0) && return nothing
    end
    column_base = (trial_local - Int32(1)) * slot_total
    s_re = 0.0f0
    s_im = 0.0f0
    a_re = 0.0f0
    a_im = 0.0f0
    @inbounds position = Int32(row_slot_offsets[row])
    @inbounds stop = Int32(row_slot_offsets[row + Int32(1)]) - Int32(1)
    while position <= stop
        @inbounds v = blocks4[column_base + Int32(row_slots[position])]
        s_re += v[1].value
        s_im += v[2].value
        a_re += v[3].value
        a_im += v[4].value
        position += Int32(1)
    end
    @inbounds dp0_column = Int32(element_dp0_dofs[trial_index])
    operator_index = row + (dp0_column - Int32(1)) * p1_count
    @inbounds single_layer[operator_index] += Complex(s_re, s_im)
    @inbounds adjoint_double_layer[operator_index] += Complex(a_re, a_im)
    return nothing
end

# One thread per (P1 row, P1 node touched by the chunk).
function _metal_tilereduce_dlp_hyp_kernel!(
    double_layer,
    hypersingular,
    blocks4,
    row_slot_offsets,
    row_slots,
    chunk_nodes,
    inc_offsets,
    inc_packed,
    node_start::Int32,
    node_count::Int32,
    slot_total::Int32,
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
    @inbounds position = Int32(row_slot_offsets[row])
    @inbounds stop = Int32(row_slot_offsets[row + Int32(1)]) - Int32(1)
    while position <= stop
        @inbounds global_slot = Int32(row_slots[position])
        chunk_position = chunk_first
        while chunk_position <= chunk_stop
            @inbounds packed = Int32(inc_packed[chunk_position])
            trial_local = (packed >> 2) + Int32(1)
            local_column = packed & Int32(3)
            @inbounds v = blocks4[local_column * group_stride + (trial_local - Int32(1)) * slot_total + global_slot]
            d_re += v[1].value
            d_im += v[2].value
            h_re += v[3].value
            h_im += v[4].value
            chunk_position += Int32(1)
        end
        position += Int32(1)
    end
    operator_index = row + (column - Int32(1)) * p1_count
    @inbounds double_layer[operator_index] += Complex(d_re, d_im)
    @inbounds hypersingular[operator_index] += Complex(h_re, h_im)
    return nothing
end

# Combined assembly: C gather, one thread per (P1 row, trial element of the chunk).
function _metal_tilereduce_combined_c_kernel!(
    combined_c,
    blocks4,
    elements,
    row_slot_offsets,
    row_slots,
    element_dp0_dofs,
    chunk_start::Int32,
    chunk_count::Int32,
    slot_total::Int32,
    p1_count::Int32,
    flux_mask,
    mask_on::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * chunk_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    trial_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    @inbounds dp0_column = Int32(element_dp0_dofs[trial_index])
    if mask_on != Int32(0)
        @inbounds flux_mask[dp0_column] == Int32(0) && return nothing
    end
    column_base = (trial_local - Int32(1)) * slot_total
    c_re = 0.0f0
    c_im = 0.0f0
    @inbounds position = Int32(row_slot_offsets[row])
    @inbounds stop = Int32(row_slot_offsets[row + Int32(1)]) - Int32(1)
    while position <= stop
        @inbounds v = blocks4[column_base + Int32(row_slots[position])]
        c_re += v[1].value
        c_im += v[2].value
        position += Int32(1)
    end
    @inbounds combined_c[row + (dp0_column - Int32(1)) * p1_count] += Complex(c_re, c_im)
    return nothing
end

# Combined assembly: A gather, one thread per (P1 row, P1 node touched by the chunk).
# Trial basis 1 is phase 0 channels 3-4, bases 2 and 3 are phase 1 channels 1-2 and 3-4.
function _metal_tilereduce_combined_a_kernel!(
    combined_a,
    blocks4,
    row_slot_offsets,
    row_slots,
    chunk_nodes,
    inc_offsets,
    inc_packed,
    node_start::Int32,
    node_count::Int32,
    slot_total::Int32,
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
    a_re = 0.0f0
    a_im = 0.0f0
    @inbounds position = Int32(row_slot_offsets[row])
    @inbounds stop = Int32(row_slot_offsets[row + Int32(1)]) - Int32(1)
    while position <= stop
        @inbounds global_slot = Int32(row_slots[position])
        chunk_position = chunk_first
        while chunk_position <= chunk_stop
            @inbounds packed = Int32(inc_packed[chunk_position])
            trial_local = (packed >> 2) + Int32(1)
            local_column = packed & Int32(3)
            phase = local_column == Int32(1) ? Int32(0) : Int32(1)   # local_column is 1-based
            @inbounds v = blocks4[phase * group_stride + (trial_local - Int32(1)) * slot_total + global_slot]
            if local_column == Int32(2)
                a_re += v[1].value
                a_im += v[2].value
            else
                a_re += v[3].value
                a_im += v[4].value
            end
            chunk_position += Int32(1)
        end
        position += Int32(1)
    end
    @inbounds combined_a[row + (column - Int32(1)) * p1_count] += Complex(a_re, a_im)
    return nothing
end

# Placeholder for the kernels' flux-mask argument when no mask is set (`mask_on = 0`).
const _METAL_TILEREDUCE_NO_MASK = Ref{Any}(nothing)

"""
    _launch_metal_tilereduce_kernels!(operators, cache, k, skip_image_singular, options)

Regular assembly with the tile-reduce kernels. The identity and every symmetry image run through
one chunk loop: each transform's pair kernel writes (the first) or adds (the rest) into the
chunk's blocks, and the S/K' and D/H gathers then run once per chunk instead of once per
transform. `options` (`MetalAssemblyOptions`) selects the combined Burton-Miller operators and the
flux mask.
"""
function _launch_metal_tilereduce_kernels!(
    operators,
    cache::MetalRegularAssemblyCache,
    k,
    skip_image_singular::Bool,
    options::MetalAssemblyOptions,
)
    element_count = length(cache.element_indices)
    element_count == 0 && return nothing
    k isa Float32 || error("pair_tilereduce supports Float32 only.")
    T = typeof(k)
    transforms = Any[(cache.vertex_offsets, cache.incident_elements, Int32(0),
                      one(k), one(k), one(k), one(k), one(k), one(k))]
    for (transform, image_cache) in zip(cache.image_transforms, cache.image_singular_caches)
        push!(transforms, (image_cache.pair_offsets, image_cache.trial_indices,
                           skip_image_singular ? Int32(1) : Int32(2),
                           T(transform.signs[1]), T(transform.signs[2]), T(transform.signs[3]),
                           T(transform.determinant * transform.signs[1]),
                           T(transform.determinant * transform.signs[2]),
                           T(transform.determinant * transform.signs[3])))
    end
    rule_count = cache.rule_count
    tables = _metal_tilereduce_tables_for(cache)
    packed = _metal_packed_pair_tables_for(cache)
    chunk_size = tables.chunk_size
    slot_total = Int32(tables.slot_total)
    group_stride = Int32(tables.slot_total * chunk_size)
    ty = _METAL_TILEREDUCE_TY
    groupsize = _metal_kernel_groupsize()
    p1_count = Int32(cache.p1_dof_count)
    timed = get(ENV, "BLAB_METAL_GATHER_TIMING", "0") == "1"
    flux_mask = options.flux_mask
    mask_on = isnothing(flux_mask) ? Int32(0) : Int32(1)
    if isnothing(flux_mask)
        isnothing(_METAL_TILEREDUCE_NO_MASK[]) && (_METAL_TILEREDUCE_NO_MASK[] = MtlArray(Int32[1]))
        flux_mask = _METAL_TILEREDUCE_NO_MASK[]
    end
    # Per element (face index) for the pair kernel; per DP0 column for the S/K' gather.
    element_flux_mask = mask_on == Int32(0) ? flux_mask :
                        MtlArray(Array(flux_mask)[Array(cache.element_dp0_dofs)])
    combined = options.bm_coupling
    beta_re = isnothing(combined) ? zero(k) : real(combined)
    beta_im = isnothing(combined) ? zero(k) : imag(combined)
    timed && Metal.synchronize()
    stamp = time()
    for chunk in 1:tables.chunk_count
        chunk_start = (chunk - 1) * chunk_size + 1
        chunk_count = min(chunk_size, element_count - chunk_start + 1)
        for (transform_index, transform) in enumerate(transforms)
            (pair_offsets, singular_trial_indices, skip_mode, trial_sign_x, trial_sign_y, trial_sign_z,
             trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z) = transform
            Metal.@metal threads=(_METAL_TILEREDUCE_TX, ty) groups=(tables.tile_count, cld(chunk_count, ty)) _metal_tilereduce_pair_kernel!(
                tables.blocks4,
                packed.points4,
                packed.normals4,
                cache.areas,
                packed.curls4,
                cache.faces,
                tables.elements,
                tables.tile_slot_offsets,
                tables.slot_entry_offsets,
                tables.slot_entries,
                Int32(element_count),
                Int32(chunk_start),
                Int32(chunk_count),
                slot_total,
                group_stride,
                k,
                Int32(cache.face_count),
                Val(packed.rule),
                Val(rule_count),
                Val(ty),
                pair_offsets,
                singular_trial_indices,
                skip_mode,
                trial_sign_x,
                trial_sign_y,
                trial_sign_z,
                trial_curl_sign_x,
                trial_curl_sign_y,
                trial_curl_sign_z,
                element_flux_mask,
                mask_on,
                transform_index == 1 ? Int32(0) : Int32(1),
                Val(!isnothing(combined)),
                beta_re,
                beta_im,
            )
        end
        stamp = _metal_gather_stage!("pairs", timed, stamp)
        node_start = tables.chunk_node_offsets[chunk]
        node_count = tables.chunk_node_offsets[chunk + 1] - node_start
        if isnothing(combined)
            _metal_launch(
                _metal_tilereduce_slp_adjoint_kernel!,
                cache.p1_dof_count * chunk_count,
                operators.single_layer,
                operators.adjoint_double_layer,
                tables.blocks4,
                tables.elements,
                tables.row_slot_offsets,
                tables.row_slots,
                cache.element_dp0_dofs,
                Int32(chunk_start),
                Int32(chunk_count),
                slot_total,
                p1_count,
                flux_mask,
                mask_on;
                groupsize=groupsize,
            )
            stamp = _metal_gather_stage!("slp_adjoint", timed, stamp)
            _metal_launch(
                _metal_tilereduce_dlp_hyp_kernel!,
                cache.p1_dof_count * node_count,
                operators.double_layer,
                operators.hypersingular,
                tables.blocks4,
                tables.row_slot_offsets,
                tables.row_slots,
                tables.chunk_nodes,
                tables.inc_offsets,
                tables.inc_packed,
                Int32(node_start),
                Int32(node_count),
                slot_total,
                group_stride,
                p1_count;
                groupsize=groupsize,
            )
        else
            _metal_launch(
                _metal_tilereduce_combined_c_kernel!,
                cache.p1_dof_count * chunk_count,
                operators.single_layer,
                tables.blocks4,
                tables.elements,
                tables.row_slot_offsets,
                tables.row_slots,
                cache.element_dp0_dofs,
                Int32(chunk_start),
                Int32(chunk_count),
                slot_total,
                p1_count,
                flux_mask,
                mask_on;
                groupsize=groupsize,
            )
            stamp = _metal_gather_stage!("slp_adjoint", timed, stamp)
            _metal_launch(
                _metal_tilereduce_combined_a_kernel!,
                cache.p1_dof_count * node_count,
                operators.double_layer,
                tables.blocks4,
                tables.row_slot_offsets,
                tables.row_slots,
                tables.chunk_nodes,
                tables.inc_offsets,
                tables.inc_packed,
                Int32(node_start),
                Int32(node_count),
                slot_total,
                group_stride,
                p1_count;
                groupsize=groupsize,
            )
        end
        stamp = _metal_gather_stage!("dlp_hyp", timed, stamp)
    end
    return nothing
end
