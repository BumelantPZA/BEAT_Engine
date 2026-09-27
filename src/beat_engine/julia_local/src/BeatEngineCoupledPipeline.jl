"""
    BeatEngineCoupledPipeline

Frequency pipelining for condensed Metal sweeps: while frequency `i` is solved and its field
evaluated on the host, frequency `i + 1` is built on another task, with its BEM operators
assembled on the GPU in advance.

The order the pipeline keeps, for every `i`:

- the GPU operators of `i + 1` are assembled once `build(i)` knows its flux columns, and those of
  `i + 2` only after `field(i)`: the GPU never runs beside a field evaluation, which gave wrong
  results on Metal;
- `field(i)` waits for the operators of `i + 1`, for the same reason;
- `build(i + 1)` starts after `solve(i)` (its MUMPS work must not overlap `i`'s MUMPS calls, since
  MUMPS keeps process-global state), or, with `fem_lane`, as soon as `build(i)` returns, with its
  dense part gated on `solve(i)`. `fem_lane` needs a solve that makes no MUMPS call: voltage-only
  excitations (zero FEM right-hand side) and no interior reconstruction.

One dense LU runs at a time, so the reused stale factor of `RefinedDenseLU` stays in sweep order.
Overlapping any other CPU-heavy stage measured slower on an M1 Pro (contention beside MUMPS).
"""
module BeatEngineCoupledPipeline

export CondensedSweepPipeline,
    pipeline_system!,
    pipeline_solved!,
    pipeline_before_field!,
    pipeline_field_done!,
    release_pipeline!

mutable struct CondensedSweepPipeline
    count::Int
    fem_lane::Bool
    # (index, prefetched_operators, on_operators_ready, dense_gate) -> system
    build::Function
    # (index, flux_columns) -> operators, as `condensed_metal_operators` returns them
    assemble::Function
    release_system::Function
    release_operators::Function
    builds::Vector{Any}
    operators::Vector{Any}
    operators_taken::BitVector
    solved::Vector{Base.Event}
    field_done::Vector{Base.Event}
    consumed::Int
    stopped::Bool   # set by `release_pipeline!`: no new builds start
end

function CondensedSweepPipeline(count::Integer; fem_lane::Bool, build, assemble, release_system, release_operators)
    return CondensedSweepPipeline(
        count, fem_lane, build, assemble, release_system, release_operators,
        Vector{Any}(nothing, count), Vector{Any}(nothing, count), falses(count),
        [Base.Event() for _ in 1:count], [Base.Event() for _ in 1:count], 0, false,
    )
end

# `build(index)`'s `on_operators_ready`: assemble the next frequency's operators once the field of
# the frequency before this one is done.
function _on_operators_ready(pipeline::CondensedSweepPipeline, index::Int)
    return (flux_columns=nothing) -> begin
        index < pipeline.count || return nothing
        gate = index >= 2 ? pipeline.field_done[index - 1] : nothing
        pipeline.operators[index + 1] = Threads.@spawn begin
            isnothing(gate) || wait(gate)
            pipeline.assemble(index + 1, flux_columns)
        end
        return nothing
    end
end

function _spawn_build!(pipeline::CondensedSweepPipeline, index::Int)
    (index <= pipeline.count && !pipeline.stopped) || return nothing
    pipeline.operators_taken[index] = true
    pipeline.builds[index] = Threads.@spawn begin
        system = pipeline.build(
            index, pipeline.operators[index], _on_operators_ready(pipeline, index),
            pipeline.fem_lane ? pipeline.solved[index - 1] : nothing,
        )
        pipeline.fem_lane && _spawn_build!(pipeline, index + 1)
        system
    end
    return nothing
end

# A task's own exception, so the error a caller sees does not depend on whether a stage ran on
# another task.
function _fetch_unwrapped(task)
    try
        return fetch(task)
    catch exception
        exception isa TaskFailedException || rethrow()
        rethrow(exception.task.result)
    end
end

"""
    pipeline_system!(pipeline, index)

The coupled system of frequency `index`: built here for the first frequency, fetched from its
build task after that.
"""
function pipeline_system!(pipeline::CondensedSweepPipeline, index::Int)
    pipeline.consumed = index
    if index == 1
        system = pipeline.build(1, nothing, _on_operators_ready(pipeline, 1), nothing)
        pipeline.fem_lane && _spawn_build!(pipeline, 2)
        return system
    end
    task = pipeline.builds[index]
    pipeline.builds[index] = nothing
    return _fetch_unwrapped(task)
end

"""Frequency `index` is solved: without `fem_lane`, the next build starts now."""
function pipeline_solved!(pipeline::CondensedSweepPipeline, index::Int)
    notify(pipeline.solved[index])
    pipeline.fem_lane || _spawn_build!(pipeline, index + 1)
    return nothing
end

"""Wait until the GPU is idle before frequency `index`'s field evaluation."""
function pipeline_before_field!(pipeline::CondensedSweepPipeline, index::Int)
    index < pipeline.count || return nothing
    task = pipeline.operators[index + 1]
    isnothing(task) && return nothing
    # A failed assembly is reported by the build that fetches it.
    try
        wait(task)
    catch
    end
    return nothing
end

"""Frequency `index`'s field evaluation is done: the GPU may assemble again."""
pipeline_field_done!(pipeline::CondensedSweepPipeline, index::Int) = notify(pipeline.field_done[index])

"""
    release_pipeline!(pipeline)

Unblock every waiting task and release whatever the sweep built but did not consume: systems
built ahead, and operators assembled for a build that never started.
"""
function release_pipeline!(pipeline::CondensedSweepPipeline)
    pipeline.stopped = true
    foreach(notify, pipeline.solved)
    foreach(notify, pipeline.field_done)
    for index in (pipeline.consumed + 1):pipeline.count
        task = pipeline.builds[index]
        isnothing(task) && continue
        try
            pipeline.release_system(fetch(task))
        catch
        end
        pipeline.builds[index] = nothing
    end
    for (task, taken) in zip(pipeline.operators, pipeline.operators_taken)
        (isnothing(task) || taken) && continue
        try
            pipeline.release_operators(fetch(task).operators)
        catch
        end
    end
    return nothing
end

end
