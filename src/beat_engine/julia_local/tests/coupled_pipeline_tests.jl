using Test

include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledPipeline.jl"))
using .BeatEngineCoupledPipeline

# Drives a sweep through `CondensedSweepPipeline` with stand-in build and GPU steps that record
# when they run, and returns the recorded events in order.
function run_pipeline_sweep(; count, fem_lane, stop_after=count, released=Ref(0))
    events = Any[]
    events_lock = ReentrantLock()
    record(event) = lock(() -> push!(events, event), events_lock)
    build = (index, operators, on_operators_ready, dense_gate) -> begin
        record((:build_start, index))
        if index > 1
            @test !isnothing(operators)
            @test fetch(operators).index == index
        end
        on_operators_ready(:flux)
        if !isnothing(dense_gate)
            wait(dense_gate)
            record((:dense, index))
        end
        record((:build_end, index))
        (index=index,)
    end
    assemble = (index, flux_columns) -> begin
        @test flux_columns == :flux
        record((:gpu_start, index))
        sleep(0.002)
        record((:gpu_end, index))
        (index=index, operators=index)
    end
    pipeline = CondensedSweepPipeline(
        count;
        fem_lane=fem_lane,
        build=build,
        assemble=assemble,
        release_system=system -> (released[] += 1),
        release_operators=operators -> nothing,
    )
    try
        for index in 1:stop_after
            @test pipeline_system!(pipeline, index).index == index
            record((:solve, index))
            pipeline_solved!(pipeline, index)
            pipeline_before_field!(pipeline, index)
            record((:field, index))
            pipeline_field_done!(pipeline, index)
        end
    finally
        release_pipeline!(pipeline)
    end
    return events
end

@testset "condensed sweep pipeline order (fem_lane=$fem_lane)" for fem_lane in (false, true)
    count = 6
    events = run_pipeline_sweep(; count=count, fem_lane=fem_lane)
    at(event) = findfirst(==(event), events)
    for index in 1:count
        @test !isnothing(at((:field, index)))
    end
    for index in 1:(count - 1)
        # The GPU never runs beside a field evaluation.
        @test at((:gpu_end, index + 1)) < at((:field, index))
        index + 2 <= count && @test at((:field, index)) < at((:gpu_start, index + 2))
        # The next build's dense part (fem_lane) or the whole next build follows this solve.
        @test at((:solve, index)) < (fem_lane ? at((:dense, index + 1)) : at((:build_start, index + 1)))
    end
    # One assembly at a time: every GPU step ends before the next one starts.
    gpu = [event for event in events if event[1] in (:gpu_start, :gpu_end)]
    @test all(gpu[position][1] == (isodd(position) ? :gpu_start : :gpu_end) for position in eachindex(gpu))
end

@testset "condensed sweep pipeline release after a stop (fem_lane=$fem_lane)" for fem_lane in (false, true)
    released = Ref(0)
    events = run_pipeline_sweep(; count=6, fem_lane=fem_lane, stop_after=2, released=released)
    # Nothing is left running, and only the systems already in flight are built and released
    # unconsumed: one ahead, or with `fem_lane` also the FEM stage of the one after it.
    ahead = fem_lane ? 2 : 1
    @test released[] <= ahead
    @test count(event -> event[1] == :build_start, events) <= 2 + ahead
end
