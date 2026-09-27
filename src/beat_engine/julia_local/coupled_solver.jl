#!/usr/bin/env julia

# Script loader for the coupled worker. The engine is the module in BeatEngineCoupledWorker.jl. It comes from
# the precompiled accelerator bundle when that bundle contains it (as solver.jl does for the exterior
# engine), else it is included from source; BLAB_BEAT_ENGINE_BUNDLE=0 forces the source include (Revise harness).

# Test (BLAB_TEST_COLD_LOG=<file>, process env): start-up stamps, see BeatEngineCoupledWorker.test_cold_log.
function _loader_cold_log(label)
    path = get(ENV, "BLAB_TEST_COLD_LOG", "")
    isempty(path) && return nothing
    compile_s = Base.cumulative_compile_time_ns()[1] / 1.0e9
    open(io -> println(io, round(time(); digits=4), " ", round(compile_s; digits=4), " ", label), path, "a")
    return nothing
end
isempty(get(ENV, "BLAB_TEST_COLD_LOG", "")) || Base.cumulative_compile_timing(true)
isempty(get(ENV, "BLAB_TEST_COLD_LOG", "")) ||
    _loader_cold_log("script_start pid=$(getpid()) process_elapsed=$(strip(read(`ps -o etime= -p $(getpid())`, String)))")

using Base64, JSON, LinearAlgebra, SparseArrays, StaticArrays, Statistics
_loader_cold_log("using_done")

#: The bundle package for the accelerator this process is configured for (same rule as solver.jl).
const BEAT_ENGINE_BUNDLE_NAME = let
    hint = lowercase(strip(get(ENV, "BLAB_BEAT_ENGINE_GPU_BACKEND", "")))
    if isempty(hint)
        active = Base.active_project()
        directory = active === nothing ? "" : lowercase(basename(dirname(active)))
        hint = directory == "julia_cuda" ? "cuda" :
            directory == "julia_rocm" ? "rocm" :
            directory == "julia_metal" ? "metal" : "cpu"
    end
    hint == "cuda" ? :BeatEngineCudaBundle :
        hint == "rocm" ? :BeatEngineRocmBundle :
        hint == "metal" ? :BeatEngineMetalBundle : :BeatEngineCpuBundle
end

#: The bundle, only if it carries the coupled worker: loading one that doesn't would put a second copy of
#: the engine next to the source include. Checked on the bundle's source text, which costs nothing.
const BEAT_ENGINE_COUPLED_BUNDLE = if get(ENV, "BLAB_BEAT_ENGINE_BUNDLE", "1") == "0"
    nothing
else
    try
        id = Base.identify_package(String(BEAT_ENGINE_BUNDLE_NAME))
        path = id === nothing ? nothing : Base.locate_package(id)
        if path !== nothing && occursin("BeatEngineCoupledWorker", read(path, String))
            @eval using $BEAT_ENGINE_BUNDLE_NAME
            @eval $BEAT_ENGINE_BUNDLE_NAME
        else
            nothing
        end
    catch
        nothing
    end
end

if BEAT_ENGINE_COUPLED_BUNDLE === nothing
    include(joinpath(@__DIR__, "BeatEngineCoupledWorker.jl"))
else
    const BeatEngineCoupledWorker = BEAT_ENGINE_COUPLED_BUNDLE.BeatEngineCoupledWorker
end
using .BeatEngineCoupledWorker
_loader_cold_log("includes_done")

if isdefined(Main, :BLAB_DEV_WORKER)
    # perf/dev_worker.jl (Revise hot reload, test harness) starts the worker itself.
elseif "--worker" in ARGS
    try
        run_worker()
    catch exception
        reclaim_accelerator_memory!()
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        exit(1)
    end
else
    try
        request = JSON.parse(read(stdin, String))
        solve_request(request)
    catch exception
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        exit(1)
    end
end
