# Hot-reloading coupled worker for `quick.py --revise` (test harness only).
#
# The normal worker (`julia_local/coupled_solver.jl --worker`) includes the engine from source, so
# every edit meant a restart and ~60 s of loading and compiling. This runs the same worker with
# Revise tracking every engine file: before each solve, edited methods are re-evaluated in place
# and only what changed is recompiled (GPU kernels included).
#
# Revise cannot apply changes to `struct` definitions or `const` values; after such an edit touch
# perf/queue/RESTART and quick.py restarts the worker. A file that fails to revise fails the job.
# Needs Revise in perf/devenv, stacked onto the engine project by JULIA_LOAD_PATH (quick.py sets it).

const BLAB_DEV_WORKER = true
using Revise

const _DEV_SOLVER = joinpath(ENV["BLAB_DEV_SOLVER_DIR"], "coupled_solver.jl")
Revise.includet(_DEV_SOLVER)   # tracks the script itself; its entry point skips `run_worker` here

# `includet` is not recursive, so track each engine file in the module it is evaluated in.
const _DEV_INCLUDE = r"include\((?:joinpath\(@__DIR__,\s*(\"src\",\s*)?)?\"(\w+\.jl)\"\)?\)"

function _dev_track_includes(file, mod)
    text = read(file, String)
    header = match(r"^(?:bare)?module\s+(\w+)"m, text)
    inner = header === nothing ? mod : getfield(mod, Symbol(header[1]))
    for inc in eachmatch(_DEV_INCLUDE, text)
        path = joinpath(dirname(file), inc[1] === nothing ? "" : "src", inc[2])
        name = inc[2]
        (isfile(path) && !occursin("Cuda", name) && !occursin("Rocm", name)) || continue
        child = match(r"^(?:bare)?module\s+(\w+)"m, read(path, String))
        child === nothing || isdefined(inner, Symbol(child[1])) || continue   # never loaded
        Revise.track(inner, path)
        _dev_track_includes(path, inner)
    end
end

_dev_track_includes(_DEV_SOLVER, Main)

function _dev_revise()
    Revise.revise(; throw=true)
    return nothing
end

run_worker()
