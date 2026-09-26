# Factor the SAWMOD FEM matrix in a loop (for `sample`). Usage: ... mumps_loop_micro.jl <fem.jls> [count=60]
using LinearAlgebra, SparseArrays, Serialization
const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
include(joinpath(@__DIR__, "..", "src", "beat_engine", "julia_local", "src", "BeatEngineMumps.jl"))
using .BeatEngineMumps
lib = BeatEngineMumps.mumps_library()
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")
d = deserialize(ARGS[1]); A = d.fem_system; retained = d.retained
ENV["BLAB_TEST_MUMPS_WK"] = "1"; ENV["BLAB_TEST_MUMPS_CNTL"] = "1=0"
s = MumpsSchurSolver(lib); mumps_analyse!(s, A, retained)
mumps_factorize!(s, A; symmetry_tolerance=64 * eps(Float32))
println("READY"); flush(stdout)
t = @elapsed for _ in 1:parse(Int, get(ARGS, 2, "60"))
    mumps_factorize!(s, A; symmetry_tolerance=64 * eps(Float32))
end
println("loop ", t)
