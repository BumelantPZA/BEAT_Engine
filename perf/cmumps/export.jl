# Export the SAWMOD FEM system (lower triangle, 1-based) and the Float64 MUMPS Schur for cmumps_micro.c.
using LinearAlgebra, SparseArrays, Serialization
const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
include(joinpath(@__DIR__, "..", "..", "src", "beat_engine", "julia_local", "src", "BeatEngineMumps.jl"))
using .BeatEngineMumps
lib = BeatEngineMumps.mumps_library()
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")
d = deserialize(ARGS[1]); A = d.fem_system; retained = d.retained; out = ARGS[2]
ENV["BLAB_TEST_MUMPS_CNTL"] = "1=0"
s = MumpsSchurSolver(lib); mumps_analyse!(s, A, retained); S = mumps_factorize!(s, A; symmetry_tolerance=64 * eps(Float32))
write(joinpath(out, "sizes.bin"), Int64[size(A, 1), length(s.irn), length(retained)])
write(joinpath(out, "irn.bin"), s.irn); write(joinpath(out, "jcn.bin"), s.jcn)
write(joinpath(out, "a.bin"), s.values); write(joinpath(out, "schur_vars.bin"), Int32.(retained))
write(joinpath(out, "schur64.bin"), S)
println("exported n=$(size(A,1)) nz=$(length(s.irn)) schur=$(length(retained))")
