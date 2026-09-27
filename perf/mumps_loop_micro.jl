# Factor the SAWMOD FEM matrix in a loop (for `sample`). Usage: ... mumps_loop_micro.jl <fem.jls> [count=60] [reduce.jls]
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
# R17-6: with a BLAB_TEST_DUMP_REDUCE file as 3rd argument, time the transducer reduction (+ its expansion,
# as BLAB_TEST_MUMPS_EXPAND does) dense vs BLAB_TEST_MUMPS_SPARSE_RHS=1 after each factorization.
if length(ARGS) >= 3
    using Statistics
    cols = deserialize(ARGS[3])
    println("reduce columns $(size(cols)), nonzero rows $(count(r -> any(!iszero, view(cols, r, :)), axes(cols, 1)))")
    zero_x = zeros(ComplexF64, length(s.schur_variables), size(cols, 2))
    t = Dict(m => (reduce=Float64[], expand=Float64[]) for m in ("0", "1")); out = Dict()
    for trial in 0:parse(Int, get(ARGS, 2, "60")) ÷ 4, m in ("0", "1")
        ENV["BLAB_TEST_MUMPS_SPARSE_RHS"] = m
        mumps_factorize!(s, A; symmetry_tolerance=64 * eps(Float32))
        tr = @elapsed r = mumps_reduce(s, cols)
        te = @elapsed e = mumps_expand(s, zero_x)
        out[m] = (r, e)
        trial > 0 && (push!(t[m].reduce, tr); push!(t[m].expand, te))
    end
    for m in ("0", "1")
        println("sparse_rhs=$m reduce $(round(1e3median(t[m].reduce), digits=2)) ms, expand $(round(1e3median(t[m].expand), digits=2)) ms")
    end
    rel(a, b) = maximum(abs, a - b) / maximum(abs, b)
    println("sparse vs dense maxrel: reduce $(rel(out["1"][1], out["0"][1])), expand $(rel(out["1"][2], out["0"][2]))")
end
