# Does MUMPS page-fault its factor workspace on every factorization, and does a persistent WK_USER help?
# Usage: julia --project=../src/beat_engine/julia_metal --threads=8 mumps_wk_micro.jl <fem.jls>
using LinearAlgebra, SparseArrays, Serialization
const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
include(joinpath(@__DIR__, "..", "src", "beat_engine", "julia_local", "src", "BeatEngineMumps.jl"))
using .BeatEngineMumps
const BM = BeatEngineMumps
lib = BM.mumps_library()
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")
d = deserialize(ARGS[1]); A = d.fem_system; retained = d.retained
function faults()
    ru = zeros(UInt8, 256); ccall(:getrusage, Cint, (Cint, Ptr{UInt8}), 0, ru)
    (reinterpret(Int64, ru[65:72])[1], reinterpret(Int64, ru[17:24])[1] + reinterpret(Int32, ru[25:28])[1] / 1e6)
end
tol = 64 * eps(Float32)
for mode in (:default, :wk_user, :default)
    s = MumpsSchurSolver(lib)
    mumps_analyse!(s, A, retained)
    wk = ComplexF64[]
    if mode == :wk_user
        info8 = s.struc.info[8]; need = info8 >= 0 ? Int(info8) : -Int(info8) * 1_000_000
        need = Int(ceil(need * (1 + BM.icntl(s.struc, 14) / 100)))
        wk = zeros(ComplexF64, need)
        s.struc.wk_user = pointer(wk)
        s.struc.lwk_user = need < typemax(Int32) ? Int32(need) : Int32(-cld(need, 1_000_000))
        println("wk_user: INFO(8)=$info8 icntl14=$(BM.icntl(s.struc, 14)) -> $(need) entries ($(round(16need/2^20, digits=1)) MiB)")
    end
    ts = Float64[]; fl = Int[]; st = Float64[]
    GC.@preserve wk for k in 1:8
        f0, s0 = faults(); t = @elapsed mumps_factorize!(s, A; symmetry_tolerance=tol); f1, s1 = faults()
        push!(ts, t); push!(fl, f1 - f0); push!(st, s1 - s0)
    end
    println(rpad(mode, 9), " fac ms ", round.(ts[2:end] .* 1e3, digits=1), "  minflt ", fl[2:end], "  sys ms ", round.(st[2:end] .* 1e3, digits=1))
    mumps_release!(s)
end
