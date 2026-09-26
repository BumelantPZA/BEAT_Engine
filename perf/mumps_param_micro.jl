# MUMPS parameter study on a BLAB_TEST_DUMP_FEM dump: factorization time, flops and Schur maxrel per variant.
# Usage: julia --project=../src/beat_engine/julia_metal --threads=8 mumps_param_micro.jl <fem.jls> "label|ICNTL spec|CNTL spec" ...
# Specs as in BLAB_TEST_MUMPS_ICNTL / _CNTL ("7=5,35=2"). Every variant also gets CNTL 1=0 and MUMPS_WK=1.
using LinearAlgebra, SparseArrays, Serialization, Statistics, Libdl
const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
include(joinpath(@__DIR__, "..", "src", "beat_engine", "julia_local", "src", "BeatEngineMumps.jl"))
using .BeatEngineMumps
lib = BeatEngineMumps.mumps_library()
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")
# MICRO_OBLAS="ztrsm_,zgemm_": these LP64 routines go to OpenBLAS32 instead of Accelerate.
let names = split(get(ENV, "MICRO_OBLAS", ""), ','; keepempty=false)
    if !isempty(names)
        m = Base.require(BeatEngineMumps.OPENBLAS32_PKGID)
        h = Libdl.dlopen(Base.invokelatest(getproperty, m, :libopenblas_path))
        for n in names
            r = BLAS.lbt_set_forward(String(n), Libdl.dlsym(h, n), :lp64); r == 0 || error("lbt_set_forward($n) = $r")
        end
        println("OpenBLAS for ", names)
    end
end
d = deserialize(ARGS[1]); A = d.fem_system; retained = d.retained
ENV["BLAB_TEST_MUMPS_WK"] = "1"
tol = 64 * eps(Float32)
ref = nothing
for spec in ARGS[2:end]
    label, icntl, cntl, keep = (split(spec, '|')..., "", "", "")[1:4]
    ENV["MICRO_KEEP"] = keep
    ENV["BLAB_TEST_MUMPS_SINGLE"] = startswith(label, "single") ? "1" : "0"
    ENV["BLAB_TEST_MUMPS_ICNTL"] = icntl
    ENV["BLAB_TEST_MUMPS_CNTL"] = isempty(cntl) ? "1=0" : "1=0," * cntl
    s = MumpsSchurSolver(lib)
    # MICRO_KEEP / per-variant 4th field "i=v,...": MUMPS KEEP entries written after initialization.
    for item in split(get(ENV, "MICRO_KEEP", ""), ','; keepempty=false)
        i, v = parse.(Int, split(item, '='))
        GC.@preserve s unsafe_store!(BeatEngineMumps._field_pointer(s.struc, :keep, Int32), Int32(v), i)
    end
    keep_now = [s.struc.keep[i] for i in 1:12]
    ta = @elapsed mumps_analyse!(s, A, retained)
    ts = Float64[]; S = nothing
    try
        for k in 1:6
            t = @elapsed S = mumps_factorize!(s, A; symmetry_tolerance=tol)
            k > 1 && push!(ts, t)
        end
    catch e
        println(rpad(label, 22), " FAILED: ", sprint(showerror, e)[1:min(end, 120)]); mumps_release!(s); continue
    end
    global ref = something(ref, S)
    err = maximum(abs, S - ref) / maximum(abs, ref)
    println(rpad(label, 22), " fac ", lpad(round(median(ts) * 1e3, digits=1), 6), " ms  (min ", round(minimum(ts) * 1e3, digits=1),
            ")  analyse ", round(ta, digits=2), " s  flops ", round(s.struc.rinfog[1] / 1e9, digits=2), " G  entries ",
            round(BeatEngineMumps.infog(s.struc, 29) / 1e6, digits=2), " M  schur maxrel ", round(err, sigdigits=2), "  keep[1:12] ", keep_now)
    mumps_release!(s)
end
