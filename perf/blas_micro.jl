# Dense kernels on the SAWMOD critical path: complex LU of the n=3116 coupled
# system (ComplexF32 and ComplexF64) and a complex GEMM of the interface-product
# shape, timed with Julia's default OpenBLAS and then with Apple Accelerate.
# Usage: julia --threads=10 blas_micro.jl [openblas|accelerate]
using LinearAlgebra, Random

const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"

function use_accelerate!()
    BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
    BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")
end

function best(f, reps=5)
    f()
    minimum(@elapsed(f()) for _ in 1:reps)
end

mode = isempty(ARGS) ? "openblas" : ARGS[1]
mode == "accelerate" ? use_accelerate!() : BLAS.set_num_threads(Threads.nthreads() - 1)
println("mode=$mode ", [(l.libname |> basename, l.interface) for l in BLAS.get_config().loaded_libs],
        " blas_threads=", BLAS.get_num_threads())

Random.seed!(1)
n = 3116
for T in (ComplexF32, ComplexF64)
    A = randn(T, n, n) + n * I
    t = best(() -> lu(A))
    x = lu(A) \ ones(T, n)
    resid = norm(A * x - ones(T, n)) / norm(ones(T, n))
    println(rpad("lu $T n=$n", 28), round(t * 1000; digits=1), " ms   resid=", resid)
end
for T in (ComplexF32, ComplexF64)
    A = randn(T, 3110, 3110); B = randn(T, 3110, 600)
    t = best(() -> A * B)
    println(rpad("gemm $T 3110x3110*600", 28), round(t * 1000; digits=1), " ms")
end
