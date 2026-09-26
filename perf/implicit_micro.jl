# Cost of the ELIM_IMPLICIT residual pieces at SAWMOD size with Accelerate: the dense matvec
# (3116², 3 RHS), the correction B (3110×1602) * (W (1602²) * g), and row-chunked threaded variants.
# Usage: julia --threads=10 perf/implicit_micro.jl
using LinearAlgebra, Random
const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")
best(f, reps=7) = (f(); minimum(@elapsed(f()) for _ in 1:reps))
ms(t) = round(t * 1e3; digits=2)

# y .= alpha*A*x .+ beta*y, rows split over tasks, one BLAS call each.
function chunked_mul!(y, A, x, alpha, beta; chunks=Threads.nthreads())
    parts = collect(Iterators.partition(axes(A, 1), cld(size(A, 1), chunks)))
    Threads.@threads for r in parts
        mul!(view(y, r, :), view(A, r, :), x, alpha, beta)
    end
    return y
end

Random.seed!(1)
n, m, k, r = 3116, 3110, 1602, 3
A = randn(ComplexF64, n, n); B = randn(ComplexF64, m, k); W = randn(ComplexF64, k, k)
x = randn(ComplexF64, n, r); y = zeros(ComplexF64, n, r); g = randn(ComplexF64, k, r); t = W * g
ym = zeros(ComplexF64, m, r)
println("BLAS threads ", BLAS.get_num_threads())
println("A*x        ", ms(best(() -> mul!(y, A, x))), " ms   chunked ", ms(best(() -> chunked_mul!(y, A, x, 1.0, 0.0))))
println("W*g        ", ms(best(() -> mul!(t, W, g))), " ms   chunked ", ms(best(() -> chunked_mul!(t, W, g, 1.0, 0.0))))
println("B*t        ", ms(best(() -> mul!(ym, B, t))), " ms   chunked ", ms(best(() -> chunked_mul!(ym, B, t, 1.0, 0.0))))
B32 = ComplexF32.(B); W32 = ComplexF32.(W); A64 = copy(A)
println("F32 B*W    ", ms(best(() -> B32 * W32)), " ms (cgemm)   F64 B*W ", ms(best(() -> B * W)), " ms (zgemm)")
println("F32 narrow ", ms(best(() -> ComplexF32.(A64))), " ms")
