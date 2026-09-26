# Stale-GMRES per-iteration pieces at n=3116, 3 RHS: F64 dense matvec plain vs row-chunked over tasks.
# Usage: julia --threads=10 matvec_micro.jl
using LinearAlgebra
const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")
best(f, reps=20) = (f(); minimum(@elapsed(f()) for _ in 1:reps))
function chunked!(y, A, x, alpha, beta, nparts)
    parts = collect(Iterators.partition(axes(A, 1), cld(size(A, 1), nparts)))
    Threads.@threads for rows in parts
        mul!(view(y, rows, :), view(A, rows, :), x, alpha, beta)
    end
    y
end
# column-chunked: each task a block of columns, partial sums reduced
function colchunked!(y, A, x, nparts, tmp)
    parts = collect(Iterators.partition(axes(A, 2), cld(size(A, 2), nparts)))
    Threads.@threads for p in eachindex(parts)
        cols = parts[p]
        mul!(view(tmp, :, :, p), view(A, :, cols), view(x, cols, :))
    end
    sum!(y, view(tmp, :, :, 1:length(parts)))
    y
end
n = 3116
for T in (ComplexF64, ComplexF32)
    A = randn(T, n, n); x = randn(T, n, 3); y = zeros(T, n, 3)
    println(T)
    println("  plain mul!      ", round(best(() -> mul!(y, A, x)) * 1e3, digits=2), " ms")
    for p in (4, 8, 10, 20)
        println("  row-chunked $p  ", round(best(() -> chunked!(y, A, x, true, false, p)) * 1e3, digits=2), " ms")
    end
    tmp = zeros(T, n, 3, 10)
    println("  col-chunked 10  ", round(best(() -> colchunked!(y, A, x, 10, tmp)) * 1e3, digits=2), " ms")
end
