# ComplexF32 LU of the n=3116 coupled system: Accelerate's cgetrf vs a right-looking blocked LU
# whose trailing updates are Accelerate cgemm calls (getrf on the tall panel, threaded row swaps,
# trsm, gemm). cgetrf ran at ~0.42 TFLOP/s in the solve (0.19 s); cgemm is the ceiling.
# Usage: julia --threads=10 lu_micro.jl
using LinearAlgebra, Random
using LinearAlgebra: BlasInt

const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")

best(f, reps=5) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

function _swap_rows!(A, ipiv, k, ke, columns)
    Threads.@threads for j in columns
        @inbounds for i in k:ke
            r = ipiv[i]
            r == i && continue
            A[i, j], A[r, j] = A[r, j], A[i, j]
        end
    end
end

# Right-looking blocked LU of an m×w matrix (m ≥ w): panels of `nb` columns, each factored by
# `inner` (Accelerate getrf when `inner == 0`, else this function with block `inner`).
function blocked_lu!(A::AbstractMatrix{T}, nb, inner=0) where {T}
    m, w = size(A)
    ipiv = Vector{BlasInt}(undef, w)
    for k in 1:nb:w
        ke = min(k + nb - 1, w)
        panel = view(A, k:m, k:ke)
        p = inner == 0 ? LAPACK.getrf!(panel; check=false)[2] : blocked_lu!(panel, inner).ipiv
        ipiv[k:ke] .= p .+ (k - 1)
        _swap_rows!(A, ipiv, k, ke, 1:(k-1))
        ke < w || continue
        _swap_rows!(A, ipiv, k, ke, (ke+1):w)
        BLAS.trsm!('L', 'L', 'N', 'U', one(T), view(A, k:ke, k:ke), view(A, k:ke, (ke+1):w))
        BLAS.gemm!('N', 'N', -one(T), view(A, (ke+1):m, k:ke), view(A, k:ke, (ke+1):w), one(T),
                   view(A, (ke+1):m, (ke+1):w))
    end
    return LU(A, ipiv, BlasInt(0))
end

Random.seed!(1)
n = 3116
A = randn(ComplexF32, n, n)
b = randn(ComplexF32, n)
flops = 8 / 3 * n^3   # complex LU, real flops
backward(F) = (x = F \ b; norm(A * x - b) / (opnorm(A, Inf) * norm(x)))

G = randn(ComplexF32, n, n); C = similar(G)
t = best(() -> mul!(C, G, G))
println(rpad("cgemm n^3", 24), round(t * 1e3; digits=1), " ms  ", round(8n^3 / t / 1e12; digits=2), " TFLOP/s")

t = best(() -> lu!(copy(A)))
println(rpad("cgetrf (Accelerate)", 24), round(t * 1e3; digits=1), " ms  ", round(flops / t / 1e12; digits=2),
        " TFLOP/s  berr ", backward(lu(A)))
reference = lu(A)
for (nb, inner) in ((512, 0), (768, 0), (1024, 0), (512, 64), (512, 128), (768, 128), (1024, 128), (1024, 256), (1560, 256))
    local t = best(() -> blocked_lu!(copy(A), nb, inner))
    F = blocked_lu!(copy(A), nb, inner)
    println(rpad("blocked nb=$nb/$inner", 24), round(t * 1e3; digits=1), " ms  ", round(flops / t / 1e12; digits=2),
            " TFLOP/s  berr ", backward(F))
end
