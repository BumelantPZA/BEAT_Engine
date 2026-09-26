# ComplexF64 A (3110×1602) * B (1602×1602): Accelerate zgemm vs 3M (three dgemms on split parts).
# Usage: julia --threads=10 gemm3m_micro.jl
using LinearAlgebra, Random
const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")
best(f, reps=5) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

function split_parts(A)
    R = Matrix{Float64}(undef, size(A)); I = similar(R); S = similar(R)
    Threads.@threads for j in axes(A, 2)
        @inbounds for i in axes(A, 1)
            a = A[i, j]; R[i, j] = real(a); I[i, j] = imag(a); S[i, j] = real(a) + imag(a)
        end
    end
    R, I, S
end
function mul3m(A, B)
    Ar, Ai, As = split_parts(A); Br, Bi, Bs = split_parts(B)
    T1 = Ar * Br; T2 = Ai * Bi; T3 = As * Bs
    C = Matrix{ComplexF64}(undef, size(A, 1), size(B, 2))
    Threads.@threads for j in axes(C, 2)
        @inbounds for i in axes(C, 1)
            C[i, j] = complex(T1[i, j] - T2[i, j], T3[i, j] - T1[i, j] - T2[i, j])
        end
    end
    C
end
Random.seed!(1)
A = randn(ComplexF64, 3110, 1602); B = randn(ComplexF64, 1602, 1602)
C = A * B
println("zgemm   ", round(best(() -> A * B) * 1e3; digits=1), " ms")
println("3M      ", round(best(() -> mul3m(A, B)) * 1e3; digits=1), " ms  relerr ", norm(mul3m(A, B) - C) / norm(C))
Ar = real.(A); Br = real.(B)
println("dgemm   ", round(best(() -> Ar * Br) * 1e3; digits=1), " ms (one real product)")
