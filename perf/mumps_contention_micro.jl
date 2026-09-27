# R17-3 micro: does a GMRES-like dense loop beside the MUMPS factorization slow MUMPS less when the loop
# runs on OpenBLAS (NEON) or plain Julia instead of Accelerate (AMX)? MUMPS itself stays on Accelerate.
# Loop = 9 x (ComplexF32 LU solve as FAST_TRS nb=128 blocked trsm + gemm, 3 columns; then an F64
# 3116^2 x 3 zgemm), like one stale-GMRES solve. Times MUMPS alone, the loop alone, and both at once.
# Usage: julia --project=../src/beat_engine/julia_metal --threads=10 mumps_contention_micro.jl <fem.jls> [trials=7] [iterations=9]
using LinearAlgebra, SparseArrays, Serialization, Statistics, Libdl, Random
const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
include(joinpath(@__DIR__, "..", "src", "beat_engine", "julia_local", "src", "BeatEngineMumps.jl"))
using .BeatEngineMumps
lib = BeatEngineMumps.mumps_library()
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")
d = deserialize(ARGS[1]); A = d.fem_system; retained = d.retained
trials = parse(Int, get(ARGS, 2, "7"))
ENV["BLAB_TEST_MUMPS_WK"] = "1"; ENV["BLAB_TEST_MUMPS_CNTL"] = "1=0"
tol = 64 * eps(Float32)
s = MumpsSchurSolver(lib); mumps_analyse!(s, A, retained)
mumps_factorize!(s, A; symmetry_tolerance=tol)

# --- OpenBLAS64 by direct ccall (not through LBT, which stays on Accelerate) ---
using OpenBLAS_jll; const OB = OpenBLAS_jll.libopenblas_handle   # the copy Julia loaded
const OB_THREADS = dlsym(OB, :openblas_set_num_threads64_)
ob_threads!(n) = ccall(OB_THREADS, Cvoid, (Cint,), n)   # int by value, not a Fortran pointer
for (T, gemm, trsm) in ((ComplexF64, :zgemm_64_, :ztrsm_64_), (ComplexF32, :cgemm_64_, :ctrsm_64_))
    @eval function ob_gemm!(alpha::$T, A::StridedMatrix{$T}, B::StridedMatrix{$T}, beta::$T, C::StridedMatrix{$T})
        m, k = size(A); n = size(B, 2)
        ccall(dlsym(OB, $(QuoteNode(gemm))), Cvoid,
              (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{$T}, Ptr{$T}, Ref{Int64},
               Ptr{$T}, Ref{Int64}, Ref{$T}, Ptr{$T}, Ref{Int64}, Clong, Clong),
              UInt8('N'), UInt8('N'), m, n, k, alpha, A, stride(A, 2), B, stride(B, 2), beta, C, stride(C, 2), 1, 1)
        C
    end
    @eval function ob_trsm!(uplo::Char, diag::Char, A::StridedMatrix{$T}, B::StridedMatrix{$T})
        m, n = size(B)
        ccall(dlsym(OB, $(QuoteNode(trsm))), Cvoid,
              (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ref{$T}, Ptr{$T},
               Ref{Int64}, Ptr{$T}, Ref{Int64}, Clong, Clong, Clong, Clong),
              UInt8('L'), UInt8(uplo), UInt8('N'), UInt8(diag), m, n, one($T), A, stride(A, 2), B, stride(B, 2), 1, 1, 1, 1)
        B
    end
end

# --- plain threaded Julia (no BLAS) ---
# C[rows, :] = beta C + alpha A[rows, :] B, rows split over spawned tasks (thread 1 is inside MUMPS).
function jl_gemm!(alpha::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, beta::T, C::AbstractMatrix{T}; chunks=8) where {T}
    m = size(A, 1)
    parts = Iterators.partition(1:m, cld(m, chunks))
    @sync for rows in parts
        Threads.@spawn begin
            @inbounds for c in axes(B, 2)
                if beta != one(T)
                    @simd for i in rows; C[i, c] *= beta; end
                end
            end
            @inbounds for j in axes(A, 2)
                b1 = alpha * B[j, 1]; b2 = alpha * B[j, 2]; b3 = alpha * B[j, 3]
                @simd for i in rows
                    a = A[i, j]
                    C[i, 1] += a * b1; C[i, 2] += a * b2; C[i, 3] += a * b3
                end
            end
        end
    end
    C
end
function jl_trsm!(uplo::Char, diag::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    @inbounds for c in axes(B, 2)
        if uplo == 'L'
            for j in 1:n
                diag == 'N' && (B[j, c] /= A[j, j]); x = B[j, c]
                @simd for i in (j+1):n; B[i, c] -= A[i, j] * x; end
            end
        else
            for j in n:-1:1
                diag == 'N' && (B[j, c] /= A[j, j]); x = B[j, c]
                @simd for i in 1:(j-1); B[i, c] -= A[i, j] * x; end
            end
        end
    end
    B
end

acc_gemm!(alpha, A, B, beta, C) = BLAS.gemm!('N', 'N', alpha, A, B, beta, C)
acc_trsm!(uplo, diag, A, B) = BLAS.trsm!('L', uplo, 'N', diag, one(eltype(A)), A, B)

# FAST_TRS's ComplexF32 LU solve with the given kernels.
function lu_solve(F, rhs, gemm!, trsm!; nb=128)
    B = ComplexF32.(rhs); L = F.factors; n = size(L, 1)
    @inbounds for i in 1:n
        r = F.ipiv[i]; r == i && continue
        for c in axes(B, 2); B[i, c], B[r, c] = B[r, c], B[i, c]; end
    end
    for k in 1:nb:n
        ke = min(k + nb - 1, n)
        trsm!('L', 'U', view(L, k:ke, k:ke), view(B, k:ke, :))
        ke < n && gemm!(-one(ComplexF32), view(L, (ke+1):n, k:ke), view(B, k:ke, :), one(ComplexF32), view(B, (ke+1):n, :))
    end
    for ke in n:-nb:1
        k = max(ke - nb + 1, 1)
        trsm!('U', 'N', view(L, k:ke, k:ke), view(B, k:ke, :))
        k > 1 && gemm!(-one(ComplexF32), view(L, 1:(k-1), k:ke), view(B, k:ke, :), one(ComplexF32), view(B, 1:(k-1), :))
    end
    B
end
const ITS = parse(Int, get(ARGS, 3, "9"))   # loop iterations (3rd argument)
function gmres_like(M64, F, X, W, gemm!, trsm!; its=ITS)
    for _ in 1:its
        P = ComplexF64.(lu_solve(F, X, gemm!, trsm!))
        gemm!(one(ComplexF64), M64, P, zero(ComplexF64), W)
        X .= W ./ norm(W)
    end
    W
end

n = 3116
Random.seed!(1)
M64 = randn(ComplexF64, n, n) ./ sqrt(n) + 4I
F = lu(ComplexF32.(M64))
X0 = randn(ComplexF64, n, 3); W = similar(X0)
variants = [
    ("a_accelerate", () -> nothing, acc_gemm!, acc_trsm!),
    ("b_openblas4", () -> ob_threads!(4), ob_gemm!, ob_trsm!),
    ("b_openblas8", () -> ob_threads!(8), ob_gemm!, ob_trsm!),
    ("c_julia", () -> nothing, jl_gemm!, jl_trsm!),
]
# correctness of the kernels against Accelerate (same maths, F32 rounding)
ref = gmres_like(M64, F, copy(X0), similar(X0), acc_gemm!, acc_trsm!)
for (name, setup, g, t) in variants[2:end]
    setup(); r = gmres_like(M64, F, copy(X0), similar(X0), g, t)
    println("check $name maxrel ", maximum(abs, r - ref) / maximum(abs, ref))
end
factor() = @elapsed mumps_factorize!(s, A; symmetry_tolerance=tol)
res = Dict(name => (alone=Float64[], loop=Float64[], mumps_beside=Float64[], loop_beside=Float64[]) for (name,) in variants)
for trial in 0:trials
    for (name, setup, g, t) in variants
        setup()
        a = factor()
        l = @elapsed gmres_like(M64, F, copy(X0), W, g, t)
        GC.gc(false)
        task = Threads.@spawn @elapsed gmres_like(M64, F, copy(X0), W, g, t)
        mb = factor()
        lb = fetch(task)
        trial == 0 && continue                   # warm-up round
        r = res[name]; push!(r.alone, a); push!(r.loop, l); push!(r.mumps_beside, mb); push!(r.loop_beside, lb)
    end
end
ms(x) = round(1e3 * median(x), digits=1)
println("variant        | MUMPS alone | loop alone | MUMPS beside | loop beside | MUMPS +ms")
for (name,) in variants
    r = res[name]
    println(rpad(name, 14), " | ", ms(r.alone), " | ", ms(r.loop), " | ", ms(r.mumps_beside), " | ", ms(r.loop_beside),
            " | ", round(ms(r.mumps_beside) - ms(r.alone), digits=1))
end
