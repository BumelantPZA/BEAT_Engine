# Where the blocked F32 LU (nb=512) spends its time: panel getrf, row swaps, trsm, gemm.
# Usage: julia --threads=10 lu_split_micro.jl
using LinearAlgebra, Random
using LinearAlgebra: BlasInt
const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")
function swap_rows!(A, ipiv, k, ke, columns)
    Threads.@threads for j in columns
        @inbounds for i in k:ke
            r = ipiv[i]; r == i && continue
            A[i, j], A[r, j] = A[r, j], A[i, j]
        end
    end
end
function blocked_lu!(A::AbstractMatrix{T}, nb, t) where {T}
    m, w = size(A); ipiv = Vector{BlasInt}(undef, w)
    for k in 1:nb:w
        ke = min(k + nb - 1, w)
        t[1] += @elapsed p = LAPACK.getrf!(view(A, k:m, k:ke); check=false)[2]
        ipiv[k:ke] .= p .+ (k - 1)
        t[2] += @elapsed swap_rows!(A, ipiv, k, ke, 1:(k-1))
        ke < w || continue
        t[2] += @elapsed swap_rows!(A, ipiv, k, ke, (ke+1):w)
        t[3] += @elapsed BLAS.trsm!('L', 'L', 'N', 'U', one(T), view(A, k:ke, k:ke), view(A, k:ke, (ke+1):w))
        t[4] += @elapsed BLAS.gemm!('N', 'N', -one(T), view(A, (ke+1):m, k:ke), view(A, k:ke, (ke+1):w), one(T), view(A, (ke+1):m, (ke+1):w))
    end
    ipiv
end
Random.seed!(1); n = 3116; A = randn(ComplexF32, n, n)
for nb in (512, 256, 384)
    t = zeros(4); blocked_lu!(copy(A), nb, t); fill!(t, 0)
    for _ in 1:5; blocked_lu!(copy(A), nb, t); end
    println("nb=$nb  getrf $(round(t[1]/5*1e3,digits=1))  swaps $(round(t[2]/5*1e3,digits=1))  trsm $(round(t[3]/5*1e3,digits=1))  gemm $(round(t[4]/5*1e3,digits=1)) ms")
end
# panel alone: tall-skinny getrf vs recursive (split columns in half, trsm + gemm)
function rec_getrf!(P::AbstractMatrix{T}, ipiv, off) where {T}
    m, w = size(P)
    if w <= 32
        p = LAPACK.getrf!(P; check=false)[2]; ipiv[off+1:off+w] .= p; return
    end
    h = w ÷ 2
    rec_getrf!(view(P, :, 1:h), ipiv, off)
    for i in 1:h  # apply left pivots to right half
        r = ipiv[off+i]; r == i && continue
        for j in h+1:w; P[i, j], P[r, j] = P[r, j], P[i, j]; end
    end
    BLAS.trsm!('L', 'L', 'N', 'U', one(T), view(P, 1:h, 1:h), view(P, 1:h, h+1:w))
    BLAS.gemm!('N', 'N', -one(T), view(P, h+1:m, 1:h), view(P, 1:h, h+1:w), one(T), view(P, h+1:m, h+1:w))
    sub = view(P, h+1:m, h+1:w)
    rec_getrf!(sub, ipiv, off + h)
    for i in 1:(w-h); ipiv[off+h+i] += h; end
    for i in h+1:w  # apply right pivots to left half
        r = ipiv[off+i]; r == i && continue
        for j in 1:h; P[i, j], P[r, j] = P[r, j], P[i, j]; end
    end
end
Pn = randn(ComplexF32, n, 512)
tg = minimum(@elapsed(LAPACK.getrf!(copy(Pn); check=false)) for _ in 1:5)
ip = zeros(BlasInt, 512)
tr = minimum(@elapsed(rec_getrf!(copy(Pn), ip, 0)) for _ in 1:5)
Q1 = copy(Pn); p1 = LAPACK.getrf!(Q1; check=false)[2]; Q2 = copy(Pn); rec_getrf!(Q2, ip, 0)
println("panel 3116x512: getrf $(round(tg*1e3,digits=1)) ms, recursive $(round(tr*1e3,digits=1)) ms, same pivots $(p1 == ip), maxdiff $(maximum(abs, Q1 - Q2))")

# Look-ahead (depth 1): after panel k's swaps and trsm, update panel k+1's columns first, factor it on
# its own task while the trailing gemm for the columns beyond it runs.
function lookahead_lu!(A::AbstractMatrix{T}, nb) where {T}
    m, w = size(A); ipiv = Vector{BlasInt}(undef, w)
    k = 1; ke = min(nb, w)
    p = LAPACK.getrf!(view(A, k:m, k:ke); check=false)[2]
    while true
        ipiv[k:ke] .= p .+ (k - 1)
        ke < w || break
        swap_rows!(A, ipiv, k, ke, (ke+1):w)
        BLAS.trsm!('L', 'L', 'N', 'U', one(T), view(A, k:ke, k:ke), view(A, k:ke, (ke+1):w))
        k2 = ke + 1; ke2 = min(ke + nb, w)
        L = view(A, (ke+1):m, k:ke)
        BLAS.gemm!('N', 'N', -one(T), L, view(A, k:ke, k2:ke2), one(T), view(A, (ke+1):m, k2:ke2))
        panel = Threads.@spawn LAPACK.getrf!(view(A, k2:m, k2:ke2); check=false)[2]
        ke2 < w && BLAS.gemm!('N', 'N', -one(T), L, view(A, k:ke, (ke2+1):w), one(T), view(A, (ke+1):m, (ke2+1):w))
        p = fetch(panel)
        k, ke = k2, ke2
    end
    # left swaps, once per panel, after all panels are done
    for k in 1:nb:w
        ke = min(k + nb - 1, w)
        k > 1 && swap_rows!(A, ipiv, k, ke, 1:(k-1))
    end
    ipiv
end
b = randn(ComplexF32, n)
for nb in (512, 384, 256)
    Q = copy(A); lookahead_lu!(Q, nb)
    t = minimum(@elapsed(lookahead_lu!(copy(A), nb)) for _ in 1:5)
    t0 = minimum(@elapsed(blocked_lu!(copy(A), nb, zeros(4))) for _ in 1:5)
    Q = copy(A); ip = lookahead_lu!(Q, nb); F = LU(Q, ip, BlasInt(0)); x = F \ b
    R = copy(A); ip0 = blocked_lu!(R, nb, zeros(4))
    println("nb=$nb lookahead $(round(t*1e3,digits=1)) ms vs blocked $(round(t0*1e3,digits=1)) ms  berr $(norm(A*x-b)/(opnorm(A,Inf)*norm(x)))  same pivots $(ip == ip0)  maxdiff $(maximum(abs, Q - R))")
end
