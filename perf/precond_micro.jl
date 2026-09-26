# Offline check of Fable idea I8: the previous frequency's ComplexF32 LU as a right preconditioner
# for GMRES on the next frequency's dense system. Input: BLAB_TEST_DUMP_DENSE dumps (n, k, A, B).
# Usage: julia --threads=10 precond_micro.jl <dir> <prev>:<next> ...
using LinearAlgebra
const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")

function load(path)
    open(path) do io
        n, k = read(io, Int64), read(io, Int64)
        A = Matrix{ComplexF64}(undef, n, n); read!(io, A)
        B = Matrix{ComplexF64}(undef, n, k); read!(io, B)
        A, B
    end
end

# GMRES (MGS, no restart) on A M⁻¹ y = b, x = M⁻¹ y; stops when the true backward-error ratio
# (RefinedDenseLU's test) is ≤ 1. Returns iterations used (matvecs), or -1.
function gmres_iters(A, F, b, threshold; maxit=60)
    n = length(b)
    prec(v) = ComplexF64.(F \ ComplexF32.(v))
    x0 = prec(b)                       # initial guess: one preconditioned solve
    r = b - A * x0
    beta = norm(r)
    V = zeros(ComplexF64, n, maxit + 1); H = zeros(ComplexF64, maxit + 1, maxit)
    V[:, 1] = r / beta
    Z = zeros(ComplexF64, n, maxit)
    for j in 1:maxit
        Z[:, j] = prec(V[:, j])
        w = A * Z[:, j]
        for i in 1:j
            H[i, j] = dot(V[:, i], w); w .-= H[i, j] .* V[:, i]
        end
        H[j+1, j] = norm(w); V[:, j+1] = w / H[j+1, j]
        e = zeros(ComplexF64, j + 1); e[1] = beta
        y = H[1:j+1, 1:j] \ e
        x = x0 + Z[:, 1:j] * y
        res = norm(b - A * x, Inf) / (norm(x, Inf) * threshold)
        res <= 1 && return j, res
    end
    return -1, NaN
end

function refine_iters(A, F, b, threshold)
    x = ComplexF64.(F \ ComplexF32.(b))
    for it in 0:10
        r = b - A * x
        ratio = norm(r, Inf) / (norm(x, Inf) * threshold)
        ratio <= 1 && return it
        x .+= ComplexF64.(F \ ComplexF32.(r))
    end
    return -1
end

dir = ARGS[1]
for pair in ARGS[2:end]
    p, q = split(pair, ":")
    Ap, _ = load(joinpath(dir, "dense_$p.bin"))
    Aq, Bq = load(joinpath(dir, "dense_$q.bin"))
    n = size(Aq, 1)
    threshold = opnorm(Aq, Inf) * eps(Float64) * sqrt(n)
    Fp = lu(ComplexF32.(Ap)); Fq = lu(ComplexF32.(Aq))
    rel = opnorm(Aq - Ap, 1) / opnorm(Aq, 1)
    fresh = [refine_iters(Aq, Fq, Bq[:, c], threshold) for c in axes(Bq, 2)]
    stale_refine = [refine_iters(Aq, Fp, Bq[:, c], threshold) for c in axes(Bq, 2)]
    gm = [gmres_iters(Aq, Fp, Bq[:, c], threshold)[1] for c in axes(Bq, 2)]
    println("$p -> $q  ‖ΔA‖/‖A‖=$(round(rel; sigdigits=3))  fresh-LU refine=$fresh  stale refine=$stale_refine  stale GMRES=$gm")
end
# per-iteration cost
A, B = load(joinpath(dir, "dense_$(split(ARGS[2], ":")[2]).bin")); F = lu(ComplexF32.(A)); v = B[:, 1]
tmv = minimum(@elapsed(A * v) for _ in 1:5); tps = minimum(@elapsed(F \ ComplexF32.(v)) for _ in 1:5)
println("matvec $(round(tmv*1e3; digits=2)) ms, F32 LU solve $(round(tps*1e3; digits=2)) ms per column")
