# Is the coupled system's dense LU faster on the GPU? MPS has only a real Float32 LU, so the
# n=3116 complex system becomes the 2n real embedding [Ar -Ai; Ai Ar]. Compares the current CPU
# path (ComplexF32 lu! with Accelerate) against: build embedding + MPS lu! + factors back to host.
# Usage: julia --threads=10 --project=../src/beat_engine/julia_metal mps_lu_micro.jl
using LinearAlgebra, Random, Metal

const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")

best(f, reps=5) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

function embed!(R, A)
    n = size(A, 1)
    Threads.@threads for j in 1:n
        @inbounds for i in 1:n
            a = A[i, j]
            R[i, j] = real(a); R[i+n, j] = imag(a)
            R[i, j+n] = -imag(a); R[i+n, j+n] = real(a)
        end
    end
    return R
end

Random.seed!(1)
n = 3116
A = randn(ComplexF32, n, n) + n * I
b = randn(ComplexF32, n)

t_cpu = best(() -> lu!(copy(A)))
F = lu!(copy(A))
println("cpu  ComplexF32 lu! n=$n        ", round(1000t_cpu; digits=1), " ms   resid ",
        norm(A * (F \ b) - b) / norm(b))

R = Matrix{Float32}(undef, 2n, 2n)
t_embed = best(() -> embed!(R, A))
dR = MtlMatrix{Float32,Metal.SharedStorage}(undef, 2n, 2n)
hR = unsafe_wrap(Array, dR)           # shared storage: the host writes the GPU buffer directly
t_gpu_only = best(() -> (copyto!(hR, R); lu!(dR); Metal.synchronize()))
t_total = best(() -> begin
    embed!(hR, A)
    G = lu!(dR)
    Metal.synchronize()
    Array(G.p)
end)
embed!(hR, A)
G = lu!(dR)
Gh = LU(hR, Vector{Int}(Array(G.p)), G.info)   # factors already in host-visible memory
rb = vcat(real(b), imag(b))
x = Gh \ rb
xc = complex.(x[1:n], x[n+1:end])
println("gpu  embed (threads)              ", round(1000t_embed; digits=1), " ms")
println("gpu  copy + MPS lu! 2n=$(2n)      ", round(1000t_gpu_only; digits=1), " ms")
println("gpu  embed-in-place + lu! + perm  ", round(1000t_total; digits=1), " ms   resid ",
        norm(A * xc - b) / norm(b))
t_solve_cpu = best(() -> F \ b)
t_solve_gpu = best(() -> Gh \ rb)
println("solve: cpu complex ", round(1000t_solve_cpu; digits=2), " ms, host real-embedded ",
        round(1000t_solve_gpu; digits=2), " ms")
