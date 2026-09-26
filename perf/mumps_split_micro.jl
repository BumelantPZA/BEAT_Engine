# Offline MUMPS study on a BLAB_TEST_DUMP_FEM dump: FEM components, full vs per-component Schur
# factorization, and whether separate MUMPS instances can factor concurrently (checked vs sequential).
# Usage: julia --project=../src/beat_engine/julia_metal --threads=8 mumps_split_micro.jl <fem.jls>
using LinearAlgebra, SparseArrays, Serialization
const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
include(joinpath(@__DIR__, "..", "src", "beat_engine", "julia_local", "src", "BeatEngineMumps.jl"))
using .BeatEngineMumps
lib = BeatEngineMumps.mumps_library()           # before the forward (see test_apply_blas!)
BLAS.lbt_forward(ACCELERATE; clear=true, suffix_hint="\x1a\$NEWLAPACK\$ILP64")
BLAS.lbt_forward(ACCELERATE; clear=false, suffix_hint="\x1a\$NEWLAPACK")
d = deserialize(ARGS[1]); A = d.fem_system; retained = d.retained
n = size(A, 1)
function components(A)
    n = size(A, 1); parent = collect(1:n)
    find(v) = (while parent[v] != v; parent[v] = parent[parent[v]]; v = parent[v]; end; v)
    rows = rowvals(A)
    for j in 1:n, e in nzrange(A, j)
        a = find(rows[e]); b = find(j); a == b || (parent[max(a, b)] = min(a, b))
    end
    roots = [find(v) for v in 1:n]
    [findall(==(r), roots) for r in unique(roots)]
end
comps = components(A)
isret = falses(n); isret[retained] .= true
println("n=$n nnz=$(nnz(A)) retained=$(length(retained)) components=$(length(comps))")
parts = []
for c in comps
    r = findall(isret[c])                         # local indices of retained vertices
    println("  component: $(length(c)) vertices, $(length(r)) retained")
    push!(parts, (vertices=c, A=A[c, c], retained=r))
end
tol = 64 * eps(Float32)
function factor(M, ret)
    s = MumpsSchurSolver(lib)
    mumps_analyse!(s, M, ret)
    t = @elapsed S = mumps_factorize!(s, M; symmetry_tolerance=tol)
    t = @elapsed S = mumps_factorize!(s, M; symmetry_tolerance=tol)   # second call: warm
    mumps_release!(s)
    (S, t)
end
S_full, t_full = factor(A, retained)
println("full: $(round(t_full*1e3, digits=1)) ms")
seq = [factor(p.A, p.retained) for p in parts]
println("per component sequential: ", [round(x[2]*1e3, digits=1) for x in seq], " ms")
# per-component Schur blocks must equal the full Schur's diagonal blocks (the off-diagonal ones are 0)
pos = Dict(v => i for (i, v) in enumerate(retained))
for (p, (S, _)) in zip(parts, seq)
    idx = [pos[p.vertices[r]] for r in p.retained]
    println("  block maxrel vs full: ", maximum(abs, S - S_full[idx, idx]) / maximum(abs, S_full[idx, idx]))
end
# concurrent: all components on separate tasks, separate instances, several times
solvers = [MumpsSchurSolver(lib) for _ in parts]
foreach(((s, p),) -> mumps_analyse!(s, p.A, p.retained), zip(solvers, parts))
for trial in 1:5
    t = @elapsed results = fetch.([Threads.@spawn mumps_factorize!(s, p.A; symmetry_tolerance=tol) for (s, p) in zip(solvers, parts)])
    err = maximum(maximum(abs, R - S) / maximum(abs, S) for (R, (S, _)) in zip(results, seq))
    println("concurrent trial $trial: $(round(t*1e3, digits=1)) ms, maxrel vs sequential $err")
end
