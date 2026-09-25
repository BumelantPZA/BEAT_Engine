# Task: make the Metal BEM far-field assembly kernel faster

This branch is BEAT Engine v0.2.0, unmodified except for this file and `fable/KERNEL_EXCERPT.jl`. It's a coupled FEM-BEM
acoustic solver in Julia; the Apple GPU path uses Metal.jl 1.10.3. You're asked for
**architecture, ideas and an implementation plan for making one GPU kernel faster**. You
write no code: another agent implements the plan and benchmarks it on the real machine.

## Hard limits of this session (read first; keep usage small)
- The user wants this to take only a small part of their usage. Work in one pass: read,
  think, write the file once, commit, push, stop. No drafts, no rewrites, no subagents, no
  web searches.
- This sandbox is Linux with no Apple GPU. **Don't install Julia, instantiate environments,
  run tests or the solver, or write code.** Nothing here can run Metal code.
- **Read only `fable/KERNEL_EXCERPT.jl`** (~880 lines, all the relevant code). Open a
  repo file only if the excerpt truly lacks something you need, and then only that
  function. Don't survey the repo.
- Out of scope: batching several frequencies into one pass, moving work between CPU and
  GPU, pipelining or overlap (already done locally), changing quadrature order or accuracy
  settings. Stay inside the core calculation.
- Any change must keep results within float32 noise of the current `pair_gather` path
  (relative L2 ≲ 1e-6 on the operators).

## Machine and problem
- M1 Pro, 16-core GPU (Apple7 family, SIMD width 32, 32 KB threadgroup memory), 16 GB
  unified memory. Peaks: ~5 TFLOPS FP32, ~200 GB/s.
- Per frequency the solver assembles four dense Galerkin operators for G = e^{ikr}/(4πr),
  ComplexF32: single layer S (P1 x DP0), adjoint double layer K' (P1 x DP0), double layer
  D (P1 x P1), hypersingular H (P1 x P1, via surface curls).
- Test mesh: 6,054 triangles, 3,110 P1 nodes. xy symmetry gives 3 mirror images (the trial
  element's coordinates are sign-flipped), so there are 4 full passes over all element
  pairs, ~147 M pairs per frequency. 3-point triangle rule (order 2), so 9 evaluations of
  G per pair (~1.3 G per frequency). Singular and adjacent pairs are skipped here; a
  separate kernel handles them in <0.1 s.

## Current kernel: `pair_gather`, the fastest of 4 existing variants
1. Pair kernel, 2-D grid (test element x trial element). The trial side is chunked by
   `BLAB_METAL_GATHER_BUDGET_MB` (512 → 440 trial elements per chunk, 14 chunks per pass,
   56 chunk iterations per frequency, 3 launches each). Each thread evaluates one pair and
   stores 48 Float32 values (3x1 S and K' blocks, 3x3 D and H blocks, re/im) to a buffer:
   192 B per pair.
2. Two gather kernels, one thread per operator entry, sum that buffer over the incident
   elements of the row node (and of the column node, for D and H) into the operators. No
   atomics and a fixed order, so it's bit-reproducible.

## Measured locally (seconds per frequency; the cost doesn't depend on frequency)
- Whole kernel ~1.60: pair stage 0.81, gather D/H 0.60, gather S/K' 0.22.
- That's ~0.8 G evaluations/s and ~35 GB/s of buffer traffic, far below peak on both, so
  the limit is something else: occupancy, register pressure, scattered gather reads, the
  write-then-read buffer, launch structure.
- Ruled out:
  - Threadgroup size 64/128/256 (`BLAB_METAL_KERNEL_GROUPSIZE`): no change. 512 fails,
    because the fused singular kernel is capped at 384 threads.
  - Gather budget: 128/256/512/2048 MB gives 1.69/1.58/1.61/1.89 s.
  - Other variants (`BLAB_METAL_REGULAR_KERNEL_MODE`):
    - `pair_atomic` (fused, float atomics scatter; the file header explains why atomics
      are slow here): 3.2 s
    - `pair_owned`: 5.5 s
    - `entry_owned`: 29 s
  - Pair tile shape is `BLAB_METAL_ATOMIC_TILE` (16x16 default); not tuned yet.

## Code
`fable/KERNEL_EXCERPT.jl` holds verbatim copies, each marked with its source file and lines
under `src/beat_engine/julia_local/src/`:
- the cache struct and launch helper
- the skip test for singular pairs
- the per-pair maths: `_metal_trial_term`, `_metal_regular_pair_blocks`
- the whole current gather path, with its header comment
- the mode dispatch and symmetry-image loop

## Deliverable: a plan, no code
**Don't write or change any code.** Another agent (Claude on the user's Mac, which has the
GPU) implements and benchmarks whatever you propose. Your output is one file,
`fable/PROPOSALS.md`, **at most ~250 lines**. Commit and push it, then stop.

1. **Diagnosis** (short): from the code and the timings, what most likely limits each
   stage: pair 0.81, gather D/H 0.60, gather S/K' 0.22. Name the evidence, and say which
   quick local measurement would confirm it, e.g. a stage timing, a counter, or a one-line
   kernel variant.
2. **Ideas, ranked, at most 6.** For each one, a few lines:
   - what changes in the kernel, and why it should help on Apple GPUs specifically
   - a bound on the gain, from the timings above
   - the accuracy risk
   - the effort
3. **Implementation plan for the top 2 ideas**, detailed enough to code without guessing:
   - kernel structure: grid and threadgroup shape, what each thread and threadgroup owns
   - threadgroup memory layout and size
   - the loop order
   - how the sums reach the operators with no atomics
   - how symmetry passes and skipped (singular or adjacent) pairs are handled
   - which existing functions to reuse
   - pitfalls
   Each one is added as a new `BLAB_METAL_REGULAR_KERNEL_MODE` value next to `pair_gather`,
   which stays untouched.
4. **Test plan:** the order to try things in, with small experiments first. Give a
   go/no-go threshold per idea. For each idea, also say what must be checked for
   correctness: relative L2 against `pair_gather` ≲ 1e-6.

Things worth judging:
- avoiding or shrinking the 48-value buffer (threadgroup tiles with in-group reduction,
  element orderings where a threadgroup owns whole node rows or columns)
- the shape of the per-pair maths (sincos/rsqrt cost, SIMD-group use, reusing loaded
  test-element data across trials, register pressure)
- sharing work across the 4 symmetry passes
- safe FP16 or mixed precision in the far field
- pair tile shape

Metal.jl facts for the plan:
- threadgroup memory is `MtlThreadGroupArray`
- barriers are `threadgroup_barrier`
- SIMD-group shuffles and reductions are available
- the kernels use Int32 indexing and must be type-stable and allocation-free

Follow `AGENTS.md` where it applies: no numerical baseline changes, and performance claims
are made only after local measurement.
