# Handoff: speed up the Metal BEM far-field assembly kernel

**Goal:** get architecture ideas for making the core GPU calculation faster. It is now the
bottleneck of a coupled FEM-BEM acoustic solve (BEAT Engine, Julia + Metal.jl, Apple M1 Pro,
16-core GPU, 16 GB). Out of scope: batching frequencies together, or moving work between
CPU and GPU. The question is only how to compute this kernel faster.

## What it computes (per frequency)
- Four dense Galerkin boundary operators for the Helmholtz kernel G = e^{ikr}/(4πr), used by
  Burton-Miller: single layer S (P1 x DP0), adjoint double layer K' (P1 x DP0), double layer
  D (P1 x P1) and hypersingular H (P1 x P1, with surface curls). ComplexF32.
- Mesh: 6,054 triangles, 3,110 P1 nodes. xy symmetry means 3 mirror images, so 4 full passes
  over all element pairs: ~147 M element pairs per frequency. Order-2 rule, 3 points per
  triangle, so 9 kernel evaluations per pair (~1.3 G evaluations). Singular/adjacent pairs
  are skipped here (a separate kernel, <0.1 s).

## Current kernel ("pair_gather", the fastest of 4 existing variants)
1. Pair kernel, 2-D grid (test element x trial element, trial side in chunks sized by a
   512 MB budget): each thread evaluates one element pair and stores 48 Float32 values
   (3x1 S and K' blocks, 3x3 D and H blocks, re/im) to a buffer, i.e. 192 B per pair.
2. Two gather kernels, one thread per operator entry: they sum the buffer over the incident
   elements of the row node and column node into S/K' and D/H. No atomics and a fixed
   summation order, so it's bit-reproducible.

## Measurements (s per frequency; the cost doesn't depend on frequency)
- Whole kernel ~1.60: pair stage 0.81, gather D/H 0.60, gather S/K' 0.22.
- That's ~0.8 G kernel evaluations/s and ~35 GB/s of buffer traffic. Both are far below the
  M1 Pro's peaks (~5 TFLOPS FP32, ~200 GB/s), so the kernel is limited by something else:
  occupancy, register pressure, scattered gather reads, or launch structure.
- Ruled out: threadgroup size 64/128/256 (no change; 512 exceeds the 384 limit of the
  singular kernel). Gather budget 128-2048 MB (256-512 best, bigger is worse). The other
  variants are slower: fused pair+atomic scatter 3.2 s (float atomics are ~10x the cost of
  evaluating G on Apple GPUs), pair_owned 5.5 s, entry_owned 29 s.

## Code (repo: BEAT Engine v0.2.0, `src/beat_engine/julia_local/src/`)
- `BeatEngineMetalGatherKernels.jl`: pair kernel, both gather kernels, launch loop (current path).
- `BeatEngineMetalAtomicKernels.jl:89`: `_metal_regular_pair_blocks`, the per-pair maths.
- `BeatEngineMetalAssembly.jl`: dispatch plus the symmetry-image loop.
- Stage timing: `BLAB_METAL_GATHER_TIMING=1`.

## Questions
- How can the 48-value intermediate buffer be avoided or reduced? For example, tiles held in
  threadgroup memory with an in-group reduction, or ordering elements so that each
  threadgroup owns whole node rows or columns.
- Is the per-pair maths the right shape for Apple GPUs: sincos/rsqrt cost, SIMD-group use,
  reusing loaded test-element data across trials?
- Can the 4 symmetry passes share work? The images only flip signs of trial coordinates.
- Are there FP16 or mixed-precision parts that are safe for the far field?
