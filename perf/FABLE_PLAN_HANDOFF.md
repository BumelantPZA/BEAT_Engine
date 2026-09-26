# Handoff: plan the next SAWMOD speedups (planning only — don't run or change code)

**Task:** read this and the named code, then write `perf/FABLE_PLAN.md`: ranked ideas with
expected gain, risk and effort, and a test plan for the top 3. Don't run anything.

**Setup:** coupled FEM-BEM acoustic solve, BEAT Engine (Julia 1.12 + Metal.jl), Apple M1 Pro
(10 CPU cores, 16-core GPU, 16 GB). Model SAWMOD: FEM order 29,665, BEM 3,110 P1 nodes / 6,054
triangles (xy symmetry, 4 passes), 3 transducers, condensed dense system order 3,116.
Frequencies are solved one at a time. Now **1.14 s/freq** (in-app 50 freqs: 141 s → ~58 s so far).
Code: `src/beat_engine/julia_local/`. Per-frequency driver: `coupled_solver.jl`
(`solve_request`); the build is `build_condensed_coupled_system`
(`src/BeatEngineCoupledCondensed.jl:1650`).

## Per-frequency time (s, median of 12 freqs from 20 Hz to 20 kHz)
| Stage | s | Code | Notes |
|---|---|---|---|
| **First half**, two overlapped branches | **0.48** | Condensed.jl `build_…` | wall = the longer branch |
| a) GPU BEM operators (S, K', D, H) | 0.45 | `BeatEngineMetalAssembly.jl:235`, `BeatEngineMetalTileReduceKernels.jl` | "pair_tilereduce" kernel; already 3.6x optimized |
| a) Burton-Miller combine | 0.02 | Condensed.jl:2274 | threaded |
| b) FEM condensation (CPU task) | 0.46 | `_build_condensation` Condensed.jl:928, `BeatEngineMumps.jl` | MUMPS LDLᵀ + Schur 0.28, Schur extract 0.05, transducer solves 0.05, mass solve 0.05, FEM assembly 0.03 |
| Block assembly | 0.14 | Condensed.jl after gate (~1977) | builds the 3116² ComplexF64 system |
| Interface elimination | 0.13 | `_flux_block_products!` Condensed.jl:838 | a ComplexF64 GEMM is 0.12 of it |
| Dense LU | 0.21 | `RefinedDenseLU` Condensed.jl:204 | Float32 LU (Accelerate) + refinement vs ComplexF64 |
| Solve | 0.075 | `solve_condensed_coupled_excitations` :2425 | dense solve 0.05 + MUMPS back substitution |
| Field (far-field points) | 0.09 | `BeatEngineMetalFieldFast.jl` | GPU |
| Request/output overhead | ~0.12 | coupled_solver.jl, Python | unmeasured |

Critical path: max(GPU 0.45, condensation 0.46) + 0.14 + 0.13 + 0.21 + 0.075 + 0.09 + ~0.12.
The CPU is mostly idle (~2 of 10 cores busy on average).

## Already tried — don't propose again
- **Overlapping frequency i+1's first half with frequency i's second half:** slower (1.65 vs
  1.14) from core contention. MUMPS crashes if i's solve overlaps i+1's factorization
  (process-global Fortran state). GPU assembly running alongside the field evaluation gives
  wrong results. Details: `perf/NOTES.md` "Round 5".
- **Prefetching next-frequency GPU operators:** no gain (contends with condensation).
- **GPU LU via MPS:** real-only, the 2n embedding is 2x slower than the CPU.
- **ComplexF32 elimination GEMM** (`BLAB_TEST_ELIM_F32`): 1.11x but error 2.7e-4. The user hasn't
  decided yet.
- The Accelerate BLAS is already in use; fast field (0.32 → 0.09); threaded Burton-Miller combine.

## Constraints
- Accuracy: max relative error ≤ ~1e-5 vs the current output (the fast field is 2.2e-5).
- Anything bit-changing goes behind a test env switch (the `BLAB_TEST_*` pattern).
- Test harness: `perf/quick.py --revise` (A/B configs, hot reload). See `perf/NOTES.md`.

## Open questions worth a plan
Ideas we haven't tested:
- **Condensation (0.46):**
  - Is re-factorizing MUMPS every frequency necessary? E.g. a frequency-dependent low-rank
    update, or an iterative method seeded from the previous frequency.
  - Tuning MUMPS threads and OpenMP.
- **Dense chain (0.14 + 0.13 + 0.21 + 0.075):**
  - Avoid forming the full ComplexF64 matrix.
  - Factor in Float32 directly.
  - Fuse the block assembly with the elimination.
  - Use a Schur/block structure instead of one LU.
- **GPU (0.45):** only helps if the condensation also gets shorter; the two branches are nearly equal.
- **Overhead (~0.12):** what is in it?
