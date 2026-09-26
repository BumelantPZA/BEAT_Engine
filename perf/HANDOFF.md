# SAWMOD Metal speedups: handoff (2026-09-26, after round 8)

To continue in a new session, open `~/Desktop/Claude/Boundarylab/beat-engine-test` and say:
"Read perf/HANDOFF.md and continue with the next idea." Full history is in `perf/NOTES.md`
(newest round on top).

## Where things stand
- Goal: shorten the whole SAWMOD coupled FEM-BEM solve (50 freqs) in Boundary Lab's
  "BEAT Engine (Apple Metal test)" solver. In-app, 50 freqs: 141 s (stock Apple Metal) → 58 s
  (round 3) → **51.2 s (round 6, user-measured 2026-09-26)**.
- Harness (12 freqs, Revise mode): 1.224 → 1.084 s/freq with round 6, maxrel 1.3e-7.
- Round 7 (in the app, not yet measured in-app): 50-freq harness sweep 0.966 → **0.935 s/freq**
  from the stale LU on top of FAST_TRS (itself −0.028 s/freq on the 12-freq set).
  **Estimated in-app 50-freq SAWMOD: ~48 s** (51.2 − 50 × 0.059 s/freq × ~0.94 for Revise mode
  ≈ 48.4 s; FAST_TRS ≈ −1.3 s, STALE_LU ≈ −1.5 s).
- Round 8 (in the app, not yet measured in-app, all bit-identical): 50-freq sweep 0.972 → **0.899
  s/freq**. **Estimated in-app 50-freq SAWMOD: ~45 s** (48.4 − 50 × 0.073 × 0.94).
- Code: this checkout, branch `perf/experiments`, pushed to the user's fork
  (`git push fork perf/experiments`, BumelantPZA/BEAT_Engine). **Never push to `origin`
  (JWSound).**
- Restore tags: `metal-test-58s` (round 3), `metal-test-bm-threaded` (round 5),
  `metal-test-round6`, `metal-test-round7`, `metal-test-round8` (current).
- App side: `../boundary-lab/src/blab/solvers/engine_distribution.py`,
  `METAL_TEST_SOLVER_OPTIONS["test_env"]`. It has no fork; its diff is kept in
  `perf/app_patches/engine_distribution.diff`. The app loads Julia at start, so restart it after
  engine edits.

App env now: `BLAB_TEST_COUPLED_PREFETCH=0`, `BLAB_METAL_REGULAR_KERNEL_MODE=pair_tilereduce`,
`BLAB_TEST_BLAS=accelerate`, `BLAB_METAL_FIELD_FAST=3`, `BLAB_TEST_DENSE_STATS=1`,
`BLAB_TEST_BM_THREADED=1`, `BLAB_TEST_GC_DEFER=1`, `BLAB_TEST_BLOCKED_LU=512`,
`BLAB_TEST_FAST_TRS=128`, `BLAB_TEST_STALE_LU=15`, `BLAB_TEST_OP_POOL=1`,
`BLAB_TEST_HOST_ROW_WEIGHTS=1`, `BLAB_TEST_FLUX_SKIP=1`.

## Per-frequency budget now (s, harness medians)
| Stage | s | Code |
|---|---|---|
| First half: GPU BEM ops 0.42 + BM combine 0.05 **tied with** FEM condensation 0.47 | 0.47 | `build_condensed_coupled_system`, `src/BeatEngineCoupledCondensed.jl` |
| … condensation parts: FEM system 0.03, MUMPS LDLᵀ+Schur 0.28, transducer solves 0.05, mass solve 0.05 | | `_build_mumps_condensation`, `BeatEngineMumps.jl` |
| Interface elimination (ComplexF64 GEMM 0.12) | 0.15 | `_flux_block_products!` |
| Dense LU (blocked ComplexF32, getrf 0.12) + checks | 0.15 | `RefinedDenseLU`, `_test_blocked_lu!` |
| Solve (dense refinement 0.058 + MUMPS back substitution) | 0.08 | `solve_condensed_coupled_excitations` |
| Field (GPU) | 0.09 | `BeatEngineMetalFieldFast.jl` |
| Per-request setup (frequency 1 only, spread over the sweep) | ~0.1 at 12 freqs | `coupled_solver.jl` |

Two timers are nested, which misled the round-2 Fable plan:
- `block_assembly_s` **contains** `interface_elimination_s`; pure assembly is ~0.01.
- `fem_schur_extraction_s` **contains** `fem_transducer_solves_s`.

**Correction (round 8):** the first half was not a tie. `interface_elim_diag_fem_stage_work_s`
(the FEM task's own work) is 0.35, while `fem_condensation_s` is the overlapped wall (= the GPU
branch). The GPU branch (BEM ops + BM combine) was critical by ~0.1; after round 8 it is
0.36 + 0.04 vs FEM 0.35, so the next GPU cut beyond ~0.05 gains nothing without FEM cuts.
GPU assembly split (`BLAB_TEST_ASM_TIMING=<file>` + `BLAB_METAL_GATHER_TIMING=1`): pair kernel
0.27, DLP/HYP gather 0.066, S/K' gather 0.037 (halved by FLUX_SKIP), singular 0.012.

## How to test
```
cd perf && PYTHONPATH=$PWD/../src nohup ../../boundary-lab/.venv/bin/python $PWD/quick.py --revise > quick.log 2>&1 &
./job.sh <name> '{"freqs":[20.0,37.5,70.2,131.6,246.6,462.0,865.8,1622.3,3039.8,5696.1,10673.4,20000.0],"rounds":2,"dump":true,"configs":{"base":{...app env...},"new":{...app env..., "BLAB_TEST_X":"1"}}}'
```
- The worker takes ~65 s to start once. Revise then hot-reloads Julia edits before each solve.
- Don't edit sources while a job runs.
- After a `struct` or changed `const` edit, touch `queue/RESTART`.
- Output is `queue/<name>.out`: s/freq, sections and maxrel vs the first config. `"dump": true`
  writes `queue/<name>.<config>.r<n>.timings.json` with every timer, including
  `interface_elim_lu_*_s` and `test_prev_{gc_s,gc_pauses,alloc_gb,iteration_wall_s}`.
- Revise mode is ~6% slower than plain; compare configs within one job (the GPU varies ~15%
  when the machine is shared).
- Micro-benchmarks: `julia --threads=10 perf/lu_micro.jl`, `perf/blas_micro.jl`.

## Rules the user set
- Short test cycles (quick.py), never two benchmarks at once, don't kill the user's app
  processes.
- Diagnose instead of restarting. After ~2 failed fixes of the same problem, report and ask.
- Bit-changing work goes behind a `BLAB_TEST_*` switch; accuracy must be maxrel ≤ ~1e-5 vs
  the app config.
- Commit and push to the fork at each step.
- Ask before installing or upgrading anything in the app checkout.
- Keep tokens low.

## What worked (all in the app)
| Round | Change | Gain |
|---|---|---|
| 1–2 | `pair_tilereduce` far-field kernel (Fable round 1), Accelerate BLAS | GPU ops 1.62 → 0.45 |
| 3 | Fast field eval (`FIELD_FAST=3`, maxrel 2.2e-5), fused dense-norm pass, prefetch off | 141 → 58 s in-app |
| 5 | Threaded Burton-Miller combine (`BM_THREADED`), exact | first half 0.54 → 0.47 |
| 6 | `GC_DEFER`: GC off during a frequency, one collection after (was 1.1 GB/freq, 5 pauses, 0.12–0.16 s), exact | −0.05 s/freq |
| 6 | (both round-6 rows together: in-app 58 → 51.2 s) | |
| 8 | `OP_POOL=1`: keep the four Metal operator buffers for the next frequency, zero them on the GPU (alloc 0.02) | −0.036 s/freq |
| 8 | `HOST_ROW_WEIGHTS=1`: symmetry row weights applied inside the threaded host BM combine instead of four GPU broadcasts + weight upload | −0.024 s/freq |
| 8 | `FLUX_SKIP=1`: 2964 of 6054 faces carry flux; skip the S/K' reduction pass (pair kernel) and S/K' gather for the others, host combine writes zeros | −0.031 s/freq |
| 8 | (round 8 together, 50-freq sweep, bit-identical: 0.972 → 0.899) | |
| 7 | `FAST_TRS=128`: the F32 LU solve as row swaps + blocked trsm with gemm updates (Accelerate getrs barely threads: 15 → 4 ms for 3 RHS), maxrel 6.6e-8 | −0.028 s/freq |
| 7 | `STALE_LU=15`: previous fresh F32 LU as GMRES preconditioner (3 RHS in lockstep, same Float64 backward-error test), reused while the last stale solve took ≤ 8 iterations (`STALE_REUSE`), fresh LU + no more reuse after a failure at the cap. Used on 17/50 freqs (23 Hz–0.9 kHz, 6–15 iterations, ~8 ms each). maxrel 1.8e-8 | −0.031 s/freq (50-freq sweep) |
| 6 | `BLOCKED_LU=512`: right-looking blocked LU, Accelerate getrf panel + cgemm trailing update (cgetrf runs at 0.3 TFLOP/s, cgemm at 2.0), maxrel 1.3e-7, refinement 1.5 → 2 steps | −0.085 s/freq |

## What didn't work — don't retry without a new reason
| Idea | Result |
|---|---|
| Pipeline: i+1's first half overlapping i's second half (round 4/5) | MUMPS crash when i's solve overlaps i+1's factorization (process-global Fortran state). GPU assembly next to the Metal field gives wrong results (maxrel 0.2). Even the correct variant is 1.65 vs 1.15 s/freq (contention). Code in `perf/attic/` |
| `EARLY_BUILD=1` (+`PREFETCH=1`): build(i+1) as a task after i's solve, GPU(i+2) gated behind i's field | Exact, no crash, but 1.13 vs 1.085: only the field leaves the path; condensation, elimination and LU run ~15% slower next to the GPU assembly. **Lesson: the machine is saturated during BLAS/MUMPS stages even when few cores look busy; any CPU-stage overlap loses.** |
| Prefetch of next-frequency GPU ops (`COUPLED_PREFETCH=1`) | No gain: the first half is tied with the condensation, and it contends |
| GPU LU via MPS | Real-only; the 2n embedding is 2x slower than the CPU |
| ComplexF32 elimination (`ELIM_F32`) and split hi/lo (`ELIM_SPLIT`) | maxrel 2.7e-4 both: the error is F32 **accumulation** (cancellation), so no F32 variant of that GEMM can meet 1e-5 |
| Fable I1: never materialize the F64 dense matrix | Pure assembly 0.01 and LU glue (isfinite/stats/convert) 0.024: nothing to gain |
| Fable I3: output-stage trimming | Interface errors, quantities, JSON and release total ~1 ms |
| Fable I5: MUMPS Schur triangle check | Short-circuits on the first entry; free already |
| Fable I7b: LAPACK binding | Already `$NEWLAPACK$ILP64` |
| Two-level blocked LU (inner blocked panel) | ~130 ms, same as single-level nb=512 (134 ms); not worth it |
| Stale LU: `STALE_REUSE=12` + `STALE_STOP=12` | 0.942 vs 0.932: the aging factor hits 13 iterations by 60 Hz and reuse stops. Frequent refresh (REUSE=8) is better |
| Stale LU as plain refinement (no GMRES) | Never converges, even at 40 Hz (‖ΔA‖/‖A‖ ≈ 0.15 per step) |
| `BLAB_MUMPS_THREADS` 6/8, `BLAB_MUMPS_SOLVE_THREADS=2` (round 8) | 1.003–1.006, no effect, bit-identical |
| 3M complex GEMM for the elimination product (3 dgemms) | 158 → 143 ms micro; split/combine overhead eats it; one dgemm already runs at ~0.45 TFLOP/s F64 (`perf/gemm3m_micro.jl`) |
| Prefetch again after round 8 (`COUPLED_PREFETCH=1`) | 0.977 vs 0.899: the GPU assembly slows the CPU stages beside it (elimination 0.15 → 0.19) |
| Skip S/K' math inside the quadrature fold for rigid trials | Not tried: S is needed for the hypersingular term; saves ~6 of ~40 ops per qpair, and SIMD groups mix two trial rows |
| Thread counts (8 vs 10), MUMPS 2/6 threads, kernel group size, gather budget | Within noise or worse (rounds 1–2, when GPU-bound; MUMPS threads untested since the tie) |

## Next ideas (untested, rough order)
1. Stale-LU follow-ups (done in round 7; `perf/precond_micro.jl` measures GMRES iterations
   offline from `BLAB_TEST_DUMP_DENSE=<dir>` dumps; `BLAB_TEST_STALE_LOG=<file>` logs decisions):
   - The GMRES iteration costs ~8 ms: F64 3-column matvec 3.4 ms (~45 GB/s, below the memory
     bandwidth) + blocked F32 solve 4 ms. Halving it extends reuse to ~2 kHz.
   - `STALE_STOP=13` would avoid the one failure at the cap (~0.12 s per sweep); marginal.
2. **First half: now GPU 0.40 vs FEM 0.35 (see correction above).** Remaining items:
   - GPU side (Fable I4): assemble S/K′ only for the DP0 columns that carry flux (interface +
     transducer + prescribed faces); rigid faces multiply zero flux. Expected 0.05–0.15 off the
     GPU branch. Where: `BeatEngineMetalAssembly.jl:235`, tile-reduce kernel column range.
   - Condensation side:
     - `BLAB_MUMPS_THREADS` 6/8 (untested since the tie).
     - `BLAB_MUMPS_SOLVE_THREADS=2` for the transducer solves (0.05).
     - Write MUMPS values straight from precomputed K/M/wall index maps instead of building a
       new sparse `fem_system` per frequency (0.03).
     - MUMPS pivot threshold `CNTL(1)`: check `INFOG(12)` (delayed pivots) first.
3. **LU panel before the Schur arrives** (Fable I9): factor the ~1,560 BEM-only columns as
   soon as the BEM block exists, and finish once S lands. This only helps after idea 2 makes
   the GPU branch finish first.
4. **Cut the remaining GC entirely:** keep sweep-persistent buffers for the big per-frequency
   arrays (dense matrix 155 MB, F32 copy 78, bem_lhs/rhs 77+77, interface block 75, …) so the
   deferred collection has nothing to do. Worth ≤0.05; measure `test_prev_gc_s` after
   re-enabling first.
5. Faster blocked LU: nb tuning with the real matrix, or a recursive LU. The micro-benchmark
   floor is ~cgemm time (120 ms for n³), so at most ~0.03 more.

Not viable under 1e-5 (Fable's analysis): interior ROM / Craig-Bampton, rational interpolation
of S(k) over frequency, F32 MUMPS, F32 elimination.
