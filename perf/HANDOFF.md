# SAWMOD Metal speedups: handoff (2026-09-26, after round 9)

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
- Round 9 (in the app, not yet measured in-app; ideas from the CUDA backend): 50-freq sweep 0.910 →
  **0.823 s/freq**, maxrel 5.8e-6. **Estimated in-app 50-freq SAWMOD: ~41 s** (45 − 50 × 0.087 × 0.94).
  Checkpoint before it: tag `checkpoint-before-fused-images`.
- Code: this checkout, branch `perf/experiments`, pushed to the user's fork
  (`git push fork perf/experiments`, BumelantPZA/BEAT_Engine). **Never push to `origin`
  (JWSound).**
- Restore tags: `metal-test-58s` (round 3), `metal-test-bm-threaded` (round 5),
  `metal-test-round6`, `metal-test-round7`, `metal-test-round8`, `metal-test-round9` (current).
- App side: `../boundary-lab/src/blab/solvers/engine_distribution.py`,
  `METAL_TEST_SOLVER_OPTIONS["test_env"]`. It has no fork; its diff is kept in
  `perf/app_patches/engine_distribution.diff`. The app loads Julia at start, so restart it after
  engine edits.

App env now: `BLAB_TEST_COUPLED_PREFETCH=0`, `BLAB_METAL_REGULAR_KERNEL_MODE=pair_tilereduce`,
`BLAB_TEST_BLAS=accelerate`, `BLAB_METAL_FIELD_FAST=3`, `BLAB_TEST_DENSE_STATS=1`,
`BLAB_TEST_BM_THREADED=1`, `BLAB_TEST_GC_DEFER=1`, `BLAB_TEST_BLOCKED_LU=512`,
`BLAB_TEST_FAST_TRS=128`, `BLAB_TEST_STALE_LU=15`, `BLAB_TEST_OP_POOL=1`,
`BLAB_TEST_HOST_ROW_WEIGHTS=1`, `BLAB_TEST_FLUX_SKIP=1`, `BLAB_TEST_IMAGE_ACCUMULATE=1`,
`BLAB_TEST_COMBINED_BM=1`, `BLAB_TEST_MUMPS_EXPAND=1`, `BLAB_TEST_MASS_THREADS=4`.

## Per-frequency budget after round 9 (s, 50-freq harness medians, total 0.823 s/freq)
Each frequency runs a **first half** as two parallel branches, then a serial **second half**.

| Stage | s | Code |
|---|---|---|
| **First half = max(GPU branch, FEM stage) ≈ 0.33** | | `build_condensed_coupled_system`, `src/BeatEngineCoupledCondensed.jl` |
| GPU branch: BEM operators (combined A/C) 0.26 + host BM combine/products 0.04 | ~0.30 | `BeatEngineMetalAssembly.jl`, `BeatEngineMetalTileReduceKernels.jl` |
| FEM stage (task, own work): MUMPS LDLᵀ+Schur 0.247, transducer solves 0.034, mass solve 0.022, FEM system 0.011 | ~0.325 | `_build_mumps_condensation`, `BeatEngineMumps.jl` |
| Interface elimination: ComplexF64 GEMM B_q·W (3110×1602·1602×1602) 0.116 + glue | 0.15 | `_flux_block_products!` |
| Dense LU: blocked ComplexF32 getrf 0.11, or stale-LU GMRES below ~1 kHz | 0.09–0.11 | `RefinedDenseLU`, `_test_blocked_lu!`, `_test_stale_gmres` |
| Solve: dense refinement + MUMPS back substitution | ~0.06 | `solve_condensed_coupled_excitations` |
| Field (GPU) | 0.08 | `BeatEngineMetalFieldFast.jl` |
| Output, release, deferred GC, other | ~0.1 | `coupled_solver.jl` |

**The first half is balanced now:** the GPU branch is ~0.30 and the FEM stage ~0.325. On a Mac with a
faster CPU the GPU branch becomes critical again; on a Mac with a bigger GPU the FEM stage does.
Both sides are worth cutting, and the serial second half (~0.45) is now the largest block.

How to read the timers (these misled earlier rounds):
- `fem_condensation_s` is the **overlapped wall** of the first half, not the FEM work. The FEM
  task's own time is `interface_elim_diag_fem_stage_work_s`.
- `block_assembly_s` contains `interface_elimination_s`; pure assembly is ~0.01.
- `fem_schur_extraction_s` contains `fem_transducer_solves_s`.
- GPU split: set `BLAB_TEST_ASM_TIMING=<file>` + `BLAB_METAL_GATHER_TIMING=1`. The stages are
  `metal_native_gather_{pairs,slp_adjoint,dlp_hyp}`, `regular_kernel`, `singular_kernel`.
  Round 9: regular kernel ~0.24, singular 0.012.

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
- Micro-benchmarks: `julia --threads=10 perf/lu_micro.jl`, `perf/blas_micro.jl`,
  `perf/gemm3m_micro.jl`, `perf/precond_micro.jl <dump dir> <prev>:<next> ...`.
- For anything frequency-dependent (stale LU), test on the real 50-point sweep
  `20*1000**(i/49)`; the 12-freq set has ratio 1.87 between points.
- Before a risky change, push a checkpoint tag to the fork (the user asks for this).

## Rules the user set
- Short test cycles (quick.py), never two benchmarks at once, don't kill the user's app
  processes.
- Diagnose instead of restarting. After ~2 failed fixes of the same problem, report and ask.
- Bit-changing work goes behind a `BLAB_TEST_*` switch; accuracy must be maxrel ≤ ~1e-5 vs
  the app config.
- Commit and push to the fork at each step.
- Ask before installing or upgrading anything in the app checkout.
- Keep tokens low.
- Goal is speed on other Macs too (stronger CPU or GPU moves the bottleneck).

## What worked (all in the app)
| Round | Change | Gain |
|---|---|---|
| 1–2 | `pair_tilereduce` far-field kernel (Fable round 1), Accelerate BLAS | GPU ops 1.62 → 0.45 |
| 3 | Fast field eval (`FIELD_FAST=3`, maxrel 2.2e-5), fused dense-norm pass, prefetch off | 141 → 58 s in-app |
| 5 | Threaded Burton-Miller combine (`BM_THREADED`), exact | first half 0.54 → 0.47 |
| 6 | `GC_DEFER`: GC off during a frequency, one collection after (was 1.1 GB/freq, 5 pauses, 0.12–0.16 s), exact | −0.05 s/freq |
| 6 | (both round-6 rows together: in-app 58 → 51.2 s) | |
| 9 | `IMAGE_ACCUMULATE=1`: xy symmetry = identity + 3 images; chunk loop outside, transforms inside, later transforms add into the pair blocks, gathers once per chunk | −0.023 s/freq, maxrel 4e-7 |
| 9 | `COMBINED_BM=1` (CUDA's combined assembly): pair kernel writes A = −D + βH and C = −S − βK′ (2 reduction passes instead of 4, one gather each), singular gather combines too; host adds only row weights and identity. Needs HOST_ROW_WEIGHTS + singular write-back `gather`. `BLAB_TEST_COMBINED_CHECK=<file>` compares operators with the stock path (~1e-7) | BEM ops 0.344 → 0.258; −0.028 s/freq, maxrel 5.4e-6 |
| 9 | `MUMPS_EXPAND=1`: transducer interior solve as the expansion (ICNTL(26)=2, x_Γ=0) of the preceding reduction, one forward sweep saved | −0.011 s/freq, bit-identical |
| 9 | `MASS_THREADS=4`: CHOLMOD interface-mass solve in 4 column chunks on parallel tasks (common is task-local) | mass 0.043 → 0.022; −0.013 s/freq, maxrel 5e-12 |
| 9 | (round 9 together, 50-freq sweep: 0.910 → 0.823, maxrel 5.8e-6) | |
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
| Prefetch of next-frequency GPU ops (`COUPLED_PREFETCH=1`) | No gain: it contends with the CPU stages beside it |
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
| Fused image kernel (all 4 transforms per thread, CUDA `fused=true`) | Gathers 0.095 → 0.026 but pair kernel 0.27 → 0.45 (Metal compiler; also with one inlined body and running sums). Reverted; IMAGE_ACCUMULATE gets the gather saving without it |
| MUMPS on OpenBLAS32 (`BLAB_TEST_BLAS=hybrid`, MUMPS threads 4/8) | factorization 0.30/0.26 vs 0.247 on Accelerate |
| MUMPS orderings (`BLAB_TEST_MUMPS_ICNTL=7=…`) | METIS (auto) 0.246; SCOTCH/PORD/AMF/QAMD/AMD all 0.375 (likely not built in, fallback) |
| MUMPS BLR (`ICNTL(35)=2`, CNTL(7) 1e-12/1e-9) | 0.41–0.43 vs 0.247: fronts too small |
| MUMPS expansion in the solve-phase back substitution | Not done: transducer condensation changes f_I after the reduction (see `_mumps_back_substitution` docstring) |
| Thread counts (8 vs 10), kernel group size, gather budget | Within noise or worse (rounds 1–2) |

## Next ideas (untested)
First, **measure rounds 7–9 in the app**: the ~41 s is an estimate, 51.2 s is the last measured value.

Known leftovers, small:
1. GPU branch (~0.30):
   - Allocate only A and C when `COMBINED_BM` is on. K′ and H are zero-filled but unused (~0.005).
   - Project C onto the interface flux on the GPU (CUDA's `build_cuda_combined_bem_blocks`,
     one owner per output entry). This moves the host products (~0.02) off the CPU.
2. FEM stage (~0.325):
   - Write MUMPS values straight from precomputed K/M/wall index maps instead of building a new
     sparse `fem_system` per frequency (0.011).
   - MUMPS pivot threshold `CNTL(1)` (via `BLAB_TEST_MUMPS_CNTL`): check `INFOG(12)` (delayed
     pivots) first.
   - `MASS_THREADS` 8 was not better than 4 here; re-test on a Mac with more cores.
3. Stale LU (`perf/precond_micro.jl` measures GMRES iterations offline from
   `BLAB_TEST_DUMP_DENSE=<dir>` dumps; `BLAB_TEST_STALE_LOG=<file>` logs decisions):
   - A GMRES iteration costs ~8 ms: F64 3-column matvec 3.4 ms (~45 GB/s) + F32 solve 4 ms.
   - GMRES-IR (Carson–Higham) would run the inner GMRES with the F32 matrix copy (half the
     bytes) and keep the F64 residual only in the outer refinement. That would extend reuse
     toward ~2 kHz.
   - `STALE_STOP=13` would avoid the one failure at the cap (~0.12 s per sweep).
4. Cut the remaining GC: keep sweep-persistent buffers for the big per-frequency arrays (dense
   155 MB, F32 copy 78, bem_lhs/rhs 77+77, interface block 75). Measure `test_prev_gc_s` first.

Fresh directions for a new search (larger, unexplored):
- **Second half, now the largest block (~0.45 s serial).**
  - The elimination GEMM (0.116) can't be F32 (accumulation error 2.7e-4).
  - An **Ozaki-scheme GEMM** could work: split F64 into F32 or integer slices, multiply exactly
    on the GPU with MPS, then sum. That gives F64 accuracy at GPU F32 speed. It would also use the
    idle GPU in the second half. Same idea for the LU trailing updates.
- **The GPU is idle in the second half** (~0.45 s) and the CPU is idle during parts of the GPU
  kernel. Earlier overlap attempts lost to CPU contention (see table). An overlap using only
  GPU work in the second half (for example the next frequency's pair kernel, without CPU
  stages) might not contend the same way, but GPU assembly next to the Metal field once gave
  wrong results. Check buffer sharing first (`perf/attic/`).
- **Pair kernel (~0.24):** the far-field quadrature order per frequency
  (`quadrature_selections`). The kernel cost is ∝ R² quadrature points. Check whether low
  frequencies use more points than their accuracy needs, keeping maxrel ≤ 1e-5.
- **Other Macs:** thread counts are tuned for this 10-core machine (`MASS_THREADS=4`, Julia 10
  threads, blocked LU nb=512, FAST_TRS nb=128). A per-core-count default or a quick
  auto-tune at startup would help other hardware.
- **MUMPS factorization 0.247** is the FEM floor with current settings. METIS is best; BLR,
  other orderings and OpenBLAS are slower. The only CUDA-side answer (cuDSS on the GPU) has no
  Metal equivalent.

Not viable under 1e-5 (Fable's analysis): interior ROM / Craig-Bampton, rational interpolation
of S(k) over frequency, F32 MUMPS, F32 elimination.
