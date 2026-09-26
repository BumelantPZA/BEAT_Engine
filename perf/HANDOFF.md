# SAWMOD Metal speedups: handoff (2026-09-26, after round 10)

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
- Round 10 (in the app, not yet measured in-app): `ELIM_IMPLICIT` + `HOST_POOL`, 50-freq sweep
  0.807 → **0.653 s/freq**, maxrel 1.1e-7. **Estimated in-app 50-freq SAWMOD: ~34 s**
  (41 − 50 × 0.15 × 0.94). Tag `metal-test-round10`. **Measured in-app (rounds 7–10): 37.4 s.**
- Code: this checkout, branch `perf/experiments`, pushed to the user's fork
  (`git push fork perf/experiments`, BumelantPZA/BEAT_Engine). **Never push to `origin`
  (JWSound).**
- Restore tags: `metal-test-58s` (round 3), `metal-test-bm-threaded` (round 5),
  `metal-test-round6`, `metal-test-round7`, `metal-test-round8`, `metal-test-round9`, `metal-test-round10` (current).
- App side: `../boundary-lab/src/blab/solvers/engine_distribution.py`,
  `METAL_TEST_SOLVER_OPTIONS["test_env"]`. It has no fork; its diff is kept in
  `perf/app_patches/engine_distribution.diff`. The app loads Julia at start, so restart it after
  engine edits.

App env now: `BLAB_TEST_COUPLED_PREFETCH=0`, `BLAB_METAL_REGULAR_KERNEL_MODE=pair_tilereduce`,
`BLAB_TEST_BLAS=accelerate`, `BLAB_METAL_FIELD_FAST=3`, `BLAB_TEST_DENSE_STATS=1`,
`BLAB_TEST_BM_THREADED=1`, `BLAB_TEST_GC_DEFER=1`, `BLAB_TEST_BLOCKED_LU=512`,
`BLAB_TEST_FAST_TRS=128`, `BLAB_TEST_STALE_LU=15`, `BLAB_TEST_OP_POOL=1`,
`BLAB_TEST_HOST_ROW_WEIGHTS=1`, `BLAB_TEST_FLUX_SKIP=1`, `BLAB_TEST_IMAGE_ACCUMULATE=1`,
`BLAB_TEST_COMBINED_BM=1`, `BLAB_TEST_MUMPS_EXPAND=1`, `BLAB_TEST_MASS_THREADS=4`,
`BLAB_TEST_ELIM_IMPLICIT=1`, `BLAB_TEST_HOST_POOL=1`.

## Exterior-only (waveguide) regression, fixed 2026-09-26 (1477e34)
The user saw 200-freq waveguide-only prototype2: stock Apple Metal 12.4 s, Metal test 32.7 s.
`test_env` writes ENV process-wide and never restores it, and `metal_direct_assembly_available()`
accepted only `pair_gather`, so `pair_tilereduce` sent exterior sweeps to the four-operator path
without the sweep pipeline (harness 0.668 vs 0.265 s/freq). Fixed: tilereduce accepted (the fused
assembler ignores the mode), and exterior requests apply `test_env` too. Now 0.238 s/freq (FIELD_FAST
helps: field 0.10 -> 0.06), maxrel 3.5e-5 vs stock (FIELD_FAST). Test with
`"request": "proto2.json"` in a job (captured by `scripts/capture_boundary_lab_request.py`).
Lesson: any new ENV switch must be checked on an exterior-only project too.

## Per-frequency budget after round 10 (s, 50-freq sweep means, total 0.653 s/freq)
Each frequency runs a **first half** as two parallel branches, then a serial **second half**.

| Stage | s | Code |
|---|---|---|
| **First half = max(GPU branch, FEM stage) ≈ 0.30** | | `build_condensed_coupled_system`, `src/BeatEngineCoupledCondensed.jl` |
| GPU branch: BEM operators (combined A/C) 0.256 + host BM combine 0.007 + products | ~0.27 | `BeatEngineMetalAssembly.jl`, `BeatEngineMetalTileReduceKernels.jl` |
| FEM stage (task, own work): MUMPS LDLᵀ+Schur 0.238, transducer solves 0.029, mass solve 0.016, FEM system 0.011 | **0.293 (critical)** | `_build_mumps_condensation`, `BeatEngineMumps.jl` |
| Interface elimination + block assembly (implicit product: nothing left) | 0.017 | `_test_dense_mul!` |
| Dense factorization: fresh F32 LU 0.11 + narrowing/cgemm 0.03 (31/50 freqs); stale ~0 (19/50) | 0.103 mean | `RefinedDenseLU`, `_test_blocked_lu!`, `_test_narrowed` |
| Solve: refinement 0.033 fresh, stale GMRES ~0.1–0.15 (6–15 its); MUMPS back substitution ~0.02 | 0.097 mean | `solve_condensed_coupled_excitations`, `_test_stale_gmres` |
| Field (GPU; CPU idle) | 0.084 | `BeatEngineMetalFieldFast.jl` |
| Between iterations (deferred GC, emit) + request setup 0.42 s / 50 | ~0.04 | `coupled_solver.jl` |

**The FEM stage is critical now** (0.293 vs GPU 0.27), and MUMPS (0.238) is its floor with current
settings. The serial second half is ~0.30 (factorization and solve trade off via the stale LU).

How to read the timers (these misled earlier rounds):
- `fem_condensation_s` is the **overlapped wall** of the first half, not the FEM work. The FEM
  task's own time is `interface_elim_diag_fem_stage_work_s`.
- `block_assembly_s` contains `interface_elimination_s`; pure assembly is ~0.01.
- `fem_schur_extraction_s` contains `fem_transducer_solves_s`.
- GPU split: set `BLAB_TEST_ASM_TIMING=<file>` + `BLAB_METAL_GATHER_TIMING=1`. The stages are
  `metal_native_gather_{pairs,slp_adjoint,dlp_hyp}`, `regular_kernel`, `singular_kernel`.
  Round 9: regular kernel ~0.24, singular 0.012.
- Per-frequency rows: `"dump": true` now also writes `queue/<job>.<config>.r<n>.rows.json` (every
  timer per frequency plus the request wall). Keys are `d.timings.<name>`; `test_prev_*` values
  belong to the **previous** frequency (row 1 has none). `test_prev_{user,system}_s` and
  `test_prev_minflt` come from getrusage (page faults!).
- `BLAB_TEST_PHASE_LOG=<file>`: timestamps for request parse/setup/each iteration/emit.
  `BLAB_TEST_ALLOC_PROFILE=<file>`: every allocation ≥ 1 MB during frequency 3, by source line.
- Job names must be new: `job.sh` returns an old `queue/<name>.out` at once if the name was used
  (round 10 names start with `r10_`). `perf/mkjob.py 50 1 name=K=V,K=V` prints a job whose `base`
  is the app env read from `engine_distribution.py`.

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
| 10 | `ELIM_IMPLICIT=1`: the flux elimination's B_q·W (0.116 s zgemm) stays out of the F64 dense matrix; every F64 residual (refinement, stale GMRES) applies it as B·(W·x), row-chunked over tasks; the F32 LU input gets it from a cgemm (35 ms). Stale-LU frequencies skip the product; their norm is an upper bound (equal to the true norm on all fresh freqs) | 50-freq 0.807 → 0.745, maxrel 1.1e-7 |
| 10 | `HOST_POOL=1`: dense matrix, BM combine outputs, interface blocks, F32 LU input (and replaced stale factors), Schur conversion, mass-solve panel/W, MUMPS Schur copy come from a pool keyed by type and size and go back at release. A fresh 155 MB array costs ~57 ms of page faults, a reused one 1.5 ms. Alloc 1.13 → 0.27 GB/freq, faults 76k → 24k, system time 0.30 → 0.10 s/freq. Note: `mul!(C, A, B, -1, 0)` is **not** bit-identical to `-(A*B)`; plain `mul!` + negate is | 50-freq 0.734 → 0.653, bit-identical |
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
| Explicit GC after each frequency (`BLAB_TEST_GC_MODE=young/full`, round 10) | young 2.4 ms but sweep 0.745 → 0.767; full 75 ms. The cost was page faults, not the collection |

## Next ideas (untested)
Rounds 7–10 measured in-app: **37.4 s** (user, 2026-09-26; estimate was ~34 s, so the Revise-mode
scaling of 0.94 overstates in-app gains by ~10%; harness 0.653 s/freq × 50 = 32.7 s + ~4.7 s app overhead).

1. **FEM stage is critical (0.293).**
   - MUMPS factorization 0.238 is the floor (METIS, Accelerate; BLR/orderings/OpenBLAS slower).
     Check `INFOG(12)` (delayed pivots) and try `CNTL(1)` via `BLAB_TEST_MUMPS_CNTL`.
   - Write MUMPS values straight from precomputed K/M/wall index maps instead of building a sparse
     `fem_system` per frequency (0.011 s and 89 MB of fresh sparse arrays, the biggest unpooled
     allocation left; `assemble_fem_dynamic_stiffness`).
   - Start the FEM stage of frequency i+1 while the GPU evaluates frequency i's field (0.084 s, CPU
     idle). Only a CPU-only stage next to a GPU-only stage; MUMPS i's back substitution must be
     finished (it is, before the field). Earlier overlaps lost to contention, but those put CPU
     stages next to CPU stages. Potential ~0.05 s/freq.
2. **Second half (~0.30).**
   - Stale GMRES costs ~12 ms per iteration now (F64 matvec + correction + F32 solve). GMRES-IR
     (inner GMRES on the F32 matrix copy) would halve the matvec bytes and could extend reuse.
   - F32 LU (0.11 s on 31/50 freqs) on the idle GPU: MPS is real-only (2x embedding lost before), but a
     hybrid (CPU panel, GPU trailing cgemm as 4 real sgemms) was never tried. M1 Pro GPU ~4.5
     TFLOP/s F32 vs CPU cgemm ~2; bigger GPUs gain more.
3. **Remaining allocations** (0.27 GB/freq, 24k faults): CHOLMOD chunk outputs (56 MB), FEM sparse
   assembly (89 MB), the first narrowed LU input per stale cycle. `BLAB_TEST_ALLOC_PROFILE` lists them.
4. **Other Macs:** thread counts are tuned for this 10-core M1 Pro (`MASS_THREADS=4`, Julia 10
   threads, blocked LU nb=512, FAST_TRS nb=128). BLAS runs on the shared AMX units (user CPU is only
   ~0.8 s per 0.61 s iteration), so core count matters less than AMX count (per cluster).

Not viable under 1e-5 (Fable's analysis): interior ROM / Craig-Bampton, rational interpolation
of S(k) over frequency, F32 MUMPS, F32 elimination.
