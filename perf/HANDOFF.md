# SAWMOD Metal speedups: handoff (updated 2026-09-27, round 17)

To continue in a new session, open `~/Desktop/Claude/Boundarylab/beat-engine-test` and say:
"Read perf/HANDOFF.md (Status + Rules) and perf/PLAN_ROUND17.md section S<n>; do S<n>." Full history is in `perf/NOTES.md`
(newest round on top).

## Accuracy policy (agreed with the user 2026-09-27; replaces the old "maxrel <= 1e-5", which was
## never the user's rule, only a default Claude set on 2026-09-26)
Explain accuracy to the user in plain terms (dB change and seconds saved), not maxrel.
- Scale: 0.001 dB arithmetic noise; **0.01 dB = invisible, upstream's "same answer" line**; 0.1 dB
  about a plot's line width; 1 dB visibly wrong. Mesh coarseness alone moves results 0.02-0.5 dB.
- Rule: changes that keep results identical or move them by ~0.001 dB may be switched on; anything
  that moves results by >= 0.1 dB is rejected whatever it saves; in between, ask the user with the
  seconds saved and the dB cost.
- Upstream's own gate (docs/Metal Backend.md, scripts/compare_coupled_precision.jl): vs a CPU Float64
  run, "5e-4 relative / 0.01 dB", dB = |20 log10(|new|/|ref|)| at points within 80 dB of the peak.
- Measuring: `quick.py` prints `maxdB` (max level difference within 60 dB of each excitation's peak,
  vs the job's first config, `OVER` if > 0.01) and, with `"detail": true`, dB and maxrel per
  frequency plus the worst outputs. The first config is the reference: `{}` = stock settings;
  `BLAB_TEST_FIELD_F64=1` = the same solve with a Float64 CPU field (true reference for the field).
- Know the floor: against a Float64 field, every Float32 field kernel, stock included, is 0.02-0.03 dB
  off at 10-20 kHz at the quietest points of the 60 dB window (Float32 summation). A max-dB limit of
  0.01 there is below what Float32 can deliver; compare against stock, not against zero.

## Status round 17 S1 (2026-09-27, NOTES "Round 17 S1"; plan perf/PLAN_ROUND17.md)
- M1 SAWMOD: **CPU chain leads by ~36 ms** (GPU +50 ms costs +5, CPU +50 ms costs +41; MUMPS ~-80 ms buys -74,
  GPU -150 ms buys -12). Next for SAWMOD: R17-3 micro, then R17-6.
- M2 prototype2 quarter: `BLAB_METAL_PIPELINE=1` 0.055 -> 0.047 s/freq (identical results); the overlap model
  turns it off (solve model 1.9 ms vs ~8 ms real). R17-2 = switch/model fix; verify on proto2 full + 200 freqs.
- M3 cold start not run yet (needs quick.py stopped; blocked in S1). Run it first in the next session.
- New hooks: `BLAB_TEST_DELAY_EXT_SOLVE`, `BLAB_TEST_COLD_LOG` (process env), overlap_plan line in PHASE_LOG,
  `mkjob.py --request=`.

## Status after round 16 (2026-09-27): GPU_PLAN T1-T5 worked through (NOTES "Round 16")
- Kept (test-only switches, off by default, app patch `perf/app_patches/round16_engine_distribution.diff`
  not yet applied): `BLAB_TEST_FUSED_IMAGE_ACC=1` + `BLAB_TEST_FUSED_PACKED=2` (T1, exterior),
  `BLAB_TEST_FIELD_MULTI=1` (T3 part 1, coupled). App path: SAWMOD 32.5 -> 28.0 s, prototype2 quarter
  14.3 -> 12.3 s (busy machine), 0.0005 / 0.002 dB.
- No gain or rejected: T2 (early combine slower in the tile-reduce kernel, runtime-loop variant crashes
  the Metal compiler; TY=8 neutral), T3 part 2 (centroid far field: 20-43 dB), T4 (3-point far pairs:
  0.24-50 dB for <= 4 %), T5 (singular split: accurate only for k h < 0.8, ~2-3 ms/freq; optional
  `BLAB_TEST_SING_SPLIT=0.6`, 0.0004 dB). T1 steps 3-4 skipped / no gain.
- Metal compiler: a kernel with two runtime-loop quadrature bodies (or a runtime test loop beside the
  tile-reduce barriers) fails at pipeline link; use one body per launch.

## Standing facts from rounds 10-15 (full text: `perf/HANDOFF_ARCHIVE.md`)
- SAWMOD per-frequency structure: pipeline `COUPLED_PREFETCH + EARLY_BUILD + PREFETCH_OPT + FEM_LANE=2`.
  CPU chain = FEM stage (MUMPS) -> elimination -> F32 LU (fresh freqs) ; solve(i) and field(i) run beside
  FEM(i+1). GPU lane = BEM operators + field. Any CPU-stage overlap beside MUMPS loses to contention;
  measure every new overlap in the pipelined 50-freq sweep (standalone micro gains vanished inside it).
- R15b: lanes tied (CPU ahead ~20 ms). State bottlenecks only from a causal test (delay/speed-up one lane).
- prototype2 (exterior) is GPU-bound 1:1. Exterior projects: check every new ENV switch there too
  (`test_env` persists in the worker ENV; the 1477e34 regression).
- MUMPS is exhausted in double precision (R13b). Cold start ~40-60 s: `coupled_solver.jl` includes the
  engine from source; only the exterior driver uses the precompiled `BeatEngineMetalBundle`.
- Code: branch `perf/experiments`, push to `fork` (BumelantPZA). **Never push to `origin` (JWSound).**
  Tags `metal-test-*` are restore points. App side: `../boundary-lab/src/blab/solvers/engine_distribution.py`
  (`METAL_TEST_SOLVER_OPTIONS`), `coupled_backend.py`; diffs in `perf/app_patches/` (the user applies them).
- App path tool (the app's own headless solve, persistent worker, run 2 = warm):
  ```
  cd ../boundary-lab
  .venv/bin/python ../beat-engine-test/perf/app_timing.py examples/Multi_region_SAWMOD/Multi_region_SAWMOD.blab.json 2 "" 50
  APP_TIMING_BACKEND=beat_metal .venv/bin/python ../beat-engine-test/perf/app_timing.py ../beat-engine-test/perf/proto2_quarter.blab.json 2 ""
  ```
  Args: project, runs, threads ("" = backend default), freq count. Env: `APP_TIMING_BACKEND`
  (beat_metal_test | beat_metal = stock), `APP_TIMING_SYMMETRY`, `APP_TIMING_EXTRA`, `APP_TIMING_DUMP=<pkl>`.
- Harness requests: `"request": "proto2.json" | "proto2q.json" | "vented_sub.json" | "compression_driver.json" | "sawmod.json"`.
- Tools: `BLAB_TEST_FUSED_TIMING=<file>` (+ `BLAB_METAL_GATHER_TIMING=1`), `BLAB_TEST_ASM_TIMING=<file>`.

## How to read the timers (these misled earlier rounds)
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
- Bit-changing work goes behind a `BLAB_TEST_*` switch and is judged by the accuracy policy below.
- Commit and push to the fork at each step.
- Ask before installing or upgrading anything in the app checkout.
- Keep tokens low. How (from the round 16 usage review, 2026-09-27: ~56 % of that session's tokens went to
  re-reading old command output, ~26 % to fixed instructions, ~18 % to writing code and thinking):
  - Trim output at the source: on a failed job print only the error type and the first `BeatEngine*.jl:<line>`
    locations (`grep -o "Reason: [^\\]*"`, `grep -o "BeatEngine[A-Za-z]*.jl:[0-9]*" | head -4`), never whole
    stack traces; no `"detail": true` unless the per-frequency dB is actually needed; `| tail`/`grep` on job output.
  - Delegate benchmark runs whose raw output isn't needed to a cheap runner agent (`horn-runner`, Haiku) with an
    exact command and a short table as the answer. Keep edit-compile-fix loops and decisions in the main session
    (an agent starts cold and re-reads the code, which costs more than it saves there).
  - Split long work into sessions at phase boundaries (e.g. one per 1-2 targets), with the state in NOTES/HANDOFF,
    so later turns don't re-read a long history.
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
| Round 11: `FEM_LANE=1` (build(i+1) at build(i)'s FEM end) | LU(i) doubles next to MUMPS(i+1) (AMX): 0.569 vs 0.549 |
| Round 11: MUMPS per FEM component / concurrent instances | One component is 227 of 240 ms; two instances concurrently crash (MUMPS_LOAD_INIT) |
| Round 11: look-ahead LU (panel k+1 factored during trailing gemm) | −15% alone (bit-identical), nothing inside the pipeline |
| Round 11: stale LU reuse 12/18 or 5/6 (was 8) | 0.490/0.498 and 0.484/0.480 vs 0.487/0.480: GMRES traffic slows MUMPS beside it |
| Round 11: sleeping Metal wait (Metal.jl spins/yields in `synchronize`) | Spin is ~7% of busy samples; no change |
| Round 11: user-interactive QoS on all Julia threads | No change |
| Explicit GC after each frequency (`BLAB_TEST_GC_MODE=young/full`, round 10) | young 2.4 ms but sweep 0.745 → 0.767; full 75 ms. The cost was page faults, not the collection |

## Next ideas (untested)
1. Transducer reduce in the FEM stage (0.016 s, 6 dense columns; the expand now overlaps the mass
   presolve): sparse RHS (ICNTL(20)).
2. Interface-mass presolve in the FEM stage (0.018 s).
3. One-time costs (~1.1 s per request): overlap request setup / first-frequency caches.
4. Fresh F32 LU (0.11 s, critical on 29/50 freqs) on the GPU: busy only 0.33 of the 0.47 s cycle, but
   GPU work beside the Metal field gave wrong results (round 5); would need its own queue and gating.
5. Anything that cuts memory traffic beside MUMPS (stale GMRES reads ~0.35 GB per iteration).
6. Other Macs: the pipeline adapts (threads = core count); MUMPS single-thread speed sets the floor.

Not viable at upstream's accuracy (Fable's analysis): interior ROM / Craig-Bampton, rational interpolation
of S(k) over frequency, F32 MUMPS, F32 elimination.
