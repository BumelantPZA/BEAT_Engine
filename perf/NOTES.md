# SAWMOD Metal performance experiments (2026-09-25)

## Round 11 (2026-09-26): in-app BLAS bug, MUMPS pivoting, early build with optimized prefetch
In-app path (`perf/app_timing.py`, the app's headless solve on the SAWMOD project, 50 freqs, warm,
8 threads): **37.7 s → 33.9 s (BLAS fix) → 29.4 s (round 11)**. The user measured 37.4 s in-app.
| Change | Harness (50-freq sweep) | maxrel |
|---|---|---|
| Fix: MUMPS loaded before the Accelerate forward (`test_apply_blas!`). OpenBLAS32's lazy JLL library forwards itself into libblastrampoline on first dlopen; the app applied `BLAB_TEST_BLAS` first, so MUMPS ran on OpenBLAS32 in-app (FEM stage 0.35 vs 0.29, 2.5 s user CPU/freq from spinning OpenBLAS threads). The harness warmup loaded MUMPS first and never saw it | in-app 0.725 → 0.649 steady | 0 |
| `MUMPS_CNTL=1=0`: no numerical pivoting (never a delayed pivot; the search cost 0.04 of 0.232 s). Failed factorization retries with CNTL(1)=0.01 | 0.642 → 0.623 | 4.4e-8 |
| `FEM_F32_SKIP=1`: no Float32 FEM system built only to be replaced by the Float64 one | 0.624 → 0.618 | 0 |
| `PREFETCH_OPT=1` + `EARLY_BUILD=1` + `COUPLED_PREFETCH=1`: prefetched GPU operators now use the round 8/9 switches (shared `_test_metal_operators`); build(i+1) overlaps field(i) | 0.625 → 0.571 | 0 |
Didn't help: a sleeping Metal wait (Metal.jl `synchronize` spins/yields; only ~7% of busy samples),
EARLY_BUILD with the old prefetch path (0.584–0.611, noisy). MUMPS stats (`BLAB_TEST_MUMPS_STATS`):
n=24947, Schur 1602, 4.2 GFLOP in 0.19 s (18 GFLOP/s: not BLAS-bound), no delayed pivots.
One-time per request: setup 0.55 s + first frequency +0.3 s (analysis, FEM matrices, caches).

## Round 10 (2026-09-26): implicit elimination product, host array pool
| Change | Harness (50-freq sweep) | maxrel |
|---|---|---|
| `ELIM_IMPLICIT=1`: B_q·W kept out of the F64 dense matrix, applied in every F64 residual; F32 LU input gets it from a cgemm | 0.807 → 0.745 | 1.1e-7 |
| + `HOST_POOL=1`: big per-frequency host arrays reused (dense, BM outputs, interface blocks, LU input, mass solve, MUMPS Schur copy) | 0.734 → 0.653 | 0 |
Findings: a fresh 155 MB array costs ~57 ms of page faults to fill, a reused one 1.5 ms. SAWMOD
allocated 1.13 GB/freq (76k faults, 0.30 s system time); now 0.27 GB (24k, 0.10 s). The deferred GC
itself is cheap (young collection 2.4 ms); the page faults were the cost. Accelerate does not
thread gemms with 1–3 columns: row-chunk them over tasks (`perf/implicit_micro.jl`). User CPU is only
~0.8 s per 0.61 s iteration: BLAS stages are bound by the shared AMX units, not core count.
Estimated in-app 50-freq SAWMOD: ~34 s (41 − 50 × 0.15 × 0.94). Both switches are in the app.
**Measured in-app (rounds 7–10 together): 37.4 s** (was 51.2 s after round 6).

## Round 9 (2026-09-26): ideas from the CUDA backend, FEM stage
| Change | Harness | maxrel |
|---|---|---|
| `IMAGE_ACCUMULATE=1` | 1.007 → 0.984 (12 freqs) | 4e-7 |
| + `COMBINED_BM=1` | 0.955 → 0.927 | 5.4e-6 |
| + `MUMPS_EXPAND=1` | 0.934 → 0.923 | 0 |
| + `MASS_THREADS=4` | 0.923 → 0.910 | 5e-12 |
| 50-freq sweep, round 8 → round 9 | 0.910 → 0.823 | 5.8e-6 |
Fused image kernel (CUDA style): slower on Metal. MUMPS on OpenBLAS, other orderings, BLR: slower.
Estimated in-app 50-freq SAWMOD: ~41 s. All four switches are in the app. Details in HANDOFF.md.

## Round 8 (2026-09-26): GPU branch is the critical path; three bit-identical cuts
FEM stage work is 0.35 vs GPU branch 0.46: the "tie" was a misread of the overlapped timer.
| Change | Harness (12 freqs) | maxrel |
|---|---|---|
| `OP_POOL=1` | 1.057 → 1.021 | 0 |
| + `HOST_ROW_WEIGHTS=1` | → 0.997 | 0 |
| + `FLUX_SKIP=1` (S/K' gather + pair-kernel reduction pass) | 1.013 → 0.982 | 0 |
| 50-freq sweep, round 7 → round 8 | 0.972 → 0.899 | 0 |
| round 8 + `COUPLED_PREFETCH=1` | 0.977 (worse) | 0 |
MUMPS threads 6/8 and solve threads 2: no effect. 3M GEMM: ~10% micro, not worth it.
Estimated in-app 50-freq SAWMOD: ~45 s. All three switches are in the app.

## Round 7 (2026-09-26): faster F32 LU solve, stale LU as GMRES preconditioner
| Change | Harness | maxrel |
|---|---|---|
| `FAST_TRS=128` (blocked trsm + gemm LU solve, getrs 15 → 4 ms) | 1.104 → 1.076 s/freq (12 freqs) | 6.6e-8 |
| `STALE_LU=15` on top (50 log freqs, 20 Hz–20 kHz) | 0.966 → 0.935 s/freq | 1.8e-8 |
| `STALE_REUSE=12` + `STALE_STOP=12` | 0.942 (worse) | |
Offline GMRES iterations with the previous freq's LU (ratio 1.151): 5–6 at 40–53 Hz, 12–13 at
0.7–0.9 kHz, 45–59 at 6–8 kHz; two steps stale: 7–8, 17–18, 55+. Both switches are in the app.
Estimated in-app 50-freq SAWMOD: 51.2 → ~48 s (FAST_TRS ≈ −1.3 s, STALE_LU ≈ −1.5 s). Details in HANDOFF.md.

## Round 6 (2026-09-26) — Fable plan (beat-engine-fable/fable/PLAN.md). Handoff: perf/HANDOFF.md — START HERE
Best config so far = app + BLAB_TEST_GC_DEFER=1 + BLAB_TEST_BLOCKED_LU=512:
**1.224 -> 1.084 s/freq (1.13x)**, 12 freqs, Revise mode, maxrel 1.3e-7. In the app: 50-freq SAWMOD 58 -> 51.2 s (user, 2026-09-26).
Timer corrections (the handoff table double-counted): block_assembly_s CONTAINS
interface_elimination_s (pure assembly ~0.01); fem_schur_extraction_s CONTAINS
fem_transducer_solves_s. New timers: interface_elim_lu_{isfinite,stats,convert,getrf}_s and
test_prev_{gc_s,gc_pauses,alloc_gb,iteration_wall_s,emit_s}.
| idea (Fable)                          | result |
| I1 implicit dense matrix             | dead: assembly 0.01, LU glue 0.024 (getrf is 0.19 of 0.21) |
| I3 output stage                      | dead: errors/quantities/emit/release ~1 ms. "Overhead" = per-request setup at freq 1 |
| GC (found by timers)                 | 1.1 GB/freq, 5 pauses, 0.12-0.16 s. GC_DEFER=1 (GC off during a freq, one collection after): -0.05 s/freq, exact |
| I7 blocked LU (cgemm trailing update) | BLOCKED_LU=512: getrf 0.19 -> 0.12, refinement 1.5 -> 2 steps, -0.085 s/freq, maxrel 1.3e-7. perf/lu_micro.jl: cgetrf 0.3 TFLOP/s, blocked 0.6, cgemm 2.0. I7(b) LAPACK binding: already $NEWLAPACK |
| I2 early build (EARLY_BUILD=1 + PREFETCH=1) | exact, no crash, but 1.13 vs 1.085: only takes the field off the path; condensation/elim/LU ~15% slower next to GPU(i+2). Off |
| I6 split-precision elimination (ELIM_SPLIT=1) | dead: maxrel 2.7e-4 = F32 accumulation (cancellation), not input rounding |
| I5 MUMPS triangle check              | dead: all(iszero) short-circuits on the first upper entry |
Budget now (s/freq): first half 0.47 (GPU 0.42+bm 0.05 tied with condensation 0.47) + elim 0.15
+ LU 0.15 + solve 0.08 + field 0.09 + per-request setup ~0.1 (12 freqs; less at 50).
Still open from the plan: I8 (previous LU as GMRES preconditioner, skips the LU at low f),
I4 + MUMPS settings (both first-half branches must shrink together), I9 panel LU before S.


## Round 5 (2026-09-26) — START HERE
Checkpoint: git tag `metal-test-58s` (branch perf/experiments, pushed to the user's fork
BumelantPZA/BEAT_Engine) = the round-3 code the app ran at 58 s. Later commits build on it.

Harness: `quick.py --revise` runs perf/dev_worker.jl (Revise from perf/devenv, stacked by
JULIA_LOAD_PATH): Julia edits are applied before every solve, so the ~65 s start happens once.
Do NOT edit sources while a job runs (the next solve picks the edit up mid-job). struct/const edits:
touch queue/RESTART. Jobs must be named `<name>.job.json` (a plain .json is ignored; that is what
looked like a hang once). Revise mode measured ~6% slower than plain (1.69/1.51/1.46 vs
1.58/1.44/1.35, 3 freqs) — fine for in-worker A/B; use plain mode for headline numbers.

Findings (12 freqs x 2, big3.out; app config 1.19 s/freq):
- `fem_condensation_s` = max(condensation work, GPU + bem_matrix): the condensation's own work is
  0.38 (diag timer), GPU 0.42 + bem_matrix 0.10 = 0.52 → the first half is GPU-bound, not MUMPS.
- BLAB_TEST_BM_THREADED=1: threaded Burton-Miller combination, bit-identical. bem_matrix 0.10 -> 0.04,
  first half 0.54 -> 0.47. End to end within noise (1.19 vs 1.19). Candidate for the app (free).
- BLAB_TEST_ELIM_F32=1: interface-elimination block products in ComplexF32. product 0.127 -> 0.032,
  total 1.07 (1.11x) but maxrel 2.7e-4 (~0.002 dB; the fast field is 2.2e-5). User's call. Off.
- App (2026-09-26): BLAB_TEST_BM_THREADED=1 added to boundary-lab's METAL_TEST_SOLVER_OPTIONS
  (engine_distribution.py). boundary-lab has no user fork, so the app-side diff is kept here:
  perf/app_patches/engine_distribution.diff (git -C ../boundary-lab apply it to restore).
- Pipeline bisection DONE (2026-09-26): the pipeline is dead — slower even when correct. Code:
  perf/attic/pipeline_bisect.round5.diff (pipeline without locks + BLAB_TEST_PIPE_SERIALIZE=
  solve,dense,gpu test locks); sources restored to the checkpoint. Results (12 freqs, Revise mode):
  | config                              | s/freq | maxrel vs app | notes
  | app                                 | 1.14-1.18 | 0          |
  | pipeline, no locks (24 freqs)       | crash  |            | segfault in MUMPS load module, freq 2
  | serialize solve                     | 1.75   | 2.1e-1     | no crash (24 + 2x12 freqs), WRONG
  | serialize solve,gpu                 | 1.65   | 0          |
  | serialize solve,dense               | 2.19   | 2.4e-1     | WRONG
  | serialize solve,dense,gpu           | 2.14   | 0          |
  | two instances, no overlap (SERIAL)  | 1.24   | 0          |
  Causes: (1) crash = i's MUMPS solve calls (JOB=3, instance A) running concurrently with i+1's
  factorization (JOB=2, instance B): MUMPS keeps process-global Fortran module state (MUMPS_LOAD);
  the round-4 per-call lock was not enough because i+1's whole condensation must not overlap i's
  solve. A.fact -> B.fact -> A.solve ordering itself is fine (exact). (2) wrong results = i+1's GPU
  operator assembly concurrent with i's Metal field evaluation (shared GPU state, not found; no
  obvious module global). Latent: any future GPU overlap (e.g. prefetch overlapping the field)
  must serialize against field evaluation. (3) speed: every stage runs ~2x slower when two
  frequencies share the 10 cores (MUMPS/OpenMP + Accelerate + Julia threads), and the solve waits
  for i+1's condensation. Old round-4 crash reports: ~/Library/Logs/DiagnosticReports/julia-*.ips.
- (was) Next: pipeline crash bisection. Re-add the round-4 pipeline
  (perf/attic) WITHOUT the locks, reproduce with ~24 freqs, then serialize one overlap pairing at a
  time with a shared test lock: i+1 condensation vs i's elimination+LU (BLAS), vs i's solve
  (MUMPS/CHOLMOD), and i+1 GPU assembly vs i's field (GPU). Report after the bisection; max 2 fixes.
- GPU LU (perf/mps_lu_micro.jl): MPS real-embedded 2n LU 408 ms vs CPU ComplexF32 lu! 189 ms. Dead end.
Per-freq budget now: first half 0.47 (GPU-bound; condensation 0.38 right behind) + elimination 0.15
+ LU 0.21 + solve 0.07 + field 0.09 + ~0.08 request/output overhead.

## Round 4 status (2026-09-26, at compaction) — START HERE
**2026-09-26 later: ROLLED BACK to the round-3 code (the 58 s in-app version).** The user asked for
the previous working version. Removed: the pipeline (coupled_solver.jl), stage_gate, the MUMPS call
lock + MUMPS server task (BeatEngineMumps.jl), the CHOLMOD mass-solve lock and the diagnostic timers.
The round-4 copies of those three files are in perf/attic/*.round4.jl. Everything below about
BLAB_TEST_COUPLED_PIPELINE / the locks describes that removed code.
Open finding from reading the timers: `fem_condensation_s` starts before the GPU stage and ends at
the fetch after bem_matrix, so it is max(condensation work, GPU 0.42 + bem_matrix 0.13), not the
condensation alone. Its parts (factorization 0.24, Schur 0.04, transducer 0.04, mass 0.04) sum to
~0.35, so GPU + bem_matrix (~0.55) is probably the real first-half critical path. Unmeasured.
In-app (user): 50-freq SAWMOD 141 s (Apple Metal) -> 58 s (Apple Metal test). App test solver =
prefetch 0 + pair_tilereduce + Accelerate + BLAB_METAL_FIELD_FAST=3 + BLAB_TEST_DENSE_STATS=1.
Harness per-freq (12 freqs): ~1.25 s. CPU is mostly idle (~2 of 10 cores busy on average).
Critical path per freq: [FEM condensation 0.6 (MUMPS, ~1 core) || GPU assembly 0.45 + bem_matrix
0.13] then serial: interface elimination 0.16 + dense LU 0.23 + solve 0.08 + field 0.09.

Tried and FAILED (unsolved, don't loop on it): pipeline BLAB_TEST_COUPLED_PIPELINE=1 (off by
default) — frequency i+1's first half (GPU asm + condensation + bem_matrix) runs as a task gated
before its dense stage, while frequency i does elimination/LU/solve/field. Two MUMPS stores
alternate by frequency (pipeline_caches in coupled_solver.jl). Crashes (segfault / bus error inside
MUMPS factorization, once in plain Julia code) at random frequencies, only when overlapped.
Two alternating instances without overlap (BLAB_TEST_PIPELINE_SERIAL=1) work fine.
Already added: global lock around every zmumps_c call + its thread-count set (BeatEngineMumps.jl
_MUMPS_CALL_LOCK; harmless when sequential), a lock on CHOLMOD mass solves. Still crashed.
Untested idea in the code: `_on_mumps_server` runs all MUMPS calls on one long-lived task with a
256 MB stack (active only with the pipeline env), theory = MUMPS keeps pointers into task stacks.
If revisiting: consider instead a single MUMPS instance and doing frequency i's MUMPS work (RHS
reduction) before i+1's factorization starts, or drop the pipeline.
Other remaining ideas (lower gain): interface elimination product 0.12 (ComplexF64 GEMM);
MUMPS threads (BLAB_MUMPS_THREADS, default 4) / condensation internals.

## Round 3: CPU side and field (2026-09-26, after the kernel work)
App test solver now: prefetch OFF + pair_tilereduce + Accelerate + BLAB_METAL_FIELD_FAST=3 +
BLAB_TEST_DENSE_STATS=1. Big batch (12 freqs x 2, big2.log): app setting before 1.74 s/freq ->
1.31 (1.33x); same with prefetch on 1.46. maxrel 2.2e-5 (from the field, see below).
- Prefetch no longer helps: GPU assembly (~0.45) hides behind FEM condensation (~0.6) anyway,
  and prefetch makes bem_matrix (0.1 alone) contend with condensation (0.6).
- Fast field (`BeatEngineMetalFieldFast.jl`, BLAB_METAL_FIELD_FAST=1/2/3): 0.32 -> 0.09 s/freq.
  maxrel 2.2e-5 vs the precise kernel (~0.0002 dB). Mode 2 (precise sin/cos, fast rsqrt only) gives
  the same 2.2e-5, so this is float32 radius rounding amplified by the phase at 20 kHz, not
  fast-math error; the precise kernel carries the same size of error. Mode 3 = fast sin/cos after
  Cody-Waite range reduction (fma), used in the app.
- Dense factorization (`RefinedDenseLU`): opnorm(A, Inf) took 96 ms (row-major walk of a
  column-major matrix) + maximum(abs) 37 ms around a 186 ms Float32 LU. `_dense_abs_stats` does
  both in one threaded row-block pass, 10 ms, bit-identical. c_factorization 0.34 -> 0.23.
- Harness: quick.py job option "dump": true writes queue/<job>.<config>.r<n>.timings.json with
  every timing key (medians over freqs).
CPU breakdown now (prefetch off, s/freq): FEM condensation ~0.6 (MUMPS factorization 0.24,
Schur extraction 0.04, transducer solves 0.04, mass solve 0.06) runs alongside GPU assembly 0.45
+ bem_matrix 0.1; then serial: interface elimination 0.16 (a ComplexF64 product 0.12), dense LU
0.23, solve 0.08, field 0.09. Next candidates: the condensation (biggest; MUMPS settings or
reusing the symbolic analysis), the elimination product.

## Kernel work: Fable's plan implemented (2026-09-26)
Result: the far-field kernel went 1.62 -> ~0.45 s/freq (3.6x); SAWMOD end to end with prefetch
2.29 -> 1.63 s/freq (1.40x) with tilereduce + Accelerate. maxrel vs pair_gather 2.6e-7 (3 freqs) /
4.8e-7 (12 freqs, 20 Hz-20 kHz), limit 1e-6. Now the default of the app's "Apple Metal test" solver
(boundary-lab src/blab/solvers/engine_distribution.py METAL_TEST_SOLVER_OPTIONS: prefetch +
BLAB_METAL_REGULAR_KERNEL_MODE=pair_tilereduce + BLAB_TEST_BLAS=accelerate).
In the app (user, 2026-09-26, SAWMOD 50 freqs): Apple Metal 141.3 s vs Apple Metal test 79.4 s
= 1.78x (2.83 -> 1.59 s/freq, matches the harness).
Plan: `../../beat-engine-fable/fable/PROPOSALS.md`. Code: new files
`julia_local/src/BeatEngineMetalGatherV4Kernels.jl`, `BeatEngineMetalTileReduceKernels.jl` (included
from BeatEngineMetal.jl), mode wiring in BeatEngineMetalAssembly.jl / Common.jl, release hooks in
BeatEngineMetalRegular.jl. pair_gather is untouched and still the default.

Kernel stage split, s/freq (prefetch off, BLAB_METAL_GATHER_TIMING=1, BLAB_TEST_ASM_TIMING, medians;
the machine was shared with other apps, runs vary ~±15%):
| mode                                   | kernel | pairs | gather D/H | gather S/K' |
| pair_gather (baseline)                 | 1.62   | 0.80  | 0.60       | 0.21        |
| tile 32x4 (no code)                    | 1.58   | 0.75  | 0.62       | 0.21        | (8x32/64x4 worse)
| pair_gather_v4 (float4 groups)         | 1.05   | 0.66  | 0.24       | 0.13        | bit-identical
| v4 + BLAB_METAL_V4_PACKED=1 (load diet)| 0.61-0.67 | 0.21 | 0.25     | 0.13        | 2.5e-7
| pair_tilereduce TY=8, min-node order   | 0.58   | 0.43  | 0.09       | 0.07        |
| pair_tilereduce TY=16, patch order     | 0.44-0.49 | 0.32-0.36 | 0.07 | 0.04       | default
- Load diet = float4 points/normals/curls + rule constants as a Val tuple (compile-time) + both
  quadrature loops unrolled. Biggest single win (pair stage 3.3x). Not bit-identical (fastmath
  reassociation after unrolling), 2.5e-7.
- tilereduce: Fable's padded tables failed its own go check (S_max 39 > 32, padded 2.44x), so slots
  are CSR per tile (unpadded). Test order = greedy compact patches (1.12 slots/element vs 1.41 for
  min-node sort; BLAB_METAL_TILEREDUCE_ORDER=min_node for the old one). BLAB_METAL_TILEREDUCE_TY
  (default 16; 8 slower), BLAB_METAL_TILEREDUCE_BARRIER=simd (SIMD-group barriers: no faster, off).
  Results are bit-reproducible run to run and across TY/barrier variants.
- The reduction roughly doubles the pair stage (0.21 -> 0.32); the gathers drop 0.38 -> 0.11.
  BLAB_TEST_TR_SKIP_REDUCE=1 is useless for attribution (the compiler then drops the maths too).
- Idea 4 (images summed in the gather) skipped: gathers are 0.11 < the plan's 0.15 threshold.
- Test hooks: BLAB_TEST_DUMP_GATHER_MESH=<file> (element order + P1 dofs, v4 mode) feeds
  perf/tiletables.py and perf/patchorder.py (host-only slot statistics). perf/asmsum.py
  summarizes BLAB_TEST_ASM_TIMING files.
End to end, prefetch on, 12 freqs x 2 rounds (big1.log):
  pf 2.29 | pf+v4p 1.88 (1.21x) | pf+tr 1.83 (1.25x) | pf+v4p+acc 1.73 (1.32x) | pf+tr+acc 1.63 (1.40x)
  Accelerate now pays off because the GPU is no longer the bottleneck.
Next bottleneck (outside the kernel): with prefetch, `bem_matrix` takes ~0.6 s/freq (0.13 without
prefetch): burton_miller_neumann_matrices + dense products on the CPU, contending with the FEM
condensation task. Also fem_condensation ~0.6-0.7 and field 0.31 (GPU). Remaining kernel idea: a
cheaper in-group reduction (fewer phases / staged slot tables), worth ~0.1 s at most.
Exterior-only note: metal_direct_assembly_available() requires pair_gather, so with tilereduce an
exterior-only project takes the non-fused Metal path (correct, possibly slower); SAWMOD is coupled.

## Status at handoff (2026-09-25 evening)
- Prefetch works and is confirmed in the app: warm SAWMOD solve 29.9 s (Apple Metal) vs
  24.6 s (Apple Metal test) = 1.22x, identical results. Nothing is running.
- The app offers both solvers side by side (see "Using the test build in the Boundary Lab GUI").
- The app froze once at the end of a solve: a Qt/PySide deadlock in upstream GUI code, not the
  solver. Fixed locally (see "GUI freeze fix"); watch whether it recurs.
- 2026-09-25 later: options 1 (Accelerate) and 2 (kernel group size) tested, no end-to-end
  gain (see "Round 2 findings"). Neither is enabled in the app test build. The remaining
  offer: upstream the prefetch, or one of the GPU-side ideas listed there.
- Working style the user wants: short test cycles (quick.py, not 110 s full runs), report
  progress, diagnose a failure once instead of restarting in a loop.

Test checkout of BEAT Engine v0.2.0, branch `perf/experiments` (uncommitted edits).
The user's real install (`../boundary-lab`, BEAT 0.2.0 from the release wheel) is untouched;
this copy is loaded only via `PYTHONPATH=<this checkout>/src` with boundary-lab's venv python.
Machine: M1 Pro, 8P+2E cores, 16-core GPU, 16 GB. Boundary Lab's default is 10 Julia threads.

## Code changes (test only)
- `julia_local/src/BeatEngineCoupledCondensed.jl`: `condensed_metal_operators(...)` plus the
  `prefetched_operators` / `on_operators_ready` kwargs on `build_condensed_coupled_system`.
- `julia_local/coupled_solver.jl`: with `BLAB_TEST_COUPLED_PREFETCH=1` the coupled loop starts
  frequency i+1's GPU BEM operator assembly as soon as frequency i's operators are ready.
  `BLAB_TEST_PREFETCH_BLAS_THREADS` optionally sets the BLAS threads while prefetch is active.
  A `solver_options["test_env"]` hook in both request handlers sets per-request env vars.

## Findings (s/freq, SAWMOD, Metal)
- Baseline 2.70-2.89 (12 freqs, ±7% run to run). Stages per freq: GPU BEM operator 1.5-1.6
  (critical path), FEM condensation ~0.5 of real work (overlapped with the BEM stage, its timer
  includes the wait), then host-only: bem_matrix 0.12, block 0.25, elim 0.23, dense fact 0.36,
  Metal field 0.31 (7,322 points, also on the GPU).
- The existing cross-frequency pipeline only covers exterior-only solves, not coupled ones.
- Prefetch: 2.21-2.24 (12 freqs), 1.29x on 2 runs each; 2.44 vs 2.76 in the 6-freq warm A/B
  (1.13x, dilutes because frequency 1 can't be prefetched). Outputs bit-identical (0.0 rel).
  Peak memory +~400 MB. It loses some gain to contention: bem_matrix 0.12->0.42 and field
  0.31->0.60-0.74 while the next assembly runs on the GPU.
- GPU floor with the current kernels ≈ 1.6 operator + 0.3 field ≈ 1.9 s/freq.
- No effect (within noise): 8 vs 10 threads, MUMPS 6 threads, prefetch with BLAS 9.
  Worse: MUMPS 2 threads (2.89), stage overlap off (3.15).

- In the app (user, 2026-09-25, same SAWMOD solve, 8 Julia threads which is the GUI's Metal
  default): first solve 84.2 old vs 79.2 test (incl.
  ~54 s Julia start + compile each); second, warm solve 29.9 vs 24.6 s = 1.22x.

## Round 2 findings (4 freqs x 2 rounds unless noted; SAWMOD, warm worker, 10 threads)
- Kernel group size: 512 can't run (the fused singular kernel is capped at 384 threads by
  register use). 64/128/256: GPU assembly 1.65-1.66 s/freq either way, totals within noise,
  with and without prefetch. No gain.
- Accelerate vs OpenBLAS: micro (n=3116) LU c32 249->199 ms, c64 430->356, GEMM c32 122->27
  (4.5x), c64 242->120. In the solve (12 freqs x 3 rounds, prefetch on): block+elim+fact
  1.15->0.76 s/freq, but total 2.37->2.33 (1.01x) because the field eval grew 0.48->0.78:
  with prefetch the run is GPU-bound (next-freq assembly ~1.65 + field ~0.3 of a 2.33 s cycle).
  Prefetch off: 2.81->2.79 (1.01x). MUMPS/FEM condensation unchanged. maxrel 6e-8..8e-8.
  Switch left in the test solver: BLAB_TEST_BLAS=accelerate (per request; off by default).
  Accelerate needs the "\x1a$NEWLAPACK$ILP64" suffix hint (as AppleAccelerate.jl uses);
  without "\x1a" libblastrampoline only forwards LP64 and every ILP64 call fails.
- Where the 1.65 s GPU assembly goes (BLAB_TEST_ASM_TIMING=<file>, BLAB_METAL_GATHER_TIMING=1):
  regular (far-field) kernel ~1.6 s, same at 100 Hz and 15 kHz; singular+images <0.1 s.
  Inside it: pair values 0.81, gather DLP/hypersingular 0.60, gather SLP/adjoint 0.22.
  Coupled path already uses quadrature order 2 (the minimum).
- Regular kernel modes (prefetch off, 3 freqs): pair_gather 1.6 s (default), pair_atomic 3.2,
  pair_owned 5.5, entry_owned 29. Gather budget MB 128/256/512/2048: 1.69/1.58/1.61/1.89
  (256 vs 512 is noise; bigger is worse).
- Harness bug fixed: quick.py used to unset only the env keys the current job named, so a
  previous job's settings leaked (one job ran with prefetch on by accident). It now resets
  every key any job has used.

## Ideas not yet tried (GPU is the bottleneck with prefetch)
- The user wants to optimize the core kernel itself (not batching, not moving work): short
  handoff for outside ideas in `GPU_KERNEL_HANDOFF.md`. For a Fable cloud session: brief is
  `CLAUDE.md` on branch `fable/gpu-kernel` (worktree `../../beat-engine-fable`, from v0.2.0),
  user guide `FABLE_CLOUD_GUIDE.md`. Fable writes `fable/PROPOSALS.md` only (no code); Claude
  implements each idea as a new BLAB_METAL_REGULAR_KERNEL_MODE here and A/Bs it with quick.py.
- Assemble two frequencies per kernel pass (geometry shared, only e^{ikr} differs) to cut the
  0.81 s pair stage; fits the prefetch pipeline.
- Take the field evaluation (~0.3 s GPU, 0.5-0.8 under contention) off the GPU critical path:
  CPU evaluation, or defer it so it doesn't overlap the next assembly.
- Ordering field evaluation to avoid contending with the prefetched assembly.
- Accelerate becomes worthwhile when the dense steps dominate again (larger models: dense
  O(n^3) grows faster than assembly O(N^2)).

## Harness
- Quick A/B, one warm worker (~60 s warm-up once, then ~80 s per two-config job):
  start `PYTHONPATH=$PWD/../src ../../boundary-lab/.venv/bin/python quick.py &` from perf/,
  wait for `queue/READY`, then
  `./job.sh j02 '{"configs":{"pf":{"BLAB_TEST_COUPLED_PREFETCH":"1"},"pf_gs128":{"BLAB_TEST_COUPLED_PREFETCH":"1","BLAB_METAL_KERNEL_GROUPSIZE":"128"}}}'`
  Restart it after editing Julia sources. Stop with a plain `kill` (SIGTERM is handled).
- Full runs: `./run.sh <name> <threads> ENV=VAL ...` (benchmark_worker.py, 12 freqs, 2 repeats,
  ~110 s each) and `python3 cmp.py <names...>`. Never run two benchmarks at once.
- Pitfalls hit: relative `--out` paths break the worker (use absolute); overlap setting is
  `off` not `0`; a chained `run.sh` once kept running after a failure, overlapping two
  benchmarks and invalidating that batch (fixed: run.sh now returns the real exit code).

## Using the test build in the Boundary Lab GUI
boundary-lab has a local (uncommitted) solver entry "BEAT Engine (Apple Metal test)" = id
`beat_metal_test`: same "metal" requests, but the solver script and Julia project come from this
checkout, and requests carry `solver_options.test_env = {BLAB_TEST_COUPLED_PREFETCH: "1"}`.
Code: `src/blab/solvers/engine_distribution.py` (METAL_TEST_* paths), `registry.py`,
`beat_engine_runtime.py` (alias metal_test -> metal), `coupled_backend.py`
(PhysicalSystemProductionBackend). Only shows up if this checkout exists. Expected side effect:
`tests/test_solver_backends.py::test_solver_backend_registry_offers_only_physical_backends` fails.
Verified end to end (3 freqs SAWMOD via headless path): results bit-identical to beat_metal,
bem_operator_s ~0 on prefetched frequencies. Edits to the Julia sources here take effect
after restarting the app (the worker is persistent).

## GUI freeze fix (boundary-lab, uncommitted)
Symptom: window froze right after a solve finished (process alive, ~1% CPU, no crash report;
the "session ended" line missing from `~/.boundary-lab/logs/startup-*.log`). `sample <pid>`
showed the main thread in `QStackedWidget.setCurrentIndex` -> layout -> `QObject::connectImpl`
waiting on Qt's connection mutex, while the solve QThread ran the worker's `deleteLater` ->
`~QObject` -> `QThreadWrapper::disconnectNotify` -> `PyGILState_Ensure` (waiting for the GIL).
Fix: `src/blab/ui/system_solve.py` and `src/blab/ui/generator_worker.py` call
`self.moveToThread(QCoreApplication.instance().thread())` before `finished.emit()`, so the
worker is destroyed on the GUI thread. Verified with a small Qt script (destroyed on main
thread) and 65 passing GUI controller tests; the race itself can't be forced. Worth reporting
upstream. If the app freezes again: `sample <pid> 3` and read the main thread and QThread stacks.

## Undo everything
- boundary-lab: `git checkout -- src/blab/solvers src/blab/ui` (removes the test solver entry
  and the freeze fix).
- This checkout: `git checkout -- src` and delete `perf/` (also removes the BLAB_TEST_BLAS and
  BLAB_TEST_ASM_TIMING hooks).
