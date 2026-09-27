# HANDOFF history archive (moved out of HANDOFF.md on 2026-09-27, round 17 S1 step 0)

Status sections after rounds 10-15, the round 10 budget, round 11 picture and round 12 survey, verbatim.

## Status after round 15 (2026-09-27): GPU timing study done, plan in perf/GPU_PLAN.md
- Operators alone: SAWMOD 250 ms (pair kernel 214 = maths ~120 + in-group reduction ~90), field 84;
  prototype2 quarter 49 ms (pairs 22, gathers 17, singular 12), field 12.6. Kernels are
  occupancy-limited (512 / 384 threads per group).
- End-to-end: prototype2 is GPU-bound (1:1). SAWMOD (definitive test, NOTES round 15b): CPU and GPU
  lanes are tied, CPU longer by ~20 ms/freq (~4 %); either lane +100 ms costs +70-78 ms, either lane
  alone faster gains <= ~20 ms. Say "tied, CPU ahead by ~20 ms", not "CPU-bound" or "GPU-bound".
  Five targets T1-T5 in GPU_PLAN.md; the user picks.

## Status after round 14 (2026-09-27) and next step
- App path SAWMOD ~24 s warm (stock Metal 131 s); prototype2 quarter 13.9 s (stock 20.5 s). Tags up to
  `metal-test-round14`. App patches in `perf/app_patches/` (user applies; field mode now 4).
- MUMPS is exhausted in double precision (NOTES round 13b). The pipeline is balanced: CPU chain
  (MUMPS 0.23 -> dense LU 0.14 on fresh freqs) vs GPU lane (BEM operators 0.25 + field 0.08).
- Next candidates: (1) GPU BEM operator kernel (helps SAWMOD's GPU lane and the prototype, whose fused
  exterior kernel spends 16 of 47 ms in two gather passes); (2) cold start ~40-60 s per app launch
  (coupled_solver.jl has no precompiled bundle); ask the user which first.

## Where things stand (after round 11, 2026-09-26)
- Goal: shorten the whole SAWMOD coupled FEM-BEM solve (50 freqs) in Boundary Lab's
  "BEAT Engine (Apple Metal test)" solver. In-app, 50 freqs: 141 s (stock) → 58 s (round 3) → 51.2 s
  (round 6) → 37.4 s (rounds 7–10, user-measured).
- **Round 11: app path (`perf/app_timing.py`, the app's own headless solve, warm, 50 freqs) 37.7 s →
  25.1 s.** Not yet measured by the user in the GUI. Harness 50-freq sweep 0.655 → 0.468 s/freq.
  - The in-app gap was a bug: MUMPS ran on OpenBLAS32 in the app (see NOTES round 11). Fixed.
  - MUMPS without pivoting (`MUMPS_CNTL=1=0`), `FEM_F32_SKIP`, `ZERO_RHS_SKIP`, `FEM_INPLACE`,
    `EXPAND_OVERLAP`. Checked on Vented_Sub (−17%, maxrel 4.1e-8), compression_driver and the waveguide.
  - Pipeline: `COUPLED_PREFETCH=1` + `EARLY_BUILD=1` + `PREFETCH_OPT=1` + `FEM_LANE=2`: build(i+1)
    (FEM stage, then elimination/LU gated behind solve(i)) starts when build(i) returns; GPU(i+1) is
    prefetched with the full round 8/9 GPU switches; field(i) waits for GPU(i+1), GPU(i+2) for field(i).
  - App patch: the Metal test backend uses `os.cpu_count()` Julia threads (10 vs 8: −4%);
    `perf/app_patches/coupled_backend.diff`.
- Code: this checkout, branch `perf/experiments`, pushed to the user's fork
  (`git push fork perf/experiments`, BumelantPZA/BEAT_Engine). **Never push to `origin` (JWSound).**
- Restore tags: `metal-test-58s`, `metal-test-bm-threaded`, `metal-test-round6` … `metal-test-round10`,
  `metal-test-mumps-blas-fix`, `metal-test-round11` (prefetch opt), `metal-test-round11b` (FEM lane).
- App side: `../boundary-lab/src/blab/solvers/engine_distribution.py` (`METAL_TEST_SOLVER_OPTIONS`) and
  `coupled_backend.py` (threads). No fork; diffs in `perf/app_patches/`. Restart the app after engine edits.

## Bottleneck survey, round 12 (2026-09-26, app path unless noted, M1 Pro, warm = run 2)
| | Test | Stock Metal | Split (test, per freq) |
|---|---|---|---|
| SAWMOD 50 freqs | 24.9 s warm (0.47), cold 84.7 s (60.4 to 1st freq) | 131.4 s warm (2.61), cold 181.6 (53.1) | CPU chain: FEM stage 0.33 (MUMPS 0.27) + LU 0.11 on fresh freqs; now MUMPS_WK -0.017 |
| prototype2 quarter, 200 freqs | 13.9 s (0.068), cold 59.1 (44.6) | 20.5 s (0.101), cold 58.5 (38.9) | GPU-bound: assembly 47 ms (regular kernel 35 = pairs 21 + lhs gather 11 + rhs gather 5.5; singular 7.3 + image 2.3; misc 3.5), field 14, CPU solve 5 |
Accuracy test vs stock: SAWMOD pressure maxrel <= 1.0e-4/freq (stale-LU range), coil current 2.6e-5;
prototype2 3.8e-5 (2 of 1.46 M points > 0.1 dB, both 74-83 dB below the max). `scratchpad/cmp_raw.py`.
Prototype sample (`sample` on the worker): host 78 % idle in Metal completion waits, cgetrf 5 %, GC 2 %.
Tried and rejected: wavelength quadrature on Metal (order 2 while kh <= 2): only 0.071 -> 0.062 s/freq,
maxrel 0.27 (order-2 assembly 37 vs 47 ms: the regular kernel is gather-bound, not evaluation-bound).
Biggest remaining items: (1) cold start ~40-60 s on the first solve after an app start, both
backends: `coupled_solver.jl` includes the engine from source (only the exterior driver has a
precompiled bundle, `julia_engine/BeatEngineMetalBundle`); (2) SAWMOD: MUMPS (single instance,
~55 % of the cycle); (3) prototype: the fused exterior kernel's two gather passes (16 of 47 ms).
Tools: `BLAB_TEST_FUSED_TIMING=<file>` (fused assembler stage split per call, with
`BLAB_METAL_GATHER_TIMING=1` for pairs/gathers), `perf/proto2q.json` (quarter-mesh request),
`APP_TIMING_DUMP=<pkl>` in app_timing.py.

## Next session: extensive tests of prototype and SAWMOD solves (user request, 2026-09-26)
State: everything committed and pushed (last `7ffde50`); harness stopped; the user has not yet pushed
the app patches (`perf/app_patches/engine_distribution.diff`, `coupled_backend.diff`) nor restarted
the app. App-path tool (the app's own headless solve, persistent worker, run 2 = warm):
```
cd ../boundary-lab
.venv/bin/python ../beat-engine-test/perf/app_timing.py examples/Multi_region_SAWMOD/Multi_region_SAWMOD.blab.json 2 "" 50
APP_TIMING_BACKEND=beat_metal .venv/bin/python ../beat-engine-test/perf/app_timing.py ../beat-engine-test/perf/proto2_quarter.blab.json 2 ""
```
Args: project, runs, threads ("" = backend default, now os.cpu_count() for the test backend), freq
count override. Env: `APP_TIMING_BACKEND` (beat_metal_test default | beat_metal = stock),
`APP_TIMING_SYMMETRY`. Per-frequency Julia timers of the last run land in the scratchpad
`app_rows<run>.json` (path hard-coded in app_timing.py; change it for a new session).
Baselines (M1 Pro, warm):
| Project | Test solver | Stock Metal |
|---|---|---|
| SAWMOD 50 freqs | 25.1 s (0.47 s/freq) | not measured this session (in-app 141 s originally) |
| prototype2 quarter mesh + xy symmetry, 200 freqs (`proto2_quarter.blab.json`; the GUI solves on the generator's `_reduced` mesh) | 13.8 s (0.068) | 18.9 s (0.094) |
| prototype2 full mesh, symmetry off (`proto2.blab.json`) | 39.5 s | 46.2 s |
User's own in-app numbers before round 11: SAWMOD 37.4 s; prototype2 stock 12.4 s (why ours is 18.9 s
is not explained: maybe fewer field points in the GUI). Accuracy harness jobs (maxrel vs a config):
`"request": "proto2.json" | "vented_sub.json" | "compression_driver.json" | "sawmod.json"` in quick.py
jobs (captured requests in perf/, some gitignored; recapture with scripts/capture_boundary_lab_request.py).
Ideas for the tests: stock vs test on SAWMOD through app_timing (accuracy: compare output quantities
too, not only time), 200-freq SAWMOD, cold first run (Julia start ~40-60 s), repeated runs on one worker.

## Per-frequency picture after round 11 (harness, 50-freq sweep, 0.468 s/freq)
The critical chain is CPU only: build(i+1) = FEM stage 0.31 (MUMPS 0.25; 0.19 when alone) → gate/
elimination 0.02 → F32 LU 0.11 fresh (29/50) / ~0 stale. Solve(i) (0.06 fresh, 0.17 stale GMRES) and
field(i) (GPU 0.08) run beside FEM(i+1). GPU lane: ops 0.25 + field 0.08 per 0.47 cycle (not critical).
Contention is now the limiter: anything run beside MUMPS slows it (GMRES traffic ≈ +0.03, the LU
doubles next to MUMPS: FEM_LANE=1 lost). MUMPS_seq cannot run two instances concurrently (crash) and
has no OpenMP; one FEM component holds 19492 of 24947 vertices, so splitting does not help.
One-time per request: setup ~0.45 s, first frequency ~1.0 s (vs 0.47 steady).
**Rule: any new overlap must be measured in the pipelined sweep; standalone micro gains (look-ahead LU
−15%) vanished inside it.**

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



# Moved from HANDOFF.md at round 17 close (2026-09-27)

## Status round 17 S4 (2026-09-27, NOTES "Round 17 S4")
- Branch A (CPU leads). R17-3 **stopped**: micro gate failed (OpenBLAS64 4 threads beside MUMPS only 19.7 ms
  better than Accelerate < 25), and a GMRES-length loop grows past the FEM stage on OpenBLAS. No switch added.
- R17-6 **kept**: `BLAB_TEST_MUMPS_SPARSE_RHS=1` (ICNTL(20)=1 in `mumps_reduce`), reduce 14.3 -> 4.5 ms.
  SAWMOD 50 f 0.565 -> 0.551 s/freq (busy machine), 0.0003 dB; V-C 0.0000 dB. In the round 17 app patch draft.
- New: `perf/mumps_contention_micro.jl`, `mumps_loop_micro.jl` 3rd arg (reduce dump), `BLAB_TEST_DUMP_REDUCE`.
- Open: R17-4 for the coupled singular kernels, R17-8 (needs Q2), R17-3 as other-Mac item. App patch not applied.

## Status round 17 S3 (2026-09-27, NOTES "Round 17 S3")
- prototype2 quarter, 200 freqs: stock 0.070, r16 0.054, **S3 0.040 s/freq** (-26 %), 0.0019 dB vs stock.
- R17-2 done: `BLAB_METAL_PIPELINE=1` (bit-identical; the overlap model underrates the solve). R17-5 done:
  `SING_SPLIT=0.4` (0.0001 dB), `POOL_ZERO2=1` (SAWMOD bit-identical, gain in noise).
- R17-4 done for prototype2: `SING_PACKED=2`, 512 threads, singular 7.8 -> 2.7 ms; 0.0015 dB vs unpacked
  (above the 0.001 gate, closer to stock than unpacked): user approved it for the app patch (2026-09-27).
- R17-7 skipped (M4: no-loads probe -8 % < 15 %). M4 table in NOTES.
- Cross-code exact checks: `BLAB_TEST_SAVE_RESULTS=<file>` + `perf/cmp_results.py`.
- App patch draft `perf/app_patches/round17_engine_distribution.diff` (after round 16's); not applied.
- Next: SAWMOD CPU chain (R17-3 micro, R17-6), per M1. quick.py still running (user stops it).

## Status round 17 S2 (2026-09-27, NOTES "Round 17 S2")
- R17-1 **stopped after steps 1-3**: user declined extending BeatEngineMetalBundle (deps + julia_metal re-resolve).
  R17-1c declined too. R17-1b not needed (Y ~4.5 s).
- Done: coupled engine is now module `julia_local/BeatEngineCoupledWorker.jl`; `coupled_solver.jl` is a thin loader
  that uses the bundle only if the bundle carries that module (today never), else includes from source.
- Results bit-identical to pre-S2 (SAWMOD, proto2q); cold/warm times unchanged. Revise harness works, module edits
  hot-reload. Cold-start gain: 0 s until the bundle is extended (then expected ~-40..-55 s per M3).
- `perf/cmp_dumps.py a.pkl b.pkl`: exact compare of APP_TIMING_DUMP pickles.
- Next: R17-5, then by M1 verdict (plan §3), or R17-1 steps 4-6 if the user approves the bundle change.

## Status round 17 S1 (2026-09-27, NOTES "Round 17 S1"; plan perf/PLAN_ROUND17.md)
- M1 SAWMOD: **CPU chain leads by ~36 ms** (GPU +50 ms costs +5, CPU +50 ms costs +41; MUMPS ~-80 ms buys -74,
  GPU -150 ms buys -12). Next for SAWMOD: R17-3 micro, then R17-6.
- M2 prototype2 quarter: `BLAB_METAL_PIPELINE=1` 0.055 -> 0.047 s/freq (identical results); the overlap model
  turns it off (solve model 1.9 ms vs ~8 ms real). R17-2 = switch/model fix; verify on proto2 full + 200 freqs.
- M3 cold start: ~60 s (SAWMOD) / ~45 s (proto2q) of the wait is host JIT + includes (X), Metal kernel compile
  only ~4.5 s (Y). So S2 = R17-1 (bundle), R17-1b dropped. Warm one-time cost ~1.5 s/request.
- New hooks: `BLAB_TEST_DELAY_EXT_SOLVE`, `BLAB_TEST_COLD_LOG` (process env), overlap_plan line in PHASE_LOG,
  `mkjob.py --request=`.
- **S1 closed (tag metal-test-round17s1). Next: S2 (R17-1).** Ask PLAN Q1 first (extend `BeatEngineMetalBundle`
  + precompile after each edit; app pre-start R17-1c). Cold runs: M3 commands in NOTES, `perf/cold_sum.py <log>`.
  quick.py is stopped; the user stops it themselves (pkill is blocked for Claude by a permission check).

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

