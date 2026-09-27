# SAWMOD Metal performance experiments (2026-09-25)

## Round 17 S4 (2026-09-27): SAWMOD CPU chain (R17-3 stopped, R17-6 kept)
Branch A (M1: CPU chain leads by ~36 ms). Checkpoint tag metal-test-round17-pre-s4.

**R17-3 micro** (`perf/mumps_contention_micro.jl <fem.jls> [trials] [its]`, 7 trials, medians, ms). MUMPS stays on
Accelerate; the loop beside it = its x (ComplexF32 FAST_TRS nb=128 solve, 3 columns + F64 3116^2 x 3 zgemm):
| loop (its) | variant | MUMPS alone | loop alone | MUMPS beside | loop beside | MUMPS + |
|---|---|---|---|---|---|---|
| 9 | a Accelerate | 175.6 | 70.3 | 204.6 | 88.1 | 29.0 |
| 9 | b OpenBLAS64 4 thr | 174.9 | 121.0 | 184.9 | 129.6 | 10.0 |
| 9 | b OpenBLAS64 8 thr | 173.2 | 87.4 | 221.6 | 144.5 | 48.4 |
| 9 | c threaded Julia | 204.0 | 165.1 | 218.4 | 318.3 | 14.4 |
| 18 | a Accelerate | 185.0 | 146.8 | 236.8 | 180.4 | 51.8 |
| 18 | b OpenBLAS64 4 thr | 178.7 | 246.0 | 203.4 | 262.0 | 24.7 |
| 18 | b OpenBLAS64 8 thr | 180.2 | 237.6 | 251.2 | 363.1 | 71.0 |
| 18 | c threaded Julia | 224.0 | 376.1 | 249.8 | 602.1 | 25.8 |
Gate (9 its, as specified): best is OpenBLAS 4 threads, 19.7 ms better than Accelerate < 25 ms: **R17-3 stopped**.
The 18-its run (loop ~ the real stale GMRES, 147 ms alone) saves 33 ms on MUMPS, but the loop grows 180 -> 262 ms
beside MUMPS; scaled to the real stale solve (0.208 s) that is ~0.30 s, past the FEM stage (0.295 s), so the solve
would turn critical. No `BLAB_TEST_SOLVE_BLAS` switch was added. Park for Macs where AMX contention is larger.
Pitfall: `openblas_set_num_threads64_` takes a C `int` by value; passing `Ref{Int64}` gives OpenBLAS a huge thread
count and every call takes ~150-200 ms. Use `OpenBLAS_jll.libopenblas_handle` for the direct ccalls.

**R17-6** MUMPS 5.9.1 guide: ICNTL(20)=1,2,3 conflicts only with ICNTL(32)=1; ICNTL(26)=1/2 says the right-hand side
"can be dense, sparse or distributed". `BLAB_TEST_MUMPS_SPARSE_RHS=1` (`_solve_phase!` kwarg in `BeatEngineMumps.jl`,
used by `mumps_reduce`; double-precision struct only): columns as Int32 CSC, `RHS` stays the dense output array.
Hook `BLAB_TEST_DUMP_REDUCE=<file>` dumps the transducer columns (6 columns, 1613 nonzero rows of 24947).
Micro (`mumps_loop_micro.jl <fem.jls> 40 <reduce.jls>`): reduce 14.3 -> 4.5 ms, the expansion after it 10.7 -> 11.0
(unchanged), reduce and expand maxrel 2e-16 vs dense.
| Set | base s/freq | srhs s/freq | maxdB |
|---|---|---|---|
| V-S SAWMOD 50 f, 3 rounds (base = app env + r16 switches incl. FIELD_MULTI) | 0.565 (r 0.565/0.603/0.563) | **0.551** (0.551/0.590/0.523) | 0.0003 |
| V-C vented_sub 12 f, 2 rounds | 0.184 | 0.181 | 0.0000 |
| V-C compression_driver 12 f, 2 rounds | 0.016 | 0.016 | 0.0000 |
Per-round gain -14/-13/-40 ms (median -14). **Kept**, added to `perf/app_patches/round17_engine_distribution.diff`.
The machine was busier than in M1 (base 0.565 vs 0.49), so compare within the job only.

## Round 17 S3 (2026-09-27): prototype2 GPU lane (R17-2, R17-5, M4, R17-4; R17-7 skipped)
All configs carry the round 16 exterior switches (r16). Checkpoint tag metal-test-round17-pre-s3.
| Step | Config | proto2q s/freq | proto2 full s/freq | maxdB |
|---|---|---|---|---|
| R17-2 | r16 -> `BLAB_METAL_PIPELINE=1` (50 f / 12 f, 2 rounds) | 0.057 -> 0.047 | 0.241 -> 0.240 | 0 vs r16 (all values identical) |
| R17-5 | pipe1 -> `SING_SPLIT=0.4` / `0.6` (200 f, 2 rounds) | 0.045-0.046 -> 0.043 / 0.043-0.044 | | 0.0001 / 0.0004 vs pipe1 |
| R17-5 | SAWMOD 12 f: `BLAB_TEST_POOL_ZERO2=1` (COMBINED_BM zeroes 2 of 4 buffers) | SAWMOD 0.814 -> 0.804 (noise) | | 0, bit-identical |
| R17-4 | pipe1+split4 -> `SING_PACKED=2` (V-P) | 0.046 -> 0.041 | 0.235 -> 0.222 | 0.0015 / 0.0009 vs unpacked |
| all | 200 f, 2 rounds: stock / r16 / pipe1+split4+packed2 | 0.070 / 0.054 / **0.040** | | 0.0019 vs stock (r16 0.0022) |

**M4 probes** (proto2q 12 f, 2 rounds, r16, synced stage timers; probes give wrong results by design):
| kernel | maxThreads | base ms | per-probe ms |
|---|---|---|---|
| `_metal_singular_fused_bm_blocks_kernel!` (sing_blocks) | 384 | 8.6 | no store 8.5, no maths 1.15 (-87 %) |
| image singular (same launch, image transforms) | | 3.5 | no maths 2.45 |
| `_metal_fast_field_kernel!` (field_s) | 1024 | 14.3 | no cis 12.9 (-10 %), no loads 13.2 (-8 %) |
| `_test_multi_field_kernel!` (SAWMOD, nd 3) | 1024 | | not probed |
R17-4 gate passed (384 < 768, no maths -87 %). R17-7 gate failed (no loads -8 % < 15 %): skipped.

**R17-4 details.** `BLAB_TEST_SING_PACKED` (Float32): pairs grouped once per singular cache by rule length
(prototype2 has 512 / 1280 / 1536 points), one launch per group with point and part counts as Vals, float4
rule points (test xi, eta, trial xi, eta), float4 vertices / normals / curls; stock output layout, so the
gather is unchanged. 1 = vertices reloaded per point, 2 = loaded once, 3 = 2 with the loop bound read at run
time. All three 512 threads; 2 and 3 same speed. Same source arithmetic, but 1e-6 relative differences
(Float32 fast-math codegen, not unrolling: 3 gives the same 0.0024 dB vs r16 at 12 f). Packed is slightly
*closer* to stock than unpacked (0.0018 vs 0.0021 dB proto2q, 0.0013 vs 0.0015 full), so it is Float32
noise, but the step gate was 0.001 dB. The user approved it for the app patch (2026-09-27).

**New hooks.** `BLAB_TEST_SING_PROBE` (1 no store, 2 no maths), `BLAB_TEST_FIELD_PROBE` (1 no cis, 2 fixed
source), PIPEINFO lines `sing_fused_bm`, `sing_packed_n<N>`, `field_fast`, `field_multi_nd<N>`;
`BLAB_TEST_SAVE_RESULTS=<file>` (every emitted result as a JSON line) + `perf/cmp_results.py a b` (exact /
maxrel / maxdB across jobs, e.g. old code vs new code: probes off were bit-identical on proto2q and SAWMOD).
Pitfall: `quick.py` compares only within a job and dumps no outputs; use SAVE_RESULTS for cross-code checks.
App patch draft: `perf/app_patches/round17_engine_distribution.diff` (after round 16's).

## Round 17 S2 (2026-09-27): R17-1 split done, bundle not extended (user declined the dependency change)
Gate: M3 X ~60/45 s >= 15 s, so R17-1 went ahead; Y ~4.5 s < 10 s, so no R17-1b micro. R17-1c declined.
Done (commit 55db2b5): `julia_local/BeatEngineCoupledWorker.jl` = the coupled engine as a module (exports
`run_worker, solve_request, reclaim_accelerator_memory!, test_cold_log`); `coupled_solver.jl` = 84-line loader
(solver.jl pattern). The loader loads the bundle only if the bundle's source mentions `BeatEngineCoupledWorker`, so today
it always includes from source (no second engine copy); `BLAB_BEAT_ENGINE_BUNDLE=0` forces that. Load-time side
effects: BLAS forwarding and all `ENV` reads were already runtime; runtime caches reset in `__init__`, MUMPS
handle `LIBRARY[]` reset in `BeatEngineMumps.__init__`. `tests/memory_mesh_tests.jl` now includes the module.
Revise: `quick.py --revise` 12-freq smoke OK (r16 0.0004 dB as before), and an edit in the module file hot-reloads.
Not done (needs the bundle Project.toml change): steps 4-5 (bundle workload, `precompile_bundle.sh`), so no
cold-start gain yet. Remaining work for a later yes: add `BeatEngineCoupledWorker.jl` + MUMPS/Serialization deps to
the bundle, a CPU coupled workload, `precompile` for Metal host entries, re-resolve `julia_metal`, time the precompile.

App path, round-16 switches via `APP_TIMING_EXTRA` (to first freq / total, s; one run each, same machine state):
| Project | before (pre-s2) cold | after cold | before warm | after warm |
|---|---|---|---|---|
| SAWMOD 50 freqs | 66.2 / 97.7 | 66.5 / 93.7 | 1.94 / 28.0 | 1.79 / 26.3 |
| proto2_quarter 200 | 50.6 / 62.4 | 51.1 / 62.9 | 0.21 / 11.24 | 0.20 / 11.27 |
Bit-identity (`perf/cmp_dumps.py`, everything outside `diagnostics`): SAWMOD IDENTICAL (50 freqs, 3550 values),
proto2q IDENTICAL (200 freqs, 5200 values). Precompile time: not measured (no bundle change).

## Round 17 S1 (2026-09-27): measurements M1, M2, M3
Hooks added (commit 55a196b, no effect when unset; smoke-tested on SAWMOD and proto2q, identical results):
`BLAB_TEST_DELAY_EXT_SOLVE=<s>` (exterior host solve), `BLAB_TEST_COLD_LOG=<file>` (process env: start-up
stamps with epoch + cumulative compile time, request phases, first-launch wall per Metal kernel name via
`@_test_cold_launch`), an `overlap_plan ...` line in `BLAB_TEST_PHASE_LOG`, `mkjob.py --request=<file>`.
HANDOFF history moved to `HANDOFF_ARCHIVE.md`.

**M1, SAWMOD lanes** (50 freqs, 3 rounds interleaved, harness; deltas vs `r16` within each round, ms/freq):
| Config | r1 | r2 | r3 | median | reading |
|---|---|---|---|---|---|
| r16 (FIELD_MULTI) s/freq | 0.495 | 0.502 | 0.467 | | |
| gpu50 (GPU lane +50 ms) | +3 | +10 | +5 | **+5** | GPU slack ~45 ms |
| fem50 (CPU chain +50 ms) | +41 | +39 | +42 | **+41** | CPU slack ~9 ms |
| mumpsfast (CPU chain ~-80 ms, wrong results) | -74 | -76 | -66 | **-74** | nearly all of it shows |
| gpufast (GPU lane ~-150 ms, wrong results) | -8 | -20 | -12 | **-12** | |
| ty8 (`BLAB_METAL_TILEREDUCE_TY=8`, identical results) | +1 | -33 | -8 | -8 | noisy; re-test in R17-5 |
**Verdict: the CPU chain leads** (fem50 - gpu50 = 36 ms >= 15, mumpsfast gains 74 >= 15). A CPU-chain cut of
~80 ms buys ~74 ms; a GPU cut buys ~12 ms. Changed since R15b (tied): FIELD_MULTI took the GPU lane down.

**M2, prototype2 quarter pipeline** (50 freqs, 3 rounds, all rounds equal to 1 ms):
| Config | s/freq |
|---|---|
| base (stock app env) | 0.070 |
| r16 (3 exterior switches), auto plan | 0.055 |
| pipe0 (`BLAB_METAL_PIPELINE=0`) | 0.055 |
| **pipe1 (`BLAB_METAL_PIPELINE=1`)** | **0.047** (-8 ms, -15 %, same maxrel 1.4e-6 / 0.0021 dB as r16) |
| depth2 (`=1`, `PIPELINE_DEPTH=2`) | 0.047 |
| solve10 (host solve +10 ms) | 0.069 (+14 ms: the solve is fully serial) |
The auto plan is **off**: `overlap_plan enabled=false reason=model assembly_model_s=0.067 solve_model_s=0.0019
saving_model_s=-0.0011`. The model's solve estimate (1.9 ms) is far below the real serial host time (~8 ms
recovered by pipelining). R17-2 is a switch-only fix: `BLAB_METAL_PIPELINE=1` for the test solver (or fix the
model's solve estimate in `BeatEngineSweepOverlap.jl`). Still to check: proto2 full mesh and 200 freqs (V-P).

**M3, cold start** (app path, `BLAB_TEST_COLD_LOG` as process env, run 1 cold / run 2 warm; summary with
`perf/cold_sum.py <log>`). Seconds from the worker script's first line (Julia runtime start before it ~1 s);
"compile" = `Base.cumulative_compile_time_ns` (summed over threads, so it can exceed wall):
| Phase | SAWMOD wall (compile) | proto2q wall (compile) |
|---|---|---|
| `using` JSON/LinearAlgebra/... | 0.4 (0.2) | 0.4 (0.2) |
| engine `include`s | 4.1 (1.3) | 4.4 (1.4) |
| to worker ready (backend check loads Metal) | 3.2 (1.9) | 3.4 (2.1) |
| request read -> env (request parse + system setup JIT) | 21.2 (21.2) | 25.8 (25.4, to first freq) |
| env -> first frequency (setup) | 7.5 (6.4) | (in the row above) |
| first frequency | 35.0 (52.9) | 18.5 (17.5) |
| of which Metal kernel first launches (Y) | 4.3 (8 kernels; tilereduce pair 1.6) | 4.7 (11; packed pair 1.7) |
| then steady | 0.51 s/freq | 0.060 s/freq |
| app_timing: to first freq / total | 70.6 / 98.0 s (warm 1.8 / 27.9) | 53.3 / 65.5 s (warm 0.26 / 11.6) |
Package-load floor (`using Metal, JSON, StaticArrays, SparseArrays`): 1.98 s.
**X (includes + host JIT that a precompiled bundle can cache) ~ 60 s SAWMOD, ~45 s proto2q; Y ~ 4.5 s.** So
R17-1 is in (X >> 15 s); R17-1b is out (Y < 10 s). Nearly all of the cold start is Julia JIT of engine code,
spread over request setup and the first frequency, not package loading or Metal shader compilation.
Q5 (warm one-time cost, SAWMOD run 2): request -> env 0.04, setup 0.65, first frequency 1.15 (vs 0.52 steady),
emit 0.21 s: ~1.5 s one-time per request.

## Round 11 (2026-09-26): in-app BLAS bug, MUMPS pivoting, early build with optimized prefetch
In-app path (`perf/app_timing.py`, the app's headless solve on the SAWMOD project, 50 freqs, warm):
**37.7 s → 33.9 s (BLAS fix) → 29.4 s (prefetch opt) → 27.1 s (FEM lane) → 25.1 s (all, 10 threads)**.
The user measured 37.4 s in-app before. Harness 50-freq sweep 0.655 → 0.468 s/freq.
Vented_Sub (captured `perf/vented_sub.json`): 0.149 → 0.123 s/freq vs round-10 settings, maxrel 4.1e-8;
compression_driver (interior FEM path) unchanged, bit-identical; waveguide (proto2) bit-identical.
| Change | Harness (50-freq sweep) | maxrel |
|---|---|---|
| Fix: MUMPS loaded before the Accelerate forward (`test_apply_blas!`). OpenBLAS32's lazy JLL library forwards itself into libblastrampoline on first dlopen; the app applied `BLAB_TEST_BLAS` first, so MUMPS ran on OpenBLAS32 in-app (FEM stage 0.35 vs 0.29, 2.5 s user CPU/freq from spinning OpenBLAS threads). The harness warmup loaded MUMPS first and never saw it | in-app 0.725 → 0.649 steady | 0 |
| `MUMPS_CNTL=1=0`: no numerical pivoting (never a delayed pivot; the search cost 0.04 of 0.232 s). Failed factorization retries with CNTL(1)=0.01 | 0.642 → 0.623 | 4.4e-8 |
| `FEM_F32_SKIP=1`: no Float32 FEM system built only to be replaced by the Float64 one | 0.624 → 0.618 | 0 |
| `PREFETCH_OPT=1` + `EARLY_BUILD=1` + `COUPLED_PREFETCH=1`: prefetched GPU operators now use the round 8/9 switches (shared `_test_metal_operators`); build(i+1) overlaps field(i) | 0.625 → 0.571 | 0 |
| `ZERO_RHS_SKIP=1`: no MUMPS reduction / mass solve of an all-zero FEM right-hand side (voltage excitations) | 0.574 → 0.549 | 0 |
| `FEM_LANE=2`: build(i+1) starts when build(i) returns; its FEM stage overlaps i's solve and field, its dense part waits for solve(i). Only when nothing after the FEM stage calls MUMPS | 0.558 → 0.492 | 0 |
| `FEM_INPLACE=1`: F64 FEM values written into the stiffness pattern (index maps) | fem_system 0.016 → 0.008 | 0 |
| `EXPAND_OVERLAP=1`: transducer interior solve (MUMPS expand) beside the mass presolve | 0.475 → 0.468 | 0 |
| App: Metal test backend threads = `os.cpu_count()` (was 8) | app path 27.1 → 26.0 s | 0 |
Didn't help: FEM_LANE=1 (LU doubles next to MUMPS), MUMPS per component / concurrent instances
(crash), look-ahead LU inside the pipeline, other stale-reuse limits, QoS. Also: a sleeping Metal wait (Metal.jl `synchronize` spins/yields; only ~7% of busy samples),
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

## Round 12 (2026-09-26): extensive app-path tests, MUMPS workspace
App path (`app_timing.py`, SAWMOD 50 freqs, warm): test 24.9 s (0.47 s/freq) vs stock Metal 131.4 s
(2.61 s/freq), 5.3x. Cold first run: test 84.7 s (60.4 s to the first frequency), stock 181.6 s (53.1 s):
the coupled solver is `include`d from source every worker start (only the exterior `solver.jl` has a
precompiled bundle), so the first solve after an app start pays ~55-60 s of load and JIT.
Accuracy test vs stock (complex pressure, 3 excitations x 7322 points): maxrel <= 1.0e-4 per frequency
(20-100 Hz, stale-LU GMRES range), <= 3e-5 above; max |dB| 0.15 at near-null points at 15 kHz.
Coil current 2.6e-5, diaphragm velocity 3.4e-6 (`scratchpad/cmp_raw.py`, APP_TIMING_DUMP=<pkl>).
Per-frequency CPU: 1.2 s user + 0.15 s system per 0.42 s wall, 24k minor faults/freq; MUMPS alone
faults its ~200 MB factor workspace in anew on every factorization (12.6k faults, 19 ms system).
`BLAB_TEST_MUMPS_WK=1`: persistent WK_USER buffer per solver (INFO(8) x (1 + ICNTL(14)/100)).
Micro (`perf/mumps_wk_micro.jl`): 233 -> 218 ms, 12.6k -> 3.0k faults. Harness bit-identical:
SAWMOD 0.467 -> 0.450 s/freq (2 rounds), Vented_Sub 0.134 vs 0.135 (4 rounds, noise ±0.007),
compression_driver 1.01x, proto2 (no MUMPS) 0.310 vs 0.308.

### Round 13 (2026-09-27): per-step SAWMOD breakdown (harness, 50 freqs, MUMPS_WK on)
Serial (EARLY_BUILD/FEM_LANE/PREFETCH off) 0.598 s/freq vs pipelined 0.465 (`queue/r13_steps.*`).
Per step, serial (alone) / pipelined (contended), s: GPU BEM operators 0.246 / hidden; MUMPS
factorization 0.180 / 0.233 (+30 %); MUMPS reduce 0.015 / 0.021, expand 0.014 / 0.018; Schur
extraction 0.015 / 0.022; interface-mass solve 0.017 / 0.023; FEM system 0.006 / 0.007; FEM stage
total 0.266 / 0.295; bem_matrix 0.020 / 0.026; block assembly 0.016 / 0.025; interface elimination
0.014 / 0.018; dense F32 LU on fresh freqs (29/49) getrf 0.108 / 0.123 + convert 0.018 + stats 0.011 +
isfinite 0.004; solve fresh 0.041 / 0.056, stale GMRES (20/49) 0.150 / 0.208; field 0.083 / 0.080.
Pipelined critical chain: FEM stage 0.295 -> elimination ~0.02 -> LU 0.14 on fresh freqs.

### Round 13b (2026-09-27): MUMPS study (the FEM stage is the critical chain)
Micros on the SAWMOD FEM dump (`perf/mumps_param_micro.jl`, `mumps_loop_micro.jl`, `cmumps/`):
- Profile (`sample`): MUMPS 97 % of the loop; zgemm 41 %, ztrsm 24 % (half in Accelerate's
  dispatch_apply), front assembly 14 %, stack/copies 8 %. 4.19 GFLOP in 0.18 s = 23 GFLOP/s.
- Orderings: METIS (auto) 4.19 GFLOP / 0.18 s; AMD, AMF, SCOTCH, PORD, QAMD all 7.77 GFLOP / 0.25 s.
- BLR (ICNTL(35)=2, CNTL(7) 1e-12..1e-6): 0.27-0.34 s, slower (fronts too small).
- KEEP(3..6) blocking sweeps: +-2 ms. VECLIB_MAXIMUM_THREADS 2/4/6 = default, 1 slower (0.21).
- ztrsm and/or zgemm on OpenBLAS32 via lbt_set_forward: 0.196-0.229 s, slower.
- BLAB_MUMPS_THREADS only sets OpenBLAS's pool: with Accelerate forwarded it does nothing.
- Contention in the pipeline: MUMPS(i) 0.271 s when frequency i-1 ran a stale-LU GMRES solve beside
  it (0.19 s, ~9 its), 0.203 s after a fresh solve, 0.183 alone. The GMRES streams the Float64
  3116^2 matrix (155 MB) per iteration.
- `BLAB_TEST_MUMPS_SINGLE=1` (cmumps, CMumpsStruc mirror, offsets from perf/cmumps/offsets.c):
  factorization 0.186 -> 0.100 s, Schur maxrel 2.2e-7 at the dump frequency. Pipeline 0.455 ->
  0.428 s/freq only (the GPU lane, operators 0.25 + field 0.08, then limits: bem_operator wait
  0.11 -> 0.22), and outputs maxrel 3e-3 at 20 Hz, 2e-4..9e-4 through 60 Hz-1.2 kHz, < 1e-5 only
  above 10 kHz (low-frequency cancellation in the Schur complement). Rejected; left off by default.
Conclusion: MUMPS is at its floor in double precision; the pipeline is now balanced between the CPU
chain (~0.45) and the GPU lane (~0.33 + interlocks), so the next gain needs the GPU BEM kernel too.

### Round 14 (2026-09-27): accuracy policy, dB metric, field kernel mode 4
- The user adopted the accuracy policy now at the top of HANDOFF.md (0.01 dB = invisible; ~0.001 dB
  per change OK; >= 0.1 dB rejected; in between ask). quick.py reports `maxdB` (within 60 dB of each
  excitation's peak) and per-frequency dB with `"detail": true`.
- Re-judged, SAWMOD 50 freqs vs the current test backend: MUMPS_SINGLE 1.11 dB (interface velocity at
  46.6 Hz), 0.433 vs 0.469 s/freq -> rejected. ELIM_F32 0.10 dB (exterior pressure 781 Hz), 0.464 vs
  0.469 -> rejected. Both: 0.416, 1.10 dB.
- Test backend vs stock settings: 0.035 dB at 17-20 kHz, all from BLAB_METAL_FIELD_FAST=3 (without it
  0.0014 dB but 0.654 s/freq). Modes 1/2: 0.054/0.035 dB (so not sin/cos).
- BLAB_TEST_FIELD_F64=1 (Float64 CPU field from the same surface data) as the true reference:
  SAWMOD worst dB: stock field 0.030, mode 3 0.045, mode 4 0.029. prototype2 quarter: stock 0.024,
  mode 3 0.022, mode 4 0.018, all 0.071 s/freq (stock field 0.099).
- MODE 4 (`BLAB_METAL_FIELD_FAST=4`): mode 3 with a precise sqrt for the distance (the phase k*r
  reaches ~1e3 rad at 20 kHz, 3 m). Same speed, now the app test setting. Precise sin/cos on top
  (tried): no gain, field 0.08 -> 0.15 s. Kahan accumulation: not measured (removed).

### Round 15 (2026-09-27): GPU operator timing study (SAWMOD and prototype2)
Tools (all test-only, off by default): `BLAB_TEST_GPU_MICRO=<file>` + `_VARIANTS="label:K=V,K=V;..."`
+ `_REPS`: at the first assembly of each wavenumber the worker reruns that assembly per variant alone
on the GPU and logs every stage (`scratchpad/msum.py` summarizes). `BLAB_TEST_PIPEINFO=<file>`: compiled
pipeline limits (maxThreads < 1024 = register-limited occupancy). `BLAB_TEST_TR_PROBE` (tile-reduce
pair kernel: 1 maths only, 2 no maths, 3 no sin/cos, 4 no skip test, 5 no maths + no store, 6 no
maths + no image read-modify-write), `BLAB_TEST_TR_FIRST_ONLY=1` (identity transform only),
`BLAB_TEST_FUSED_PROBE` (exterior fused pair kernel: 1 no store, 2 no maths), fused singular sub-stages
under `BLAB_METAL_GATHER_TIMING=1`, `BLAB_TEST_FIELD_INFO=<file>` (field call sizes and walls).
Probes give wrong results: timing only.

SAWMOD (6054 elements, 3-point rule, 3110 P1, xy symmetry = 4 transforms), alone on the GPU, ms per
frequency, same at 200 Hz / 5 kHz / 20 kHz: **operators 250** = pair kernel 214 + gathers 20 (A 14.5,
C 5.5) + singular 12 + image singular 2.3 + operator zeroing 2.8. Field: 3 calls (one per excitation)
x 28 ms = 84 (7322 points x 72648 sources).
- Pair kernel split: maths only 103-121, reduction only 86-98 (roughly additive). No sin/cos -6. Per
  transform: maths ~32, reduction ~21. Reduction without any device store -3, without the image
  read-modify-write -11: the reduction cost is threadgroup memory + barriers + slot loops.
- Occupancy: tile-reduce pair kernel maxThreads 512 of 1024 (12 KB threadgroup memory); maths-only 576,
  reduction-only 832; no-skip variant 448 and +35 % time (same work): occupancy-bound maths.
- TY=8: 202 vs 214 (-12 ms, free); TY=4 worse. Singular parts 2/8/16/32: none better than 4.
- **End-to-end ceiling:** 50-freq pipelined sweep with the operators 2.5x faster (identity only):
  0.515 -> 0.496 s/freq (4 %). On the M1 Pro, SAWMOD is CPU-chain bound (MUMPS -> LU); the GPU lane
  (ops 0.25 + field 0.08) hides behind it. GPU work pays on SAWMOD only on Macs with a weaker GPU
  relative to the CPU, or after the CPU chain shrinks.

prototype2 quarter (1233 elements, 6-point rule = 36 evaluations per pair, 677 P1, 4 transforms),
exterior fused path (upstream's kernel: no packed loads, no tile reduction, gathers per transform):
**assembly 49** = pairs 22 + lhs gather 11 + rhs gather 6 + singular 8.1 + image singular 3.6 (of which
value-buffer zeroing 1.4, Sauter-Schwab blocks 8.6, gathers 1.6; 15377 adjacent pairs, 4 parts) +
alloc/identity/row weights/rhs reduce ~3.5. Field 1 call, 12.6 ms (7322 x 29592).
- Pair kernel is maths-bound: no store -2 ms, no maths 5.6 ms. maxThreads 384 (register-limited).
  Tile 16x16/32x8/8x32/32x4 and chunk budget 128/512/2048 MB: all within 0.5 ms.
- **End-to-end ceiling:** 50-freq sweep with the pair maths removed: 0.073 -> 0.057 s/freq (1:1).
  prototype2 is GPU-bound; every GPU millisecond counts.
Plan with five targets: `perf/GPU_PLAN.md`.

### Round 15b (2026-09-27): definitive bottleneck test for SAWMOD (lane delay sensitivity)
Idle waits (sleep: no CPU/GPU use) added to one lane, 50-freq pipelined sweep, 2 rounds interleaved:
`BLAB_TEST_DELAY_GPU=<s>` (end of the GPU operator assembly) and `BLAB_TEST_DELAY_FEM=<s>` (end of
the FEM stage, the start of the CPU chain).
| Config | s/freq | vs base |
|---|---|---|
| base | 0.509 (0.502-0.516) | |
| GPU lane +100 ms | 0.587 | +78 ms |
| CPU chain +100 ms | 0.579 | +70 ms (the sleep also frees cores: less contention beside it) |
| both +100 ms | 0.598 | +89 ms |
With round 15's speed-up test (GPU operators -150 ms -> only -19 ms/freq): **the two lanes are tied;
the CPU chain is longer by ~20 ms per frequency (~4 % of the cycle)**. Both tests give the same ~20 ms
GPU slack (+78 = 100 - 22; -19). They are coupled, not independent: the dense step waits for the GPU
matrices, and the field waits for the CPU solution. Consequence: shortening either lane alone gains at
most ~20 ms/freq (~1 s per 50 freqs); larger gains need both lanes shorter.

### Round 16 (2026-09-27): GPU_PLAN targets T1-T5
**T1 (exterior fused kernel), done.** Switches (test-only, off by default): `BLAB_TEST_FUSED_IMAGE_ACC=1`,
`BLAB_TEST_FUSED_PACKED=2`. Harness, quick.py, maxdB vs stock settings:
| Project | stock | T1 | dB |
|---|---|---|---|
| prototype2 quarter + xy symmetry, 50 freqs | 0.077 (0.073 rerun) | 0.058 s/freq (1.26-1.33x) | 0.0021 |
| prototype2 full mesh, no symmetry, 12 freqs | 0.279 | 0.247 (1.13x) | 0.0015 |
| SAWMOD 12 freqs (coupled path, untouched) | 0.658 | 0.660 | 0 (bit-identical) |
- Step 1 `FUSED_IMAGE_ACC=1`: transforms 2-4 add into the pair blocks, gathers once per chunk: 0.077 -> 0.071.
- Step 2 `FUSED_PACKED=2`: float4 points/normals/curls, rule constants as Val, trial fold unrolled, test
  loop at runtime: 512 threads/group (was 384), 0.069 -> 0.058. `FUSED_PACKED=1` (both loops unrolled,
  36 bodies) is slower: 0.080.
- Step 3 (tile-reduce gathers) skipped: after step 1 the gathers are 4.4 ms (lhs 2.9 + rhs 1.5) and the
  SAWMOD tile reduction costs ~21 ms per transform.
- Step 4 `FUSED_POOL=1` (pooled singular value buffers and rhs partials): 0.058 vs 0.058, no gain; left off.
- Dead end: `FUSED_TR` draft (exterior via the coupled tile-reduce COMB kernels): 0.146 vs 0.082, the
  6-point rule makes that kernel 85 ms (vs 22). Diff in perf/attic/t1_fused_via_tilereduce.diff.
Stage split after T1 (proto2q, synced timers): pairs 18.9 ms (4 transforms), lhs gather 2.9, rhs 1.5,
singular 8.8 (blocks 8.9 of it), image singular 4.2.

**T2 (leaner tile-reduce pair kernel, SAWMOD), no gain; code left behind switches, off.**
GPU micro (`BLAB_TEST_GPU_MICRO`, SAWMOD alone on the GPU, ms per assembly, pair kernel / wall):
| Variant | pairs | wall | maxThreads |
|---|---|---|---|
| base (COMB) | 216-225 | 257-263 | 512 |
| `TR_EARLY_COMB=1` (per-test-point combine, unrolled test loop) | 323 | 361 | 448 |
| `TR_EARLY_COMB=2` (runtime test loop) | Metal compiler fails at pipeline link ("Compilation to native code failed") | | |
| `METAL_TILEREDUCE_TY=8` | 207-215 | 244-255 | |
- Early combination costs occupancy here (448) although it helps the fused exterior kernel (T1): this
  kernel also holds 12 KB threadgroup memory and the barrier/reduction code. Step 2 (image loop in
  registers) needs step 1, so it was not attempted (round 9 already showed it is slower without it).
- TY=8: bit-identical, -9 ms GPU alone, but the 50-freq pipelined sweep is 0.547 vs 0.532 (noise band;
  the CPU lane leads by ~20 ms, round 15b). Worth re-testing on a GPU-weaker Mac; not switched on.
- Harness note: a failed request kills quick.py, and its `.job.json` stays in queue/ and is re-run at the
  next start: delete leftover `queue/*.job.json` before restarting. The GPU micro hook runs once per
  (file, k): use a new file name after a failure.

**T3 (field).** Part 1 done, part 2 rejected.
- Part 1 `BLAB_TEST_FIELD_MULTI=1` (`evaluate_galerkin_field_metal_multi`, `_test_multi_field_kernel!`):
  all excitations of a frequency in one pass (Green's value per (point, source) once, ND weight sets,
  each drive summed in the stock order). SAWMOD 50 freqs, 2 rounds: **0.530 -> 0.504 s/freq**, field
  section 0.09 -> 0.04, maxdB 0.0005. Vented_Sub and prototype2 (one excitation): falls back, bit-identical.
  Hook: coupled_solver.jl `exterior_pressure` output (no excitation weights).
- Part 2 `BLAB_TEST_FIELD_FAR_RULE=c/kappa` (group of R sources -> one weight-centroid source when
  |x - centroid| > c h and k h < kappa): prototype2 quarter, 50 freqs, vs stock:
  | Setting | s/freq | maxdB |
  |---|---|---|
  | stock / FIELD_MULTI | 0.074 / 0.075 | 0 |
  | c=10, kh<0.3 | 0.072 | 20.2 |
  | c=5 or 20, kh<1 | 0.069 / 0.071 | 36.7 |
  | c=10, any kh | 0.066 | 42.9 |
  Rejected: the pressure varies linearly over an element, so a centroid source is only first-order
  accurate, and the quiet points of the 60 dB window amplify it. Even the best case saves only 8 ms.
  A correct version needs per-drive first moments (dipole correction) or a lower-order rule, not tried.

**T5 (singular split), exterior path only; small gain, off by default.** `BLAB_TEST_SING_SPLIT=<kappa>`:
G0 = 1/(4 pi r) parts per (pair, part) once per singular cache and transform (Sauter-Schwab, k = 0 pass
`_test_singular_static_kernel!`, 25 floats), per frequency `_test_sing_split_kernel!` combines them with
k and adds G1 = (e^{ikr}-1)/(4 pi r) on the regular R x R rule (cancellation-free forms). Used only while
k h_max < kappa. prototype2 quarter, 50 freqs, on top of T1:
| kappa | s/freq | maxdB vs T1 |
|---|---|---|
| off (T1) | 0.055-0.057 | 0 |
| 0.4 | 0.052 | ~0 (0.0021 vs stock, same as T1) |
| 0.6 | 0.055 (noisy run) | 0.0004 |
| 0.8 | 0.054 | 0.0023 |
| 1.2 | 0.058 | 0.107 |
| 2 | 0.054 | 0.30 |
| ungated | 0.048 | 56 (5.6 kHz+: > 16 dB) |
The G0 part is right (< 0.0001 dB below 400 Hz); G1 on the regular rule fails on touching elements once
k h grows (its -k^2 r/(8 pi) term has a kink at r = 0). Only the low-frequency part of a sweep can use it,
so the gain is ~2-3 ms/freq (~4 %). Not ported to the coupled (SAWMOD) singular kernel: 14 ms in the GPU
lane, which is not the longer lane on this M1 Pro.

**T4 (distance-adaptive quadrature), rejected.** `BLAB_TEST_FAR_ORDER=rho/kh` (exterior fused packed kernel,
T1 mode 2 only, rules > 3 points): far pairs (centroid distance >= rho x sum of circumradii, k x radii <
kh) use the 3-point rule in a second launch (`FAR` 1 = near pass, 2 = far pass; one kernel holding both
rule bodies crashes the Metal compiler at pipeline link, like TR_EARLY_COMB=2). prototype2 quarter,
50 freqs, 1 round, machine busy (WindowServer/Beeper ~45 % CPU each; absolute times ~+50 %):
| Setting | s/freq | maxdB vs stock |
|---|---|---|
| T1 | 0.079 | 0.0021 |
| rho 2, any kh | 0.076 | 42.2 |
| rho 3, any kh | 0.078 | 43.8 |
| rho 5, any kh | 0.080 | 49.9 |
| rho 3, kh < 1 | 0.079 | 0.24 |
The pair kernel is not the only cost any more after T1 (gathers, singular, field, solve), the second
launch re-reads every pair, and the 3-point rule cannot follow the phase once k h ~ 1. SAWMOD already
uses the 3-point rule (only a 1-point rule would be cheaper, first-order like T3 part 2): not tried.

**Round 16 summary, app path** (`perf/app_timing.py`, the app's headless solve, run 2 = warm; machine
shared with WindowServer/Beeper at ~45 % CPU each, so absolute times are above the round 14 baselines;
`APP_TIMING_EXTRA` adds switches on top of the app's test_env):
| Project | current app settings | + round 16 switches | gain |
|---|---|---|---|
| SAWMOD 50 freqs (+FIELD_MULTI) | 32.5 s (0.618 s/freq steady) | 28.0 s (0.518) | -14 % |
| prototype2 quarter 200 freqs (+FUSED_IMAGE_ACC, FUSED_PACKED=2, FIELD_MULTI) | 14.3 s (0.070) | 12.3 s (0.060) | -14 % |
Accuracy (quick.py, vs stock): SAWMOD 0.0005 dB, prototype2 0.002 dB. App patch:
`perf/app_patches/round16_engine_distribution.diff` (not applied; the user applies it).
