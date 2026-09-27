# Round 17 plan: BEAT Engine (Apple Metal test) speed

**Context.** Round 16 (tag `metal-test-round16`) left SAWMOD at 28.0 s warm on the app path (stock
131 s) and prototype2 quarter at 12.3 s (0.060 s/freq). Cold start (45-60 s before the first frequency)
is untouched. The round 16 app patch is **not applied** yet (checked 2026-09-27: `engine_distribution.py`
has no `FIELD_MULTI` / `FUSED_*` keys), so every job below adds those switches explicitly. Goal of round
17: cut what the user feels (cold start, both projects' sweeps) and stay useful on other Macs. The
first session only measures, and its results pick the order of the rest.

---

## 1. Where the time goes now (M1 Pro)

### SAWMOD (coupled, 50 freqs): harness 0.504 s/freq with FIELD_MULTI; app 28.0 s (busy machine)
These are stage times from timers. They are not a claim about the bottleneck.
| Lane | Stage | s/freq | Source |
|---|---|---|---|
| CPU chain | FEM stage (MUMPS factorization 0.183 alone, 0.203 after a fresh solve, 0.271 after a stale GMRES; mean 0.231) | 0.295 | R13, R13b |
| | interface elimination | ~0.02 | R13 |
| | F32 LU on fresh freqs (29/49): getrf 0.123 + convert/stats/isfinite 0.033 | 0.156 fresh (0.092 mean) | R13 |
| | solve(i), running beside FEM(i+1): fresh 0.056, stale GMRES 0.208 (20/49) | | R13 |
| GPU lane | BEM operators: pair kernel 214 ms (maths ~120 + in-group reduction ~90), gathers 20, singular 14, zeroing 3 | 0.25 | R15 |
| | field (FIELD_MULTI) | ~0.04 | R16 |

Causal facts:
- R15b: the lanes were tied, with the CPU ahead by ~20 ms (GPU +100 ms cost +78 ms, CPU +100 ms cost +70 ms, GPU -150 ms gained only -19 ms).
- R16: the GPU lane got 50 ms shorter (field) and the sweep gained 26 ms (0.530 -> 0.504). That is more than the 20 ms slack, because the field sits on the path that crosses lanes (solve(i) -> field(i) -> GPU(i+2)).
- **Which lane leads now is unknown. M1 answers it.**

A measured cost that is not a timer share: MUMPS runs **48 ms/freq slower inside the pipeline than alone**
(0.231 vs 0.183). 20 ms of that shows on every frequency. The other 28 ms (the mean of a 68 ms extra) comes
after stale-GMRES frequencies.

### prototype2 quarter (exterior, 200 freqs): 0.058-0.060 s/freq, GPU-bound 1:1
The causal test was R15: removing 17 ms of pair maths saved 16 ms/freq.
| Stage (synced timers, R16) | ms/freq |
|---|---|
| pair kernel (4 transforms) | 18.9 |
| gathers (lhs 2.9 + rhs 1.5) | 4.4 |
| singular (Sauter-Schwab blocks) | 8.8 |
| image singular | 4.2 |
| field (7322 points x 29592 sources) | 12.6 |
| alloc / identity / row weights / rhs reduce (R15) | ~3.5 |
| **sum** / **wall** | **52.4** / **58-60** |

About 6-8 ms/freq fall outside every GPU stage: the host dense solve (~5 ms, R12) and host glue. Whether
they overlap GPU work or leave the GPU idle is unknown (M2). A model decides the overlap
(`metal_sweep_overlap_plan`, `src/BeatEngineSweepOverlap.jl:98`). No round has measured
`BLAB_METAL_PIPELINE` or `BLAB_METAL_PIPELINE_DEPTH`.

### Cold start (app path, R12)
SAWMOD waits 60.4 s before the first frequency (84.7 s total); prototype2 waits 44.6 s (59.1 s total).
Stock Metal waits 53.1 s and 38.9 s.
- Both projects go through `coupled_solver.jl`, which `include`s ~16 k lines on every worker start: itself, `BeatEngineCoupled`, `CoupledCondensed`, `Core` and the backend files.
- The precompiled `BeatEngineMetalBundle` is already a path dependency of `src/beat_engine/julia_metal`, but only the exterior `solver.jl` loads it.
- How the wait splits between package load, `include`, host JIT and Metal kernel compilation is unknown (M3).
- For scale: one cold start costs about 2.3x a warm SAWMOD solve and 3.6x a warm prototype2 solve.

### How other Macs shift this
- A stronger GPU (Max) shortens the GPU lane, so the CPU chain sets SAWMOD's pace. R17-3 and R17-6 matter there.
- A weaker GPU (base M, 8-10 cores) makes the GPU lane 1.6-2x longer, so the GPU items (R17-4, -5, -7, -8) pay off for SAWMOD too.
- prototype2 is GPU-bound on any Mac.
- M1's +50 ms delays double as a model of these Macs: +50 ms on SAWMOD's ~0.29 s GPU lane behaves like a GPU ~17 % slower.

### Open questions and the measurement that answers each
| # | Question | Answered by |
|---|---|---|
| Q1 | Which SAWMOD lane leads after R16, and by how much? | M1 delays |
| Q2 | What would a shorter CPU chain, or a shorter GPU lane, buy SAWMOD now? | M1 speed-up probes |
| Q3 | Are prototype2's 6-8 ms outside the GPU stages idle GPU time? | M2 |
| Q4 | How does the cold start split? | M3 cold run |
| Q5 | Where does the app path's ~2 s one-time cost go (28.0 s vs 50 x 0.518)? | M3 warm run (phase log) |
| Q6 | Are the singular and field kernels limited by occupancy (registers), loads or ALU? | M4 |

**M1: SAWMOD lanes** ([R], ~12 min, 50 freqs, 3 rounds interleaved)
```
cd ~/Desktop/Claude/Boundarylab/beat-engine-test/perf; F=BLAB_TEST_FIELD_MULTI=1
./job.sh r17_m1 "$(python3 mkjob.py 50 3 r16=$F gpu50=$F,BLAB_TEST_DELAY_GPU=0.05 fem50=$F,BLAB_TEST_DELAY_FEM=0.05 mumpsfast=$F,BLAB_TEST_MUMPS_SINGLE=1 gpufast=$F,BLAB_TEST_TR_FIRST_ONLY=1 ty8=$F,BLAB_METAL_TILEREDUCE_TY=8)" 1800 >/dev/null; grep -v '^    ' queue/r17_m1.out
```
How to read it:
- GPU slack is about 50 minus the change from `gpu50`; CPU slack is about 50 minus the change from `fem50`. The sleep frees cores, so the CPU cost reads slightly low.
- The change from `mumpsfast` is what an ~80 ms shorter CPU chain buys. The change from `gpufast` is what an ~150 ms shorter GPU lane buys.
- The probes give wrong results, so ignore their maxdB.
- Rule: **the CPU leads** if (fem50 - gpu50) >= 15 ms and mumpsfast gains >= 15 ms. **The GPU leads** in the mirror case. Anything else is **tied**.
- R15b's spread was ±7 ms, so compare the three rounds within this one job only.
- [M], optional: split MUMPS time per frequency by the previous solve's type, from `queue/r17_m1.r16.r1.rows.json` (a 5-line snippet; take the timer key from `analyze_rows.py`).

**M2: prototype2 pipeline** ([R], ~3 min; proto2q, 50 freqs, 3 rounds)
- Every config gets `BLAB_TEST_FUSED_IMAGE_ACC=1,BLAB_TEST_FUSED_PACKED=2,BLAB_TEST_FIELD_MULTI=1`.
- Configs on top: `pipe0` = `BLAB_METAL_PIPELINE=0`, `pipe1` = `=1`, `depth2` = `=1,BLAB_METAL_PIPELINE_DEPTH=2`, `solve10` = `BLAB_TEST_DELAY_EXT_SOLVE=0.01` (new hook), `r16` with `BLAB_TEST_PHASE_LOG=queue/r17_m2.phase` (records the overlap plan).
- If `solve10` costs >= 8 ms, the host solve runs serially with the GPU and R17-2 has up to 6-8 ms to gain.
- If `solve10` costs ~0 and all pipeline settings land within ±1 ms, the GPU already runs 1:1 and R17-2 is closed.

**M3: cold start** ([R] after the hook exists, ~6 min; quick.py must not be running)
Set `BLAB_TEST_COLD_LOG=<file>` as a *process* environment variable, not through `APP_TIMING_EXTRA`, so the
worker sees it while it loads. Then run `app_timing.py <project> 2 "" [50]`: run 1 is cold, run 2 warm.
- SAWMOD: add `APP_TIMING_EXTRA=BLAB_TEST_FIELD_MULTI=1`.
- proto2_quarter: add the three exterior switches.
- Package-load floor: `julia --project=src/beat_engine/julia_metal --startup-file=no -e '@time using Metal, JSON, StaticArrays, SparseArrays'`.

Report a table of wall time and host compile time (`Base.cumulative_compile_time_ns()[1]`) for each phase:
- process start to `using` done;
- `include`s done;
- worker ready;
- request setup;
- frequencies 1 and 2;
- the sum and count of Metal kernel first-launch walls.

The warm run's phase log answers Q5.

**M4: kernel probes** (in S3; [M] adds the probes, [R] runs them)
- `BLAB_TEST_PIPEINFO` on `_metal_singular_fused_bm_blocks_kernel!` (`BeatEngineMetalBurtonMiller.jl:1014`), `_metal_fast_field_kernel!` (`BeatEngineMetalFieldFast.jl:73`) and `_test_multi_field_kernel!` (:201).
- New timing-only probes: `BLAB_TEST_SING_PROBE` (1 = no store, 2 = no maths) and `BLAB_TEST_FIELD_PROBE` (1 = no cis, 2 = fixed source, so no loads).
- Run on proto2q, 12 freqs, with `BLAB_TEST_FUSED_TIMING=<file>` and `BLAB_METAL_GATHER_TIMING=1`.

---

## 2. Ranked ideas

Validation sets used below:
- **V-S**: SAWMOD, 50 freqs, 2-3 rounds.
- **V-P**: proto2q (50 freqs, 2 rounds) plus proto2 full (12 freqs).
- **V-C**: Vented_Sub and compression_driver, 12 freqs.
- **V-A**: final `app_timing.py` runs, warm, both projects.

Accuracy is maxdB against `base`, the stock app environment, under the HANDOFF policy.

### R17-1 Cold start: load the coupled engine from the precompiled bundle
- **Helps:** both projects, on every app launch.
- **Gain:** today about 59 s (SAWMOD) and 44 s (prototype2) pass before the first frequency. The bundle removes the `include` and host-JIT time (call it X, from M3), minus the pkgimage load (~1-3 s, measured in S2). For example, X = 30 s would bring SAWMOD's cold solve from 84.7 s to ~57 s and prototype2's from 59 s to ~32 s. Metal kernel compilation (Y) stays; see R17-1b.
- **Accuracy:** exact, same code. Check bit-identity.
- **Effort and risk:** medium-high, about one session.
  - Load-time side effects in `coupled_solver.jl` must move to runtime or `__init__`: the `BLAB_TEST_BLAS` forwarding at ~l.51-65 and every `ENV` read at load time.
  - No MUMPS library handles may be stored in `const`s.
  - The workload needs a tiny coupled request that solves on the CPU backend.
  - Any engine edit invalidates the cache, and the app starts its worker with a 300 s timeout (`WorkerPool.get_worker(startup_timeout_s=300.0)`, `src/beat_engine/worker.py`). So a `perf/precompile_bundle.sh` must run after every engine commit.
- **What:**
  - Extend `julia_engine/BeatEngineMetalBundle/src/BeatEngineMetalBundle.jl` with the coupled modules.
  - Turn `coupled_solver.jl` into a module file plus a thin script loader, like `solver.jl` l.36-66, including its `BLAB_BEAT_ENGINE_BUNDLE=0` fallback. The Revise harness keeps using that fallback.
  - Add a coupled CPU workload, plus `precompile` calls for the Metal host entry points that a CPU workload never reaches.
  - **Install flag:** the bundle's `Project.toml` gains deps such as `MUMPS_seq_jll` and `Serialization`, so `julia_metal` gets re-resolved. Every package is already in its manifest, so no download is expected. Ask first anyway (Q1).
- **First step:** M3.
- **Validate:**
  - V-A cold and warm runs for both projects: warm time unchanged within noise.
  - Outputs bit-identical between the bundle and `include` paths (`APP_TIMING_DUMP` pickles, `BLAB_BEAT_ENGINE_BUNDLE=0` vs `1`).
  - `quick.py --revise` still works.
- **Stop if:**
  - M3 shows `include` + host JIT under 15 s;
  - precompiling the coupled code still fails after 2 fixes; or
  - the pkgimage build takes over 300 s and can't be moved out of the app's start.
- **R17-1b (only if Y >= 10 s):**
  - First, a micro: can Julia 1.12.6 with Metal 1.10.3 compile two kernels at once on two threads?
  - If yes, `BLAB_TEST_KERNEL_PREWARM=1`: when a request arrives, compile the sweep's kernel set on a spare thread while request setup runs (MUMPS analysis, FEM assembly).
  - A quick grep of the installed GPUCompiler 2.5.0 found no on-disk cache hook, and the bundle docstring says kernel compiles "cannot be cached to disk".
- **R17-1c (app change, Q1):** pre-start the worker and run a warm-up solve when a project opens. That hides the rest of the cold start whenever the user waits about a minute before solving.

### R17-2 prototype2: close the gap between GPU work and wall time (exterior pipeline)
- **Helps:** prototype2 and every exterior-only project.
- **Gain:** at most 6-8 ms/freq (58-60 wall minus the 52.4 stage sum), i.e. up to -10-13 %, or -1.2-1.6 s per 200 freqs. M2 gives the real number.
- **Accuracy:** exact. The sweep keeps the BLAS thread count the same with or without overlap (comment at `coupled_solver.jl` ~l.1089).
- **Effort and risk:**
  - Low if a switch fixes it: app env `BLAB_METAL_PIPELINE=1` or `_DEPTH=2`, or retuned model constants (`METAL_OVERLAP_COST_SECONDS_DEFAULT` 0.003, `METAL_OVERLAP_HOST_SLOWDOWN_DEFAULT` 0.1).
  - Medium if field(i) must stop waiting for the host solve before assembly(i+2) is enqueued. The code is `produce_metal_system` and the sweep loop, `coupled_solver.jl` ~l.1055-1200.
- **First step:** M2.
- **Validate:** V-P, bit-identical, then V-A for prototype2 at 200 freqs.
- **Stop if:** `solve10` costs < 3 ms and all pipeline settings land within ±1 ms.
- Not in any "didn't work" table (no round measured these switches).

### R17-3 SAWMOD: move the solve beside MUMPS off the AMX units
- **Helps:** SAWMOD and any MUMPS project; most where the CPU chain leads (M1; stronger-GPU Macs).
- **Gain:** the 48 ms/freq contention on MUMPS (see §1).
  - Untested hypothesis: solve(i) competes with MUMPS(i+1) for the shared AMX units, since both run Accelerate zgemm/ztrsm.
  - Bandwidth is not the limit: a stale GMRES moves ~2.1 GB in ~0.2 s, about 10 GB/s.
  - If moving solve(i)'s dense work to NEON (OpenBLAS64 through a direct `ccall`; OpenBLAS_jll ships with Julia, nothing to install) halves the contention, the CPU chain loses ~24 ms. End to end that gives min(24 ms, the CPU lead measured in M1).
- **Accuracy:** order-only. The F64 residual and the backward-error stop test stay the same; only the F32 LU solves round differently. Expect well under 0.001 dB.
- **New reason vs "didn't work":** those rows moved *MUMPS* off Accelerate (hybrid, lbt forwards: slower). Here MUMPS keeps AMX and only the solve beside it moves.
- **Effort and risk:** half a session for the micro, then `BLAB_TEST_SOLVE_BLAS=openblas` in `_test_stale_gmres` (`BeatEngineCoupledCondensed.jl:583`) and the refinement residual / `FAST_TRS` path. Risk: the solve slows down on NEON. If the stale GMRES (0.208 s) grows past the FEM stage (0.295 s), it becomes critical, because elimination(i+1) waits for solve(i).
- **First step:** `perf/mumps_contention_micro.jl`, built from `mumps_loop_micro.jl` and the SAWMOD FEM dump. It times the MUMPS factorization alone and beside a GMRES-like loop (F64 3116²x3 gemm + F32 blocked trsm, 9 iterations) in three variants: (a) Accelerate, (b) OpenBLAS64 at 4 or 8 threads, (c) a threaded Julia loop with no BLAS.
- **Validate:** V-S (3 rounds) and V-C, maxdB <= 0.001.
- **Stop if:**
  - MUMPS beside (b) or (c) is not at least 25 ms faster than beside (a); or
  - M1 shows the CPU does not lead. In that case, park it as a CPU-bound-Mac item.

### R17-4 Singular kernels: load diet, one launch per pair type
- **Helps:**
  - prototype2: singular 8.8 + image singular 4.2 = 13.0 ms/freq, 22 % of the stage sum.
  - SAWMOD GPU lane: singular 12 + image 2.3 ms.
- **Gain:** the same recipe on the regular kernel (T1 step 2: float4 tables, rule constants as `Val`, 384 -> 512 threads per group) took 0.069 -> 0.058 s/freq. If the singular kernels are also register-limited, expect -25-35 %: prototype2 -3-4.5 ms/freq (-5-8 %, -0.6-0.9 s per 200 freqs) and SAWMOD GPU -3-5 ms.
- **Accuracy:** order-only.
- **What:**
  - `_metal_singular_fused_bm_blocks_kernel!` reads strided `face_vertices`, `normals` and `curls`, plus a rule of runtime length per (pair, part) via `rule_offsets`.
  - New switch `BLAB_TEST_SING_PACKED=1`: packed float4 geometry (reuse `_metal_packed_pair_tables_for`), pairs grouped by rule type (coincident / edge / vertex) once per singular cache, one launch per type with the point count as a `Val`. One quadrature body per launch respects the Metal compiler limit.
  - The coupled kernels (`_metal_singular_{slp_adjoint,dlp_hyp}_blocks_kernel!`, `BeatEngineMetalSingular.jl:450/538`) follow only if M1 says the GPU leads.
- **Effort and risk:** medium, about one session. Risk: pipeline-link crashes; keep one body per launch.
- **First step:** M4 (PIPEINFO + SING_PROBE).
- **Validate:** V-P with maxdB <= 0.001 vs stock; SAWMOD 12 freqs bit-identical (its path is untouched).
- **Stop if:** PIPEINFO already reaches >= 768 threads and the no-maths probe is < 30 % faster, or the first variant doesn't save >= 2 ms/freq on proto2q.

### R17-5 Switch on cheap existing items (after re-measuring)
- **Items:**
  - `BLAB_TEST_SING_SPLIT=0.4/0.6` (exterior). T5 measured ~2-3 ms/freq on the low-frequency part and 0.0004 dB at 0.6, which is in the "may be switched on" band.
  - `BLAB_METAL_TILEREDUCE_TY=8`: SAWMOD GPU -9 ms alone, bit-identical. New reason to retry: R16 judged it inside the noise with 1 round and before FIELD_MULTI; M1 has 3 rounds.
  - COMBINED_BM zeroing: the operator pool zeroes 4 buffers but only 2 are written (-1.5 ms GPU; from the GPU_PLAN small items).
- **Gain:** prototype2 -2-3 ms/freq. SAWMOD GPU lane -10 ms, which only counts if the GPU leads, or on GPU-weaker Macs.
- **Accuracy:** SING_SPLIT is a quadrature change (<= 0.0004 dB measured). TY8 and the zeroing are exact.
- **Effort:** low.
- **First step:** M1's `ty8` config, then a proto2q job at 200 freqs with SING_SPLIT off / 0.4 / 0.6, 2 rounds.
- **Stop if:** no gain outside the noise, or > 0.001 dB.

### R17-6 MUMPS sparse right-hand sides for the transducer reduction
- **Helps:** SAWMOD's CPU chain and any transducer project.
- **Gain:**
  - The reduction takes 0.015 s alone and 0.021 s contended per freq (R13).
  - Its 6 columns are nonzero only on the driver-coupled rows, so `ICNTL(20)=1` lets MUMPS prune the forward sweep. Estimate -10-15 ms on the FEM stage.
  - End to end: min(that, the CPU lead).
- **Accuracy:** order-only (it skips work on exact zeros).
- **What:** in `mumps_reduce` (`BeatEngineMumps.jl:625`), fill `irhs_sparse` / `irhs_ptr` / `rhs_sparse` / `nz_rhs` (fields already present in the `ZMumpsStruc` mirror, l.90-99), behind `BLAB_TEST_MUMPS_SPARSE_RHS=1`. This was "Next ideas 1" in HANDOFF and has never been tried.
- **Effort and risk:** low-medium. MUMPS might not allow `ICNTL(20)` together with `ICNTL(26)=1` and a Schur complement.
- **First step:** check the MUMPS 5.9 user guide, then run `perf/mumps_loop_micro.jl` with the switch.
- **Validate:** V-S and V-C, bit-identical or <= 0.001 dB.
- **Stop if:** the guide forbids it, or the micro saves < 5 ms.

### R17-7 Field kernel: two points per thread (gated by a probe)
- **Helps:** the prototype2 field (12.6 ms/freq) and SAWMOD's field (~40 ms, GPU lane).
- **Gain:**
  - 216.7 M (point, source) evaluations in 12.6 ms is ~17 G/s. On the 16-core GPU (~2.65 T lane-cycles/s) that is ~150 lane-cycles per evaluation, which looks mostly ALU-bound.
  - Two points per thread would only amortize the per-source loads (3 float4 per thread per source) and the loop overhead. Estimate 10-25 %: prototype2 -1.3-3 ms/freq, SAWMOD GPU -4-10 ms.
- **Accuracy:** order-only. Each point keeps its source order, so the result should be bit-identical.
- **What:** `BLAB_TEST_FIELD_PPT=2` in `_metal_fast_field_kernel!` and `_test_multi_field_kernel!`.
- **First step:** M4 (`FIELD_PROBE=2` and PIPEINFO).
- **Validate:** V-P and V-S (12 freqs), bit-identical.
- **Stop if:** the no-loads probe is < 15 % faster.

### R17-8 Field on a mirror-symmetric sphere grid (needs the app and your decision, Q2)
- **Measured in this session:** the sphere's 6600 points are a golden-spiral grid. Only ~10 % of the 7322 field points have their x or y mirror image in the set (same in both projects).
- **Gain:**
  - With a mirror-closed grid of the same density, the engine would evaluate each mirror orbit once: ~2000-2400 of 7322 points.
  - prototype2: field 12.6 -> ~3.5-4 ms, i.e. -8.5 ms/freq (-14 %, -1.7 s per 200 freqs).
  - SAWMOD: field ~40 -> ~12 ms, i.e. GPU lane -28 ms. R16 showed about half of a field saving reaches the sweep.
- **Accuracy:** exact at the evaluated points. The mirrored values are the same sums in another order (order-only). The balloon's sample positions change; its accuracy does not.
- **What:**
  - App: the sphere generator in Boundary Lab (to be located) emits a mirror-closed set when symmetry is on.
  - Engine: `BLAB_TEST_FIELD_MIRROR=1` finds the orbits once per request (a hash of rounded coordinates, as used here), evaluates one representative per orbit and scatters the results.
- **Effort:** medium; app plus engine.
- **Stop if:** you prefer the current grid.

---

## 3. Suggested order
1. **S1: measure** (cheap). *[S1 done 2026-09-27 except M3: M1 = CPU leads by ~36 ms; M2 = pipeline on saves 8 ms/freq, so R17-2 is in]* Trim HANDOFF (step 0), add 3 small hooks, then M1, M2, M3.
2. **R17-2** if M2 shows >= 3 ms. *[yes: 8 ms]* A switch-only fix goes straight into the round 17 app patch draft.
3. **R17-5** (cheap; M1 already covers TY8).
4. **R17-1** if M3 shows X >= 15 s. After it: R17-1b if Y >= 10 s, and R17-1c if you say yes.
5. By M1's verdict:

| M1 says | Next | Then |
|---|---|---|
| **CPU leads (>= 15 ms)** ← M1 result | R17-3 micro, then R17-6 | R17-4 for prototype2 only |
| GPU leads | R17-4 (both kernels), R17-8 if approved | R17-7 |
| Tied | R17-4 (prototype2), R17-8 if approved | R17-3 / R17-6 as other-Mac insurance |

6. **R17-7** last (lowest confidence). Its M4 probes run in S3 anyway.

## 4. Questions for the user
1. **Cold start.** May S2 extend `BeatEngineMetalBundle`? Its deps change, and `julia_metal` gets re-resolved from packages that are already installed. After that, a precompile script must run (a few minutes) after each engine edit, before the next app launch. Separately: may the app pre-start the worker when a project opens (R17-1c, an app change)?
2. **Sphere grid.** Should Boundary Lab use a mirror-symmetric balloon grid when a project uses symmetry (R17-8)? That saves ~14 % on prototype2 and ~28 ms of SAWMOD's GPU lane.
3. **Round 16 app patch.** Should it be applied before S1? If not, the app runs use `APP_TIMING_EXTRA`.
4. **Another Mac.** Can you do one `app_timing.py` run per project at the end on a base M or Max Mac? The answer decides how much weight the SAWMOD GPU items get.

## 5. Working method and token budget

### Sessions
Each session starts from files. Suggested opening prompt: "Read perf/HANDOFF.md (Status + Rules) and
perf/PLAN_ROUND17.md section S<n>; do S<n>."

Each session ends the same way:
- a `NOTES.md` "Round 17<x>" section (a numbers table, no logs);
- at most 10 lines of HANDOFF status;
- decision marks in this plan;
- commit, tag `metal-test-round17<x>`, push to `fork`.

| Session | Contents | Model | [R] = horn-runner (Haiku), [M] = main |
|---|---|---|---|
| S1 Measure | Step 0: trim HANDOFF. Hooks: `BLAB_TEST_DELAY_EXT_SOLVE`, overlap-plan line in `BLAB_TEST_PHASE_LOG`, `BLAB_TEST_COLD_LOG` (coupled_solver.jl phase stamps + first-launch walls in `_metal_launch`, `BeatEngineMetalCommon.jl:210`), `--request=` flag for `mkjob.py`. Then a 12-freq smoke test that the hooks change nothing when unset, then M1, M2, M3, then decisions | Sonnet | [M] hooks, smoke test, write-up. [R] M1, M2, M3 |
| S2 Cold start | R17-1 (+1b micro) | Sonnet (Opus if precompile debugging stalls) | [M] edits and precompile fixes. [R] cold/warm app runs, bit-identity compare |
| S3 prototype2 GPU | R17-2 (code part, if M2 needs it), R17-5, M4, R17-4. R17-7 only if gated in. Split at R17-4 if the session runs long | Opus for kernel edits | [M] kernels. [R] V-P jobs, 200-freq job |
| S4 SAWMOD lane | Per M1: R17-3 micro, then switch, then R17-6; or the GPU items | Sonnet | [M] micro code, switches. [R] V-S, V-C |
| S5 Wrap-up (fold into S4 if short) | `perf/app_patches/round17_engine_distribution.diff`, V-A for both projects, HANDOFF | Sonnet | [R] V-A. [M] patch, handoff |

**Delegate to horn-runner:** any step whose raw output the main session doesn't need, i.e. job runs,
app_timing runs and package-load timing. **Keep in the main session:** anything that edits, compiles or
fixes code, the micros while they are being written, and every decision.

**Runner brief template:**

Start `quick.py` once per session. Delete leftover `queue/*.job.json` files first, and never run
app_timing while quick.py has a job running.

### Output hygiene (every step)
- **Jobs:** `./job.sh <name> "<json>" <timeout> >/dev/null; grep -v '^    ' queue/<name>.out` gives one line per config. Add the section lines only when a stage split is needed (`grep -A1 '^<config>'`).
- **Failures:** use only the two greps above, never whole stack traces. **Revise compile errors:** `grep -m3 -o "ERROR: [^\\]*"` plus the BeatEngine locations.
- **Rows:** `perf/analyze_rows.py <rows.json> | head -30`. Never `cat` the JSON dumps.
- **app_timing:** pipe through `2>&1 | tail -4`.
- No `"detail": true` unless per-frequency dB is the question; then `grep -m10`.
- Don't re-read a file already read in the session. Use `grep -n` plus an offset read. Don't poll long jobs with repeated `tail`: `job.sh` blocks, so let it.
- After ~2 failed fixes of the same problem, stop and report.

### Token estimate (rough)
Processed tokens ≈ turns x average context; most of it is cache reads. The fixed start context is ~35 k
after the HANDOFF trim, vs ~40 k+ today.

**Why step 0 pays off:** HANDOFF is ~281 lines (~10 k tokens) and ~150 of those lines are history
(status after rounds 10-14, the R10-R12 budgets and surveys). Moving that history to a NOTES archive saves
~5 k tokens on every turn, i.e. ~1 M over round 17's ~200 main turns.

| Session | Main | Runner (Haiku) |
|---|---|---|
| S1 | ~25 turns x ~50 k ≈ 1.2 M | 3 briefs ≈ 0.6 M |
| S2 | ~45 x ~70 k ≈ 3 M | 2 ≈ 0.5 M |
| S3 | ~55 x ~75 k ≈ 4 M | 3 ≈ 0.7 M |
| S4 | ~40 x ~60 k ≈ 2.4 M | 2 ≈ 0.5 M |
| S5 | ~12 x ~45 k ≈ 0.5 M | 2 ≈ 0.4 M |
| **Total** | **≈ 11 M** | **≈ 2.7 M** |

For comparison, the same work in one long session (~180 turns, with the context growing to 200 k+) would
process ~20-25 M.

---

## Considered and dropped in this planning session (with the reason)
- **Real-arithmetic MUMPS (dmumps):** SAWMOD regions 1-3 have `bulk_loss_factor` 0.02, so the FEM matrix K - k²M - i·η·k²·M_b is complex (`assemble_fem_dynamic_stiffness`, `BeatEngineCoupled.jl:564`). It would only help lossless projects.
- **GPU trailing update for the F32 LU:**
  - A complex LU of n = 3116 is ~81 GFLOP.
  - The panels (~21 GFLOP) run at cgetrf's 0.3 TFLOP/s, ~70 ms. The trailing gemm (~60 GFLOP) runs at 2.0 TFLOP/s, ~30 ms.
  - So the GPU could save at most ~30 ms on fresh freqs, ≤ 18 ms/freq mean, with round 5's risk of GPU work beside the field.
- **Inner GMRES with an F32 operator:**
  - Today a stale solve moves 155 + 77 MB per iteration, ~2.1 GB for 9 iterations.
  - The F32 variant would still move ~2.1 GB: F32 conversion 0.23 + 9 x 154 MB + 3 outer F64 residuals 0.47 GB.
- **Galerkin reciprocity** (compute each pair once for both (i,j) and (j,i)): sin/cos is only 6 of 214 ms. The accumulations and tile reductions would double, with T2's occupancy risk.
- **Caching the interface-mass solve:** `_mass_block_solve` runs on the per-frequency Schur blocks (`BeatEngineCoupledCondensed.jl` ~l.1340), so nothing is frequency-invariant.
- **Field mirror (engine only, on today's grids):** only ~10 % of points have mirror partners, so it would save ~0.6 ms.
