# GPU solve: five optimization targets (2026-09-27, from the round 15 timing study)

Data: `perf/NOTES.md` "Round 15". All times are M1 Pro, ms per frequency, measured alone on the GPU
with `BLAB_TEST_GPU_MICRO` (cost does not depend on frequency).

## Where the GPU time goes
| Stage | SAWMOD (coupled) | prototype2 quarter (exterior) |
|---|---|---|
| Pair kernel: maths | ~120 (9 evaluations per pair, 147 M pairs) | ~17 (36 evaluations per pair, 6.1 M pairs) |
| Pair kernel: in-group reduction (tile-reduce) | ~90 (4 x ~21, one per symmetry transform) | none (writes 24 floats per pair, ~5) |
| Gathers into matrices | 20 | 17 (4 lhs + 4 rhs gathers, one per transform) |
| Singular corrections | 14 | 12 |
| Zeroing / small passes | 3 | 3.5 |
| **Operators total** | **250** | **49** |
| Field | 84 (3 calls x 28, one per excitation) | 12.6 (1 call) |

What limits the kernels: occupancy (registers). The SAWMOD pair kernel compiles to 512 threads per
group, prototype2's to 384, of 1024 possible; a variant with the same work but 448 threads ran 35 %
slower. Not the limit: sin/cos (-6 ms if removed entirely), device-memory stores (-3), tile shapes,
chunk budget, singular part count.

How much end-to-end time is at stake:
- **prototype2 is GPU-bound**: removing 17 ms of pair maths took the 50-freq sweep from 73 to 57 ms
  per frequency. Every millisecond below saves a millisecond per frequency.
- **SAWMOD on this M1 Pro is not**: 2.5x faster operators moved the sweep 0.515 -> 0.496 s/freq (4 %),
  because the CPU chain (MUMPS -> LU) is now the longer lane. GPU work pays off for SAWMOD on Macs
  whose GPU is weaker relative to the CPU (base M-chips with 8-10 GPU cores, newer Pro chips with
  faster CPUs), where the GPU lane becomes the longer one again.

Accuracy labels (policy at the top of HANDOFF.md): **order-only** = same maths, different Float32
summation order, expected ~1e-7 relative (well under 0.001 dB); **quadrature** = different numerical
integration, results move, must be measured in dB and brought to the user if between 0.001 and 0.1 dB.

---

## T1. Bring the SAWMOD kernel work to the exterior (fused) path
Gain: prototype2 assembly 49 -> ~30 ms, sweep ~0.073 -> ~0.055 s/freq (about -25 %, 200 freqs
~14 s -> ~10.5 s). SAWMOD: none (different path). Accuracy: order-only.

Why: exterior projects run upstream's fused kernel, which has none of rounds 1-9: rule points and
geometry come from strided device arrays, the gathers read 36 scattered floats per matrix entry, and
they run once per symmetry transform (4 lhs + 4 rhs gathers = 17 ms for a 677 x 677 matrix).

Steps (each behind `BLAB_TEST_FUSED_*`, each measured on proto2q + full proto2 with quick.py):
1. `FUSED_IMAGE_ACC=1`: the pair kernel of transforms 2-4 adds into the pair blocks instead of
   overwriting; the rhs and lhs gathers run once after all transforms (as `IMAGE_ACCUMULATE` does for
   the coupled path). Expected gathers 17 -> ~4.5 ms. Smallest change, most of the gain.
2. `FUSED_PACKED=1`: port the load diet to `_metal_regular_pair_fused_blocks`: float4 points,
   normals and curls from `_metal_packed_pair_tables_for`, rule constants as a `Val` tuple, both
   quadrature loops unrolled (the in-loop Burton-Miller combination it already has stays). Expected
   maths 17 -> ~12 ms. Check `BLAB_TEST_PIPEINFO` stays >= 384 threads.
3. `FUSED_TILEREDUCE=1` (only if step 1 leaves the gathers > ~3 ms): reuse `MetalTileReduceTables`
   (same cache type) and the 2-phase COMB reduction; lhs via `_test_tilereduce_a_kernel!` writing the
   system matrix, rhs via a C gather that multiplies by the Neumann data per drive into `rhs_partial`.
4. Pool `rhs_partial` and the singular value buffers (`Metal.zeros` per call today, 1.4 + 0.9 ms).
Gate: proto2 quarter and full, maxdB vs stock <= 0.001 at 12 and 200 freqs; SAWMOD untouched.

## T2. Leaner tile-reduce pair kernel: combine early, sum the images in registers
Gain: SAWMOD pair kernel 214 -> ~120 ms (GPU lane 0.33 -> ~0.24 s/freq). End to end on this M1 Pro
at most ~4 % (~1 s of 24 s); on GPU-weaker Macs up to ~20 %. Also speeds up T1's kernel. Accuracy:
order-only.

Why: the reduction costs ~21 ms per transform, x4 transforms, and occupancy is capped at 512 threads.
The COMB path computes all 48 values (S, K', D, H) and combines only at the end, so 48 accumulators
are live through the quadrature loops; the fused exterior kernel shows the combination can happen
per test point (24 live values).

Steps:
1. `TR_EARLY_COMB=1`: in `_metal_packed_test_point`, form C = -S - beta*K' and the lhs vector
   u = -d + beta*(-k^2 n.n') h per test point before the outer-product expansion (as
   `_metal_regular_pair_fused_blocks` does, with beta instead of i/k), keep the curl term after the
   loop. 24 accumulators instead of 48. Measure PIPEINFO (target >= 768 threads) and maths time.
2. `TR_IMAGE_LOOP=1`: one thread loops over the 4 transforms (runtime `while` loop, signs from a
   4-entry constant table, per-transform skip test), adding into the same 24 registers, then one
   2-phase reduction instead of four. Expected reduction 90 -> ~22 ms. Round 9 tried this without
   step 1 and the pair kernel got slower (0.27 -> 0.45 s: 48 live values, unrolled bodies); stop if
   PIPEINFO drops below 512 or the maths time grows by more than the reduction saves.
3. `BLAB_METAL_TILEREDUCE_TY=8` (measured -12 ms today): re-measure on top, confirm bit-identity.
Gate: SAWMOD 12 freqs and 50 freqs, Vented_Sub, compression_driver: maxdB <= 0.001 vs current.

## T3. Field: all excitations in one pass (then, optionally, cheaper far sources)
Gain, part 1: SAWMOD field 84 -> ~35 ms (GPU lane -50 ms), bit-identical per excitation.
prototype2: none (one excitation). Part 2 (quadrature): prototype2 field 12.6 -> ~4 ms (-8 ms, ~11 %
of its sweep), SAWMOD a further ~-20 ms.

Part 1 steps: `FIELD_MULTI=1`. The coupled solver calls the field once per excitation
(`coupled_solver.jl` ~3387 and ~3454); collect the excitations' pressure and Neumann columns and call
once. In `_evaluate_galerkin_field_metal_fast`, build `weights4` per drive (drive-major) and give the
kernel `Val(NDRIVE)` accumulators: distance, Green's value and normal projection are computed once
per source and applied to every drive, each drive summing in the same order as today.
Part 2 steps: `FIELD_FAR_RULE=1`: per element, a 1-point (centroid) source when the evaluation point
is further than c x element size (c ~ 10); precompute per-element centroid sources (weights and
normals folded) next to `points4`. Branch per source element inside the loop (uniform across a SIMD
group because neighbouring threads are neighbouring points). Measure against `BLAB_TEST_FIELD_F64`,
report dB and time to the user (quadrature change).

## T4. Distance-adaptive quadrature for far element pairs
Gain: prototype2 pair maths ~17 -> ~6 ms (-11 ms, ~15 % of its sweep); SAWMOD maths ~120 -> ~60
(far pairs 1 point instead of 3). Accuracy: quadrature; must be measured and brought to the user.

Why: every pair uses the same rule (prototype2: 6 points per triangle = 36 evaluations) whether the
two triangles touch or are a metre apart. Round 12 lowered the order everywhere when k*h was small
and failed (0.27 relative) because near pairs need the full order; standard BEM practice picks the
order per pair from distance / element size (and k*h).

Steps: `TR_FAR_ORDER=d1,d2` (and the same for the fused kernel after T1). Precompute per element a
centroid and radius (float4 table). Per pair: rho = distance / (r_test + r_trial); full rule if
rho < d1, 3 points if rho < d2, 1 point beyond, and never below 3 points when k*h > ~1. Both rule
bodies unrolled inside the kernel; check PIPEINFO (a second body may cost registers; if so, split
near and far pairs into two launches over pair-block lists). Tune d1, d2 on proto2 + SAWMOD +
Vented_Sub against the stock rule: report a table of time saved vs dB for 2-3 settings and let the
user choose.

## T5. Singular corrections: precompute the frequency-independent part once per sweep
Gain: singular stage prototype2 ~12 -> ~3 ms (-9 ms, ~12 % of its sweep), SAWMOD ~14 -> ~4 ms.
Accuracy: quadrature (smooth remainder on a regular rule); likely neutral or better if the static
part is done in Float64; must be measured.

Why: the 15377 adjacent pairs (prototype2) each run a Sauter-Schwab rule (17,664 points over the four
pair types), every frequency, although only the 1/r part of the kernel is singular and that part does
not depend on frequency.

Steps: `SING_SPLIT=1`. Write G = G0 + G1 with G0 = 1/(4 pi r) and G1 = (e^{ikr} - 1)/(4 pi r)
(bounded, ~ik/4pi at r = 0; its gradient is bounded too). Once per sweep, on the GPU or CPU in Float64:
per adjacent pair, the G0 blocks S0 (3), K'0 (3), D0 (9), the basis part of H0 (9) and G0 total (for
the curl term). Store as ComplexF32 (~15k pairs x ~35 values, a few MB). Per frequency: combine them
with beta and k^2 in a small kernel, and add the G1 contribution of adjacent pairs with a regular
6-point x 6-point rule through the regular pair code (it no longer skips adjacent pairs). Keep the
image-singular pairs on the same scheme. Validate per pair against the current Sauter-Schwab result
(`scripts/validate_metal_singular_summation.jl` pattern), then in dB on proto2 and SAWMOD.

---

## Small items found on the way (free or nearly)
- `TILEREDUCE_TY=8`: -12 ms SAWMOD (confirm bit-identity).
- Under `COMBINED_BM`, the operator pool zeroes four buffers but only two are written: -1.5 ms.
- Exterior path allocates and zeroes `rhs_partial` and the singular value buffers every call: -2 ms.

## Suggested order
T1 (largest, safe, prototype2) -> T2 step 1 (shared maths for both kernels) -> T3 part 1 (safe,
SAWMOD) -> T2 step 2 -> T5 -> T4 and T3 part 2 (accuracy trades: ask the user with numbers).
