# usage: perf/analyze_rows.py queue/<job>.<config>.r1.rows.json [more ...]
# Mean of every timer, and a per-frequency table of the main stages.
import json, sys
for path in sys.argv[1:]:
    r = json.load(open(path)); rows = r["rows"]
    T = lambda d, k: d.get("d.timings." + k, 0.0) or 0.0
    n = len(rows)
    print(f"== {path}  wall {r['wall_s']:.2f} s  ({r['wall_s']/n:.3f} s/freq, {n} freqs)")
    keys = sorted({k[10:] for d in rows for k in d if k.startswith("d.timings.")})
    means = {k: sum(T(d, k) for d in rows) / n for k in keys}
    for k in sorted(keys, key=lambda k: -means[k]):
        if means[k] >= 0.002: print(f"  {k:48s} {means[k]:.4f}")
    cols = ["fem_condensation_s", "interface_elim_diag_fem_stage_work_s", "fem_condensation_factorization_s",
            "bem_operator_s", "interface_elimination_s", "coupled_factorization_s", "solve_s", "field_s", "test_pre_emit_wall_s"]
    short = ["cond", "femwork", "mumps", "bemop", "elim", "fact", "solve", "field", "pre_emit"]
    print("  hz      " + " ".join(f"{s:>8s}" for s in short) + "  refit  iterwall(next row)")
    for i, d in enumerate(rows):
        nxt = T(rows[i + 1], "test_prev_iteration_wall_s") if i + 1 < n else float("nan")
        hz = d.get("hz", i)
        print(f"  {i:3d}    " + " ".join(f"{T(d, c):8.3f}" for c in cols) +
              f"  {d.get('d.dense_refinement_iterations', '')!s:>4}  {nxt:.3f}")
