"""Summarize BLAB_TEST_ASM_TIMING files: median per key (s per assembly call)."""
import statistics, sys
from pathlib import Path
keys = ("metal_native_regular_kernel", "metal_native_gather_pairs", "metal_native_gather_dlp_hyp",
        "metal_native_gather_slp_adjoint")
for path in sys.argv[1:]:
    rows = [dict(kv.split("=") for kv in line.split()[1:]) for line in Path(path).read_text().splitlines()]
    rows = rows[2:]  # drop the uncounted compile run
    med = {k: statistics.median(float(r[k]) for r in rows if k in r) for k in keys if any(k in r for r in rows)}
    print(f"{Path(path).stem:14s} n={len(rows):2d}  " + "  ".join(f"{k.split('_')[-1]} {v:.3f}" for k, v in med.items()))
