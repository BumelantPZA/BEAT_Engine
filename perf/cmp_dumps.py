"""Bit-identity of two APP_TIMING_DUMP pickles: every value outside "diagnostics" (timings, provenance)
must match exactly. Usage: cmp_dumps.py a.pkl b.pkl  ->  "IDENTICAL n=<freqs> leaves=<k>" or the first diffs."""
import pickle, sys
import numpy as np

a, b = (pickle.load(open(p, "rb")) for p in sys.argv[1:3])
diffs, leaves = [], 0


def walk(x, y, path):
    global leaves
    if isinstance(x, dict) and isinstance(y, dict):
        if x.keys() != y.keys():
            diffs.append(f"{path}: keys differ")
        for k in x.keys() & y.keys():
            if k != "diagnostics":
                walk(x[k], y[k], f"{path}.{k}")
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)) and not (x and isinstance(x[0], (int, float))):
        if len(x) != len(y):
            diffs.append(f"{path}: len {len(x)} vs {len(y)}")
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, f"{path}[{i}]")
    else:
        leaves += 1
        try:
            same = np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
        except TypeError:
            same = x == y
        if not same:
            diffs.append(path)


if len(a) != len(b):
    diffs.append(f"freq count {len(a)} vs {len(b)}")
for i, (x, y) in enumerate(zip(a, b)):
    walk(x, y, f"[{i}]")
print(f"IDENTICAL n={len(a)} leaves={leaves}" if not diffs else f"DIFF {len(diffs)}: " + "; ".join(diffs[:5]))
