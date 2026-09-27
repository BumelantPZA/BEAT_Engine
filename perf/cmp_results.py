# usage: perf/cmp_results.py a.jsonl b.jsonl
# Compares two BLAB_TEST_SAVE_RESULTS files (one result per line; the last result per frequency wins)
# with quick.py's rules: max level difference in dB within 60 dB of each peak, max relative error,
# and whether every output is bit-identical.
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_worker import outputs_of  # noqa: E402
from quick import db_error, decode  # noqa: E402


def load(path):
    by_freq = {}
    for line in open(path):
        result = json.loads(line)
        freq = float(result["freq_hz"])
        by_freq[freq] = {k.partition("@")[0]: decode(v) for k, v in outputs_of(result).items()}
    return by_freq


a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b))
worst_db = worst_rel = 0.0
identical = True
for f in common:
    for k in a[f]:
        x, y = a[f][k], b[f][k]
        identical &= np.array_equal(x, y)
        worst_rel = max(worst_rel, float(np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-30)))
        worst_db = max(worst_db, db_error(x, y))
print(f"freqs {len(common)} (only a {len(set(a) - set(b))}, only b {len(set(b) - set(a))})  "
      f"identical {identical}  maxrel {worst_rel:.1e}  maxdB {worst_db:.4f}")
