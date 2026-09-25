"""Host-only check for pair_tilereduce: slots per 16-element test tile under a few orderings."""
import sys
import numpy as np
rows = np.loadtxt(sys.argv[1], skiprows=1, dtype=np.int64)
dofs = rows[:, 1:4]
n = len(dofs)
TX = int(sys.argv[2]) if len(sys.argv) > 2 else 16

def report(name, order):
    d = dofs[order]
    slots = [len(np.unique(d[t:t + TX])) for t in range(0, n, TX)]
    tiles = len(slots)
    smax = max(slots)
    print(f"{name:12s} tiles={tiles} S_max={smax} mean={np.mean(slots):.1f} p99={np.percentile(slots, 99):.0f} "
          f"slot_total(padded)={tiles * smax} ({tiles * smax / n:.2f}x) unpadded={sum(slots)} ({sum(slots) / n:.2f}x)")

report("as_is", np.arange(n))
report("min_node", np.lexsort((np.arange(n), dofs.min(1))))
report("sorted_all", np.lexsort((np.sort(dofs, 1)[:, 2], np.sort(dofs, 1)[:, 1], np.sort(dofs, 1)[:, 0])))
