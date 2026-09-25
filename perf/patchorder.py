"""Greedy compact-patch element ordering: grow each 16-element patch by picking the
unassigned element sharing the most nodes with the patch (ties: lowest min node)."""
import sys, heapq
import numpy as np
from collections import defaultdict
rows = np.loadtxt(sys.argv[1], skiprows=1, dtype=np.int64)
d = rows[:, 1:4]; n = len(d); TX = int(sys.argv[2]) if len(sys.argv) > 2 else 16
node_elems = defaultdict(list)
for e in range(n):
    for v in d[e]: node_elems[v].append(e)
assigned = np.zeros(n, bool); order = []
seed_order = list(np.lexsort((np.arange(n), d.min(1))))
seed_ptr = 0
last_patch_nodes = set()
while len(order) < n:
    # seed: unassigned element touching the previous patch if possible, else lowest min node
    seed = None
    for v in sorted(last_patch_nodes):
        for e in node_elems[v]:
            if not assigned[e]: seed = e; break
        if seed is not None: break
    if seed is None:
        while assigned[seed_order[seed_ptr]]: seed_ptr += 1
        seed = seed_order[seed_ptr]
    patch = [seed]; assigned[seed] = True; nodes = set(d[seed])
    score = defaultdict(int)
    for v in d[seed]:
        for e in node_elems[v]:
            if not assigned[e]: score[e] += 1
    while len(patch) < TX and score:
        best = max(score, key=lambda e: (score[e], -d[e].min(), -e))
        del score[best]; patch.append(best); assigned[best] = True
        for v in d[best]:
            if v not in nodes:
                nodes.add(v)
                for e in node_elems[v]:
                    if not assigned[e]: score[e] += 1
    order.extend(patch); last_patch_nodes = nodes
order = np.array(order)
dd = d[order]
slots = [len(np.unique(dd[t:t+TX])) for t in range(0, n, TX)]
print(f"patch TX={TX} S_max={max(slots)} mean={np.mean(slots):.1f} unpadded={sum(slots)} ({sum(slots)/n:.2f}x)")
nodes = [len(np.unique(dd[s:s+440])) for s in range(0, n, 440)]
print(f"chunk(440) nodes mean={np.mean(nodes):.0f}")
