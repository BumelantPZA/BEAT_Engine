import json,sys,statistics
for p in sys.argv[1:]:
    r=json.load(open(p)); fr=r["frequencies"]
    keys=[]
    for f in fr:
        keys+= [k for k in f["timings"] if k not in keys]
    tot={k:sum(f["timings"].get(k,0) for f in fr) for k in keys}
    print(f"{p}: wall {r.get('wall_s',0):.1f}s  n={len(fr)} startup {r.get('startup_s',0):.1f} warmup {r.get('warmup_s',0):.1f} peak {r.get('peak_mb')}")
    for k,v in sorted(tot.items(),key=lambda x:-x[1])[:18]: print(f"   {k:45s} {v:8.2f}")
