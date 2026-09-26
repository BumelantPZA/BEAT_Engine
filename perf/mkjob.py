# usage: perf/mkjob.py <12|50> <rounds> NAME=K=V,K=V ...   (prints a job json; "base" is the app env)
import json, re, sys
src = open("/Users/aleksanderspitalniak/Desktop/Claude/Boundarylab/boundary-lab/src/blab/solvers/engine_distribution.py").read()
base = dict(re.findall(r'"(BLAB_[A-Z_]+)": "([^"]*)"', src))
freqs = [20.0,37.5,70.2,131.6,246.6,462.0,865.8,1622.3,3039.8,5696.1,10673.4,20000.0] if sys.argv[1] == "12" else \
        [round(20*1000**(i/49), 2) for i in range(50)]
configs = {"base": base}
for arg in sys.argv[3:]:
    name, _, kv = arg.partition("=")
    c = dict(base)
    for pair in filter(None, kv.split(",")):
        k, _, v = pair.partition("=")
        c[k] = v
    configs[name] = c
print(json.dumps({"freqs": freqs, "rounds": int(sys.argv[2]), "dump": True, "configs": configs}))
