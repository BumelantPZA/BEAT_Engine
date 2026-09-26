"""In-app overhead: run a project through the app's headless path on the Metal test backend,
N times on one persistent worker, with a timestamp per frequency.
Usage (from ../boundary-lab): .venv/bin/python ../beat-engine-test/perf/app_timing.py <project.blab.json> [runs=2] [threads=8] [freq_count]
"""
import os, sys, time, tempfile
BACKEND = os.environ.get("APP_TIMING_BACKEND", "beat_metal_test")
from blab.headless import load_headless_solve_spec, HeadlessSolveSpec, load_headless_project, prepare_headless_solve, run_headless_solve
from blab.solvers.coupled_backend import PhysicalSystemProductionBackend

path = sys.argv[1]; runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
threads = sys.argv[3] if len(sys.argv) > 3 else None
nfreq = int(sys.argv[4]) if len(sys.argv) > 4 else None
t0 = time.perf_counter()
project = load_headless_project(path)
if nfreq:
    import dataclasses
    project = dataclasses.replace(project, preferences=dataclasses.replace(project.preferences, freq_count=nfreq))
t1 = time.perf_counter()
if os.environ.get("APP_TIMING_SYMMETRY"):
    import dataclasses
    project = dataclasses.replace(project, symmetry=os.environ["APP_TIMING_SYMMETRY"])
prepared = prepare_headless_solve(project, load_headless_solve_spec(None), backend_id=BACKEND)
t2 = time.perf_counter()
print(f"load {t1-t0:.2f} s  prepare {t2-t1:.2f} s  symmetry {project.symmetry}  meshes",
      [getattr(m, "file", None) or getattr(m, "name", "") for m in prepared.request.compiled_system.meshes], flush=True)
import json
import blab.solvers.coupled_backend as cb
RAW = []
_orig = cb.system_frequency_result_from_dict
def _capture(raw):
    RAW.append((time.perf_counter(), raw.get("diagnostics", {}).get("timings", {})))
    return _orig(raw)
cb.system_frequency_result_from_dict = _capture
backend = PhysicalSystemProductionBackend(bem_backend=BACKEND, julia_threads=threads)
for run in range(runs):
    marks = []
    def emit(e, marks=marks):
        marks.append((time.perf_counter(), e.get("event"), e.get("message", "")))
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        out = tmp + "/out"
        run_headless_solve(project, prepared, output_dir=out, backend_id=BACKEND,
                           public_request={}, backend=backend, event_callback=emit)
        done = time.perf_counter()
    freq = [m[0] for m in marks if m[1] == "frequency_completed"]
    first = freq[0] - start if freq else float("nan")
    steady = (freq[-1] - freq[1]) / (len(freq) - 2) if len(freq) > 2 else float("nan")
    print(f"run {run+1}: total {done-start:.2f} s  to first freq {first:.2f} s  "
          f"steady {steady:.3f} s/freq  after last {done-freq[-1]:.2f} s  n={len(freq)}", flush=True)
    json.dump([{"t": t, **tm} for t, tm in RAW], open(f"/private/tmp/claude-501/-Users-aleksanderspitalniak-boundary-lab/4b9c8f71-9ea4-4b86-b7b8-1b0f0a794056/scratchpad/app_rows{run+1}.json", "w"))
    RAW.clear()
    for m in marks[:0]:
        if m[1] == "status": print(f"   {m[0]-start:7.2f} status {m[2][:90]}")
