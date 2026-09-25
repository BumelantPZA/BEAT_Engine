"""Warm-worker A/B server for quick SAWMOD experiments.

Starts one Metal worker (compile paid once), then serves jobs dropped into
perf/queue/*.json:  {"configs": {"name": {"ENV": "value", ...}, ...},
                     "freqs": [..] (optional), "rounds": 2 (optional)}
Configs run interleaved (a b a b ...). Every env key used by any config of any job so
far is unset for configs that don't name it. Results go to perf/queue/<job>.out.
Accuracy is the max relative L2 difference against the first config.

`quick.py --revise` runs perf/dev_worker.jl instead: Julia source edits are picked up before every
solve (Revise), so the ~60 s start happens once. After changing a `struct` or a `const`, touch
perf/queue/RESTART and the worker restarts before the next job. Without --revise, restart the
server after editing Julia sources.
"""

import base64
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PERF = Path(__file__).resolve().parent
sys.path.insert(0, str(PERF.parent / "scripts"))
from benchmark_worker import outputs_of, submit, system_request  # noqa: E402
from beat_engine import EngineWorker, engine_paths  # noqa: E402

QUEUE = PERF / "queue"
SEEN_KEYS = set()  # every env key any job has set; unset unless a config names it
DEFAULT_FREQS = [round(20.0 * 1000.0 ** (i / 5), 1) for i in range(6)]  # 20 Hz - 20 kHz
SECTIONS = ("bem_operator_s", "fem_condensation_s", "bem_matrix_s", "block_assembly_s",
            "interface_elimination_s", "coupled_factorization_s", "field_s")


def decode(packed):
    raw, shape = packed
    return np.frombuffer(base64.b64decode(raw), dtype=np.complex128)


def run_config(worker, env, freqs, scratch):
    request = system_request(PERF / "sawmod.json", "metal", "float32", freqs)
    request["solver_options"]["test_env"] = env
    wall, rows = submit(worker, request, scratch)
    timings = [r.get("timings") or (r.get("diagnostics") or {}).get("timings") or {} for r in rows]
    sections = {k: sum(t.get(k, 0.0) for t in timings) / len(rows) for k in SECTIONS}
    outputs = {}
    for r in rows:
        outputs.update(outputs_of(r))
    run_config.last_timings = [{"t": r.get("timings"), "d": r.get("diagnostics")} for r in rows]
    return wall / len(freqs), sections, {k: decode(v) for k, v in outputs.items()}


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def run_job(worker, job, out):
    configs = job["configs"]
    freqs = job.get("freqs", DEFAULT_FREQS)
    rounds = job.get("rounds", 2)
    SEEN_KEYS.update(k for env in configs.values() for k in env)
    keys = sorted(SEEN_KEYS)
    walls = {name: [] for name in configs}
    secs = {name: [] for name in configs}
    reference, worst = None, {}
    lines = []
    for name, env in configs.items():  # uncounted: compiles each config's code path
        run_config(worker, {k: env.get(k) for k in keys}, [200.0, 5000.0], QUEUE / "request.json")
    for round_index in range(rounds):
        for name, env in configs.items():
            full = {k: env.get(k) for k in keys}
            per_freq, sections, outputs = run_config(worker, full, freqs, QUEUE / "request.json")
            walls[name].append(per_freq)
            if job.get("dump"):
                flat = [{} for _ in run_config.last_timings]
                for f, t in zip(flat, run_config.last_timings):
                    flatten("", t, f)
                keys_all = sorted({k for f in flat for k in f})
                med = {k: statistics.median(f.get(k, 0.0) for f in flat) for k in keys_all}
                (QUEUE / f"{out.stem}.{name}.r{round_index + 1}.timings.json").write_text(json.dumps(med, indent=1))
            secs[name].append(sections)
            if reference is None:
                reference = outputs
            err = max(np.linalg.norm(outputs[k] - reference[k]) / max(np.linalg.norm(reference[k]), 1e-30)
                      for k in reference)
            worst[name] = max(worst.get(name, 0.0), err)
            lines.append(f"round {round_index + 1} {name:14s} {per_freq:.3f} s/freq")
            out.write_text("\n".join(lines) + "\n")
    base = statistics.median(walls[next(iter(configs))])
    lines.append(f"\nfreqs={freqs} rounds={rounds}  (s/freq, median [min-max], sections are medians)")
    for name in configs:
        w = walls[name]
        med = {k: statistics.median(s[k] for s in secs[name]) for k in SECTIONS}
        sec = "  ".join(f"{k.removesuffix('_s').replace('interface_', 'if_').replace('coupled_', 'c_')} {v:.2f}"
                        for k, v in med.items())
        lines.append(f"{name:14s} {statistics.median(w):.3f} [{min(w):.3f}-{max(w):.3f}]  "
                     f"{base / statistics.median(w):.2f}x  maxrel {worst[name]:.1e}\n    {sec}")
    out.write_text("\n".join(lines) + "\n")


def main():
    # SIGTERM would otherwise kill Python without running `finally`, orphaning the Julia worker.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    QUEUE.mkdir(exist_ok=True)
    paths = engine_paths("metal")
    revise = "--revise" in sys.argv
    script, environment = paths.system_solver, None
    if revise:
        script = PERF / "dev_worker.jl"
        environment = dict(os.environ, BLAB_DEV_SOLVER_DIR=str(paths.system_solver.parent),
                           JULIA_LOAD_PATH=f"@:{PERF / 'devenv'}:@stdlib")

    def start():
        worker = EngineWorker(julia_executable="julia", solver_script=script, julia_threads="10",
                              julia_project=paths.project, environment=environment)
        started = time.perf_counter()
        worker.ensure_started()
        run_config(worker, {}, [200.0, 5000.0], QUEUE / "warmup.json")
        (QUEUE / "READY").write_text(f"warm in {time.perf_counter() - started:.1f} s (revise={revise})\n")
        print(f"worker warm in {time.perf_counter() - started:.1f} s (revise={revise})", flush=True)
        return worker

    (QUEUE / "RESTART").unlink(missing_ok=True)
    worker = start()
    try:
        while True:
            if (QUEUE / "RESTART").exists():
                (QUEUE / "RESTART").unlink()
                print("restarting worker", flush=True)
                worker.terminate()
                worker = start()
            stray = [p.name for p in QUEUE.glob("*.json") if not p.name.endswith((".job.json", ".timings.json"))
                     and p.name not in ("request.json", "warmup.json")]
            if stray and not getattr(main, "warned", False):
                print(f"ignored (jobs must be named <name>.job.json): {stray}", flush=True)
                main.warned = True
            jobs = sorted(QUEUE.glob("*.job.json"))
            if not jobs:
                time.sleep(0.5)
                continue
            job_path = jobs[0]
            out = job_path.with_name(job_path.name.replace(".job.json", ".out"))
            try:
                run_job(worker, json.loads(job_path.read_text()), out)
            except BaseException as exc:  # keep serving after a bad job
                out.write_text(f"FAILED: {exc!r}\n")
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):   # SIGTERM arrives as SystemExit
                    raise
            job_path.rename(job_path.with_suffix(".done"))
            print(f"done {job_path.name}", flush=True)
    finally:
        worker.terminate()


if __name__ == "__main__":
    main()
