"""Pivotal K12+rewind wrapper around the canonical T4 harness.

Loads scripts/run_w_sweep_freeze_safe.py UNMODIFIED from the same directory and
monkey-patches phase2_cmd to inject extra phase-2-only env from the
PIVOTAL_P2_ENV JSON env var (phase 1 stays env-clean). "{rundir}" inside a
value expands to the per-run directory (per-run DS4_PACE_LOG paths).
Writes p2env.json next to the harness manifest for provenance.
"""
import importlib.util
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rw = _load("run_w_sweep_freeze_safe")

P2_ENV = json.loads(os.environ.get("PIVOTAL_P2_ENV", "{}"))

_orig_phase2_cmd = rw.phase2_cmd


def phase2_cmd(args, w, p2prompt_file, mask_file, seed):
    env, cmd = _orig_phase2_cmd(args, w, p2prompt_file, mask_file, seed)
    rundir = str(pathlib.Path(str(p2prompt_file)).resolve().parent)
    for k, v in P2_ENV.items():
        env[str(k)] = str(v).replace("{rundir}", rundir)
    return env, cmd


rw.phase2_cmd = phase2_cmd


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rc = rw.main(argv)
    # provenance: record the injected phase-2 env next to the manifest
    try:
        args = rw.parse_args(argv)
        out = pathlib.Path(args.outdir)
        if out.exists():
            (out / "p2env.json").write_text(
                json.dumps(P2_ENV, indent=2), encoding="utf-8")
    except Exception:
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
