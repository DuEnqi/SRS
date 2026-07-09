#!/usr/bin/env python3
"""Run v9 extension LLM experiments sequentially with progress logging."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path(r"c:\Users\Enqi Du\Documents\Downloads\STALE\runs\ext_v9_llm_n100")
LOG = OUT / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOCK = OUT / ".running.lock"
ENV = Path(r"c:\Users\Enqi Du\Documents\Downloads\STALE\STALE-main\STALE\.env")
N_FAMILY = 100
N_CLASS = 100

STEPS = [
    ("W1 retrieval-fairness", "--retrieval-fairness", "retrieval_fairness.json"),
    ("W2 op-f1", "--op-f1", "op_f1.json"),
    ("W4 baselines", "--baselines", "baselines.json"),
]


def load_env() -> None:
    if not ENV.is_file():
        return
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def log(msg: str, fh) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    fh.write(line)
    fh.flush()
    print(line, end="")


def run_step(label: str, flag: str, artifact: str, common: list[str], fh) -> int:
    out_file = OUT / artifact
    if out_file.is_file() and out_file.stat().st_size > 100:
        log(f"SKIP {label}: {artifact} already exists", fh)
        return 0
    args = [flag, *common]
    log(f"START {label}: {' '.join(args)}", fh)
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, "-u", "stale_experiments_v9ext.py", *args],
        cwd=ROOT,
        env=os.environ.copy(),
    )
    dt = time.time() - t0
    log(f"DONE  {label}: exit={p.returncode} elapsed={dt/60:.1f}min", fh)
    if p.returncode == 0 and not out_file.is_file():
        log(f"WARN  {label}: exit 0 but {artifact} missing", fh)
    return p.returncode


def main() -> int:
    load_env()
    os.environ["PYTHONUNBUFFERED"] = "1"
    OUT.mkdir(parents=True, exist_ok=True)

    if LOCK.exists():
        try:
            pid = int(LOCK.read_text(encoding="utf-8").strip())
            # Windows: os.kill(pid, 0) doesn't work; just warn if lock is fresh
            age = time.time() - LOCK.stat().st_mtime
            if age < 3600:
                print(f"Another run may be active (lock pid={pid}, age={age:.0f}s). "
                      f"Delete {LOCK} if stale.")
                return 1
        except (ValueError, OSError):
            pass

    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    common = [
        "--use-llm", "--n-per-family", str(N_FAMILY),
        "--n-per-class", str(N_CLASS), "--out", str(OUT),
    ]
    rc = 0
    try:
        with LOG.open("w", encoding="utf-8") as fh:
            log(f"output dir: {OUT}", fh)
            log(f"log file: {LOG}", fh)
            log(f"pid: {os.getpid()}", fh)
            for label, flag, artifact in STEPS:
                rc = run_step(label, flag, artifact, common, fh)
                if rc != 0:
                    log(f"ABORT after {label}", fh)
                    return rc
            log("ALL STEPS COMPLETE", fh)
    finally:
        LOCK.unlink(missing_ok=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
