#!/usr/bin/env python3
"""Launch one DeePMD job and resume from the newest ordinary checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
from pathlib import Path


CHECKPOINT_RE = re.compile(r"model\.ckpt-(\d+)\.pt$")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(workdir: Path) -> Path | None:
    candidates: list[Path] = []
    latest = workdir / "model.ckpt.pt"
    if latest.is_file():
        candidates.append(latest.resolve())
    pointer = workdir / "checkpoint"
    if pointer.is_file():
        pointed = pointer.read_text(errors="replace").strip()
        if pointed:
            pointed_path = Path(pointed)
            if not pointed_path.is_absolute():
                pointed_path = workdir / pointed_path
            if pointed_path.is_file():
                candidates.append(pointed_path.resolve())
    candidates.extend(
        candidate.resolve()
        for candidate in workdir.glob("**/model.ckpt-*.pt")
        if candidate.is_file() and "model_ema.ckpt" not in candidate.name
    )
    unique = {path for path in candidates if checkpoint_step(path) >= 0}
    return max(unique, key=checkpoint_step) if unique else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("input.json"))
    parser.add_argument("--logdir", type=Path, default=Path("logs"))
    parser.add_argument("--restart", choices=("auto", "off"), default="auto")
    args = parser.parse_args()

    workdir = Path.cwd().resolve()
    input_path = (workdir / args.input).resolve()
    logdir = (workdir / args.logdir).resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if len(visible) != 1:
        raise RuntimeError(f"Expected one visible GPU, found {visible}")
    checkpoint = resolve_checkpoint(workdir) if args.restart == "auto" else None
    command = [
        "dp",
        "--pt",
        "train",
        "-l",
        str(logdir / "train.deepmd.log"),
        "-o",
        "normalized_input.json",
        str(input_path),
    ]
    if checkpoint is not None:
        command.extend(["--restart", str(checkpoint)])

    status_path = workdir / "run_status.json"
    payload = {
        "state": "starting",
        "launcher_pid": os.getpid(),
        "host": os.uname().nodename,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cuda_visible_devices": visible,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "command": command,
    }
    atomic_json(status_path, payload)
    print("command: " + " ".join(command), flush=True)
    child = subprocess.Popen(command, cwd=workdir, env=os.environ.copy())
    payload.update({"state": "running", "training_pid": child.pid})
    atomic_json(status_path, payload)

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signal.SIGTERM if signum == signal.SIGHUP else signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, forward)
    return_code = child.wait()
    payload.update(
        {
            "state": "completed" if return_code == 0 else "stopped_or_failed",
            "return_code": return_code,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "latest_checkpoint": str(resolve_checkpoint(workdir) or ""),
        }
    )
    atomic_json(status_path, payload)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
