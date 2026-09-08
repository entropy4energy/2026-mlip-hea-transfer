#!/usr/bin/env python3
"""Build a hash-backed index of a copied, validated H100 tensile campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


REQUIRED_FILES = (
    "run.complete.json",
    "stress_strain.dat",
    "lammps.stdout",
    "lammps.stderr",
    "log.lammps",
    "tension_final.data",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: empty plan")
    return rows


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"{path}: no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def trajectory_population(path: Path) -> tuple[int, set[int]]:
    frames = 0
    populations: set[int] = set()
    with path.open(encoding="utf-8", errors="replace") as handle:
        iterator = iter(handle)
        for line in iterator:
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                populations.add(int(next(iterator).strip()))
                frames += 1
    return frames, populations


def final_trace(path: Path) -> tuple[int, float]:
    count = 0
    final_strain = math.nan
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = [float(field) for field in line.split()]
            if len(values) != 14 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: invalid stress--strain row")
            count += 1
            final_strain = values[2]
    if count == 0 or final_strain < 0.1998:
        raise ValueError(f"{path}: incomplete stress--strain trace")
    return count, final_strain


def validate_gate(path: Path, label: str, rows_field: str, expected: int) -> dict[str, object]:
    record = load_json(path)
    if record.get("status") != "passed" or record.get("errors") not in ([], None):
        raise ValueError(f"{label} did not pass cleanly")
    if record.get(rows_field) != expected:
        raise ValueError(f"{label} does not report {expected} {rows_field}")
    return record


def build(args: argparse.Namespace) -> dict[str, object]:
    plan = args.plan.resolve()
    snapshot_root = args.snapshot_root.resolve()
    plan_rows = read_tsv(plan)
    if len(plan_rows) != 18 or {row["stage"] for row in plan_rows} != {"tension"}:
        raise ValueError("plan is not the fixed 18-row tension matrix")
    plan_hash = digest(plan)
    plan_receipt = load_json(plan.with_suffix(plan.suffix + ".receipt.json"))
    if plan_receipt.get("plan_sha256") != plan_hash:
        raise ValueError("plan receipt hash mismatch")

    execution = validate_gate(args.execution_validation.resolve(), "execution validation", "passed_rows", 18)
    analysis = load_json(args.analysis_validation.resolve())
    if analysis.get("status") != "passed" or analysis.get("errors") not in ([], None):
        raise ValueError("independent analysis validation did not pass cleanly")
    if analysis.get("replica_rows_replayed") != 18 or analysis.get("group_rows_replayed") != 6:
        raise ValueError("independent analysis replay has the wrong population")

    campaign = load_json(args.campaign.resolve())
    model = args.model.resolve()
    if digest(model) != campaign.get("model_sha256"):
        raise ValueError("model hash differs from frozen campaign model")

    index_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    for row in plan_rows:
        run_dir = snapshot_root / "runs" / "tension" / row["run_id"]
        missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
        trajectories = sorted(run_dir.glob("*.lammpstrj"))
        if missing or len(trajectories) != 1:
            raise ValueError(f"{row['run_id']}: missing={missing}, trajectories={len(trajectories)}")
        receipt_path = run_dir / "run.complete.json"
        receipt = load_json(receipt_path)
        if receipt.get("status") != "passed" or receipt.get("plan_sha256") != plan_hash:
            raise ValueError(f"{row['run_id']}: completion receipt/plan binding failed")
        stdout_text = (run_dir / "lammps.stdout").read_text(encoding="utf-8", errors="replace")
        if " to gpu 0" not in stdout_text or " to cpu" in stdout_text:
            raise ValueError(f"{row['run_id']}: required H100 execution marker missing")
        trace_rows, final_strain = final_trace(run_dir / "stress_strain.dat")
        trajectory = trajectories[0]
        frames, populations = trajectory_population(trajectory)
        expected_atoms = int(row["natoms"])
        if frames == 0 or populations != {expected_atoms}:
            raise ValueError(f"{row['run_id']}: trajectory population mismatch")
        if trajectory.stat().st_size > int(campaign["trajectory_max_bytes"]):
            raise ValueError(f"{row['run_id']}: trajectory exceeds campaign ceiling")

        hashes = {name: digest(run_dir / name) for name in REQUIRED_FILES}
        index_rows.append(
            {
                "task_id": int(row["task_id"]),
                "run_id": row["run_id"],
                "orientation": row["orientation"],
                "temperature_K": int(row["temperature_K"]),
                "chemical_seed": int(row["chemical_seed"]),
                "velocity_seed": int(row["velocity_seed"]),
                "natoms": expected_atoms,
                "strain_rate_ps_inverse": float(row["strain_rate_ps_inverse"]),
                "source_run_dir": row["output_dir"],
                "snapshot_run_dir": str(run_dir),
                "included": "true",
                "execution_device": "gpu_0",
                "input_data_sha256": row["data_sha256"],
                "model_sha256": campaign["model_sha256"],
                "run_receipt_sha256": hashes["run.complete.json"],
                "stress_strain_sha256": hashes["stress_strain.dat"],
                "trajectory_sha256": digest(trajectory),
                "lammps_stdout_sha256": hashes["lammps.stdout"],
                "lammps_stderr_sha256": hashes["lammps.stderr"],
                "log_lammps_sha256": hashes["log.lammps"],
                "final_data_sha256": hashes["tension_final.data"],
                "trace_rows": trace_rows,
                "final_engineering_strain": final_strain,
            }
        )
        trajectory_rows.append(
            {
                "task_id": int(row["task_id"]),
                "run_id": row["run_id"],
                "orientation": row["orientation"],
                "temperature_K": int(row["temperature_K"]),
                "chemical_seed": int(row["chemical_seed"]),
                "velocity_seed": int(row["velocity_seed"]),
                "natoms": expected_atoms,
                "frames": frames,
                "bytes": trajectory.stat().st_size,
                "gigabytes_decimal": trajectory.stat().st_size / 1.0e9,
                "source_path": f"{row['output_dir']}/{trajectory.name}",
                "snapshot_path": str(trajectory),
                "sha256": digest(trajectory),
                "ovito_compatible_lammps_dump": "true",
            }
        )

    index_path = snapshot_root / "tension_snapshot_index.csv"
    trajectory_path = snapshot_root / "tension_trajectory_catalog.csv"
    atomic_csv(index_path, index_rows)
    atomic_csv(trajectory_path, trajectory_rows)
    return {
        "schema_version": 1,
        "status": "passed",
        "campaign_id": campaign["campaign_id"],
        "source_campaign_root": args.source_campaign_root,
        "snapshot_root": str(snapshot_root),
        "plan": str(plan),
        "plan_sha256": plan_hash,
        "plan_receipt_sha256": digest(plan.with_suffix(plan.suffix + ".receipt.json")),
        "model": str(model),
        "model_sha256": digest(model),
        "execution_validation": str(args.execution_validation.resolve()),
        "execution_validation_sha256": digest(args.execution_validation.resolve()),
        "analysis_validation": str(args.analysis_validation.resolve()),
        "analysis_validation_sha256": digest(args.analysis_validation.resolve()),
        "runs_indexed": len(index_rows),
        "runs_included": sum(row["included"] == "true" for row in index_rows),
        "trajectory_files": len(trajectory_rows),
        "trajectory_total_bytes": sum(int(row["bytes"]) for row in trajectory_rows),
        "index": str(index_path),
        "index_sha256": digest(index_path),
        "trajectory_catalog": str(trajectory_path),
        "trajectory_catalog_sha256": digest(trajectory_path),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": digest(Path(__file__).resolve()),
        "execution_validation_record_status": execution.get("status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--source-campaign-root", required=True)
    parser.add_argument("--execution-validation", type=Path, required=True)
    parser.add_argument("--analysis-validation", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
