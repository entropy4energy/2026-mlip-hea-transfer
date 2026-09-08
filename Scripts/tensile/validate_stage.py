#!/usr/bin/env python3
"""Reconstruct execution/completeness checks for one H100 stage."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from common import atomic_json, atomic_tsv, load_json, read_tsv, sha256


FINAL_DATA = {"relax": "relaxed.data", "equilibrate": "equilibrated.data", "nve": "nve_final.data", "tension": "tension_final.data"}
TRACE = {"equilibrate": "equilibration_samples.dat", "nve": "nve_energy_samples.dat", "tension": "stress_strain.dat"}
RELAX_FORCE_LIMIT_EV_PER_A = 0.02
RELAX_NORMAL_STRESS_LIMIT_GPA = 0.10


def atom_count(path: Path) -> int:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:30]:
        if line.strip().endswith(" atoms"):
            return int(line.split()[0])
    raise ValueError(f"{path}: no atom count")


def numeric_rows(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(field) for field in line.split()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}: non-finite trace value")
        rows.append(values)
    if not rows:
        raise ValueError(f"{path}: empty trace")
    return rows


def dump_populations(path: Path) -> tuple[int, set[int]]:
    frames = 0
    populations: set[int] = set()
    with path.open(encoding="utf-8", errors="replace") as handle:
        iterator = iter(handle)
        for line in iterator:
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                populations.add(int(next(iterator).strip()))
                frames += 1
    return frames, populations


def validate_row(row: dict[str, str], plan_hash: str, trajectory_limit: int) -> dict[str, object]:
    root = Path(row["output_dir"])
    errors: list[str] = []
    receipt_path = root / "run.complete.json"
    try:
        receipt = load_json(receipt_path)
        if receipt.get("status") != "passed" or receipt.get("plan_sha256") != plan_hash:
            errors.append("completion receipt status/plan binding failed")
        failure_path = root / "run.failed.json"
        if failure_path.is_file():
            if receipt.get("completion_basis") != "postcondition_only_repair_A5":
                errors.append("failure/completion coexistence lacks the A5 repair basis")
            for path_key, hash_key in (
                ("original_failure", "original_failure_sha256"),
                ("source_restart", "source_restart_sha256"),
                ("amendment", "amendment_sha256"),
                ("implementation_amendment", "implementation_amendment_sha256"),
                ("manifest", "manifest_sha256"),
            ):
                bound = Path(str(receipt.get(path_key, "")))
                if not bound.is_file() or receipt.get(hash_key) != sha256(bound):
                    errors.append(f"A5 repair binding failed: {path_key}")
        for output in receipt.get("outputs", []):
            path = Path(output["path"])
            if not path.is_file() or sha256(path) != output["sha256"]:
                errors.append(f"output hash failed: {path}")
    except Exception as error:  # surfaced in the durable result
        receipt = {}
        errors.append(f"completion receipt unreadable: {error}")
    final_data = root / FINAL_DATA[row["stage"]]
    if not final_data.is_file():
        errors.append("final data missing")
    else:
        try:
            if atom_count(final_data) != int(row["natoms"]):
                errors.append("final atom population differs from plan")
        except Exception as error:
            errors.append(str(error))
    trace_rows = 0
    final_strain: float | str = ""
    device_execution = "not_applicable"
    relax_max_force: float | str = ""
    relax_max_normal_stress: float | str = ""
    if row["stage"] == "relax":
        try:
            with (root / "relax_summary.csv").open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            if len(summary_rows) != 1:
                raise ValueError("relaxation summary does not contain exactly one row")
            summary = summary_rows[0]
            relax_max_force = float(summary["max_force_eV_per_A"])
            relax_max_normal_stress = float(summary["max_normal_stress_error_GPa"])
            values = [relax_max_force, relax_max_normal_stress]
            if not all(math.isfinite(value) for value in values):
                errors.append("relaxation summary contains a non-finite gate value")
            if relax_max_force > RELAX_FORCE_LIMIT_EV_PER_A:
                errors.append("relaxation maximum force exceeds 0.02 eV/A")
            if relax_max_normal_stress > RELAX_NORMAL_STRESS_LIMIT_GPA:
                errors.append("relaxation normal-stress error exceeds 0.10 GPa")
            if int(summary["natoms"]) != int(row["natoms"]) or summary["orientation"] != row["orientation"]:
                errors.append("relaxation summary population/orientation differs from plan")
        except Exception as error:
            errors.append(f"relaxation gate reconstruction failed: {error}")
    if row["stage"] in TRACE:
        try:
            values = numeric_rows(root / TRACE[row["stage"]])
            trace_rows = len(values)
            if row["stage"] == "equilibrate" and values[-1][1] < 39.99:
                errors.append("equilibration trace ends before 40 ps")
            if row["stage"] == "nve" and values[-1][1] < 9.99:
                errors.append("NVE trace ends before 10 ps")
            if row["stage"] == "tension":
                strains = [value[2] for value in values]
                final_strain = strains[-1]
                if final_strain < 0.1998:
                    errors.append("tension trace did not reach 20% strain")
                if any(b + 1.0e-10 < a for a, b in zip(strains, strains[1:])):
                    errors.append("engineering strain is not monotonic")
        except Exception as error:
            errors.append(f"trace validation failed: {error}")
    if row["stage"] == "tension":
        stdout_path = root / "lammps.stdout"
        if not stdout_path.is_file():
            device_execution = "missing"
            errors.append("tension LAMMPS stdout is missing")
        else:
            stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
            if " to gpu 0" in stdout_text:
                device_execution = "gpu_0"
            elif " to cpu" in stdout_text:
                device_execution = "cpu_fallback"
                errors.append("tension model executed on CPU instead of the required H100 GPU")
            else:
                device_execution = "unknown"
                errors.append("tension GPU execution marker is missing")
    trajectory_bytes = 0
    trajectory_frames = 0
    trajectory_populations: set[int] = set()
    for trajectory in root.glob("*.lammpstrj"):
        trajectory_bytes += trajectory.stat().st_size
        frames, populations = dump_populations(trajectory)
        trajectory_frames += frames
        trajectory_populations.update(populations)
        if trajectory.stat().st_size > trajectory_limit:
            errors.append(f"trajectory exceeds hard ceiling: {trajectory}")
    if row["stage"] != "relax":
        if trajectory_frames == 0:
            errors.append("trajectory has no frames")
        if trajectory_populations != {int(row["natoms"])}:
            errors.append("trajectory atom population is inconsistent")
    return {
        "task_id": int(row["task_id"]), "run_id": row["run_id"],
        "status": "passed" if not errors else "failed", "errors": "; ".join(errors),
        "relax_max_force_eV_per_A": relax_max_force,
        "relax_max_normal_stress_error_GPa": relax_max_normal_stress,
        "trace_rows": trace_rows, "final_strain": final_strain,
        "device_execution": device_execution,
        "trajectory_frames": trajectory_frames, "trajectory_bytes": trajectory_bytes,
        "receipt": str(receipt_path), "receipt_sha256": sha256(receipt_path) if receipt_path.is_file() else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("relax", "equilibrate", "nve", "tension"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = args.plan.resolve()
    rows = read_tsv(plan)
    if {row["stage"] for row in rows} != {args.stage}:
        raise ValueError("plan stage mismatch")
    plan_receipt = load_json(plan.with_suffix(plan.suffix + ".receipt.json"))
    plan_hash = sha256(plan)
    if plan_receipt.get("plan_sha256") != plan_hash:
        raise ValueError("plan receipt hash mismatch")
    campaign = load_json(Path(__file__).resolve().parents[1] / "config" / "campaign.json")
    results = [validate_row(row, plan_hash, int(campaign["trajectory_max_bytes"])) for row in rows]
    failed = sum(row["status"] != "passed" for row in results)
    summary = args.output.resolve().with_suffix(".summary.tsv")
    atomic_tsv(summary, results)
    payload = {
        "schema_version": 1, "status": "passed" if failed == 0 else "failed",
        "stage": args.stage, "plan": str(plan), "plan_sha256": plan_hash,
        "planned_rows": len(rows), "validated_rows": len(results),
        "passed_rows": len(results) - failed, "failed_rows": failed,
        "summary": str(summary), "summary_sha256": sha256(summary),
        "validator": str(Path(__file__).resolve()), "validator_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
