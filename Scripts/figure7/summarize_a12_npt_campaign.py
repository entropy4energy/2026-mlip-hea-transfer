#!/usr/bin/env python3
"""Reduce the exact A12 600/900 K dependency ledger without available-case means."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import summarize_a5_npt_campaign as a5
import summarize_npt_campaign as raw


A12_PROTOCOL_SHA256 = "f73b3118f907a02d187ec43c0ad03c1878c1dabf2de42ebd8f8fb78c1e8b918f"
EXPECTED_ROWS = 48
EXPECTED_BRANCHES = {
    ("DPA2__bcc_parent", "bcc"),
    ("DPA3__bcc_parent", "bcc"),
    ("DPA4__bcc_parent", "bcc"),
    ("DPA4__fcc_parent", "fcc"),
}
CHEMICAL_SEEDS = {20260825, 20262843, 20264861}
VELOCITY_SEEDS = {20260901, 20260903}
TEMPERATURES = {600, 900}
_READ_NUMERIC = raw.read_numeric


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def analysis_arrays(path: Path, columns: tuple[str, ...]):
    """Remove only the LAMMPS fix-ave/time step-zero row, when present."""

    values = _READ_NUMERIC(path, columns)
    steps, times = values.get("step"), values.get("time_ps")
    if steps is not None and times is not None and steps.size and steps[0] == 0.0 and times[0] == 0.0:
        return {name: values_array[1:] for name, values_array in values.items()}
    return values


def atomic_csv(path: Path, values: list[dict[str, object]]) -> None:
    if not values:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(values); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def ledger_key(row: dict[str, str]) -> tuple[str, str, int, int, int]:
    return (
        row["model_id"], row["simulated_phase"], int(row["chemical_seed"]),
        int(row["velocity_seed"]), int(float(row["temperature_K"])),
    )


def blocked_replica(ledger: dict[str, str]) -> dict[str, object]:
    return {
        "task_id": ledger["equilibration_task_id"],
        "runnable_task_id": ledger["production_task_id"],
        "model_id": ledger["model_id"],
        "architecture": ledger["architecture"],
        "training_parent": ledger["training_parent"],
        "simulated_phase": ledger["simulated_phase"],
        "chemical_seed": int(ledger["chemical_seed"]),
        "velocity_seed": int(ledger["velocity_seed"]),
        "target_temperature_K": int(float(ledger["temperature_K"])),
        "natoms": int(ledger["natoms"]),
        "dependency_status": ledger["dependency_status"],
        "endpoint_status": "blocked_stationarity_or_execution",
        "blocking_reason": ledger["blocking_reason"],
        "cna_scientific_status": a5.CNA_SCIENTIFIC_STATUS,
        "fixed_cutoff_cna_used_for_endpoint_eligibility": "false",
        "endpoint_eligibility_basis": a5.CNA_ELIGIBILITY_BASIS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-validation", type=Path, required=True)
    parser.add_argument("--dependency-ledger", type=Path, required=True)
    parser.add_argument("--replicas", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    validation_path = args.stage_validation.resolve()
    ledger_path = args.dependency_ledger.resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    plan_path = Path(str(validation["plan"])).resolve()
    summary_path = Path(str(validation["summary_csv"])).resolve()
    if validation.get("plan_sha256") != sha256(plan_path) or validation.get("summary_csv_sha256") != sha256(summary_path):
        raise ValueError("A12 production validation hash chain failed")
    if validation.get("planned_tasks") != validation.get("validated_tasks"):
        raise ValueError("A12 production validation did not inspect its complete executable plan")
    plan_rows = rows(plan_path, "\t")
    summary_rows = rows(summary_path)
    plans = {row["task_id"]: row for row in plan_rows}
    stages = {row["task_id"]: row for row in summary_rows}
    if len(plans) != len(plan_rows) or set(plans) != set(stages) or any(row["calculation"] != "npt" for row in plan_rows):
        raise ValueError("A12 production plan/validation population is invalid")
    ledger_rows = rows(ledger_path, "\t")
    ledger = {ledger_key(row): row for row in ledger_rows}
    expected = {
        (model, phase, chemical, velocity, temperature)
        for model, phase in EXPECTED_BRANCHES
        for chemical in CHEMICAL_SEEDS
        for velocity in VELOCITY_SEEDS
        for temperature in TEMPERATURES
    }
    if len(ledger_rows) != EXPECTED_ROWS or len(ledger) != EXPECTED_ROWS or set(ledger) != expected:
        raise ValueError("A12 dependency ledger is not the exact 48-row population")
    production_ids = {row["production_task_id"] for row in ledger_rows if row["production_task_id"]}
    if production_ids != set(plans):
        raise ValueError("A12 dependency ledger and production plan differ")
    for plan in plan_rows:
        protocol = Path(plan["protocol_path"])
        if sha256(protocol) != A12_PROTOCOL_SHA256:
            raise ValueError(f"A12 protocol hash mismatch: {plan['task_id']}")

    raw.read_numeric = analysis_arrays
    replicas: list[dict[str, object]] = []
    for key in sorted(ledger):
        item = ledger[key]
        task_id = item["production_task_id"]
        if not task_id:
            replicas.append(blocked_replica(item))
            continue
        try:
            result = raw.analyze_replica(plans[task_id], stages[task_id])
            result.update(
                {
                    "task_id": item["equilibration_task_id"],
                    "runnable_task_id": task_id,
                    "dependency_status": "runnable_pending_execution",
                    "cna_scientific_status": a5.CNA_SCIENTIFIC_STATUS,
                    "fixed_cutoff_cna_used_for_endpoint_eligibility": "false",
                    "endpoint_eligibility_basis": a5.CNA_ELIGIBILITY_BASIS,
                }
            )
            replicas.append(result)
        except (KeyError, OSError, ValueError) as error:
            result = blocked_replica(item)
            result["endpoint_status"] = "blocked_analysis_error"
            result["blocking_reason"] = str(error)
            replicas.append(result)
    groups = a5.build_groups(replicas)
    if len(replicas) != 48 or len(groups) != 8:
        raise ValueError("A12 reducer output population is incorrect")
    replicas_path = args.replicas.resolve(); groups_path = args.groups.resolve(); receipt_path = args.receipt.resolve()
    for path in (replicas_path, groups_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    atomic_csv(replicas_path, replicas); atomic_csv(groups_path, groups)
    replica_counts = Counter(str(row["endpoint_status"]) for row in replicas)
    group_counts = Counter(str(row["status"]) for row in groups)
    receipt = {
        "schema_version": 1, "status": "passed",
        "scientific_status": "complete" if group_counts == {"eligible": 8} else "partial_with_explicit_blocks",
        "stage_validation": str(validation_path), "stage_validation_sha256": sha256(validation_path),
        "plan": str(plan_path), "plan_sha256": sha256(plan_path),
        "stage_summary": str(summary_path), "stage_summary_sha256": sha256(summary_path),
        "dependency_ledger": str(ledger_path), "dependency_ledger_sha256": sha256(ledger_path),
        "replicas": str(replicas_path), "replicas_sha256": sha256(replicas_path),
        "groups": str(groups_path), "groups_sha256": sha256(groups_path),
        "analyzer": str(Path(__file__).resolve()), "analyzer_sha256": sha256(Path(__file__).resolve()),
        "raw_replica_analyzer": str(Path(raw.__file__).resolve()), "raw_replica_analyzer_sha256": sha256(Path(raw.__file__).resolve()),
        "bootstrap_seed": a5.BOOTSTRAP_SEED, "bootstrap_draws": a5.BOOTSTRAP_DRAWS,
        "replica_rows": len(replicas), "replica_status_counts": dict(replica_counts),
        "group_rows": len(groups), "group_status_counts": dict(group_counts),
        "protocol_sha256": A12_PROTOCOL_SHA256, "fixed_cutoff_cna_promoted": False,
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
