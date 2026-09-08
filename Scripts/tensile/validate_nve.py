#!/usr/bin/env python3
"""Independently gate the six 10-ps NVE qualification trajectories."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from common import atomic_json, read_tsv, require_complete_receipt, sha256


COLUMNS = (
    "step", "time_ps", "natoms", "temperature_K", "pressure_bar",
    "potential_eV_per_atom", "kinetic_eV_per_atom", "total_eV_per_atom",
)
LIMIT_MEV_PER_ATOM_PS = 0.1


def analyze(path: Path, target_temperature: float) -> dict[str, object]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != len(COLUMNS):
            raise ValueError(f"{path}:{number}: expected {len(COLUMNS)} fields")
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{number}: non-finite value")
        rows.append(dict(zip(COLUMNS, values, strict=True)))
    if len(rows) < 400:
        raise ValueError(f"{path}: too few NVE samples")
    time = np.asarray([row["time_ps"] for row in rows])
    energy = np.asarray([row["total_eV_per_atom"] for row in rows])
    populations = {int(round(row["natoms"])) for row in rows}
    errors = []
    if len(populations) != 1 or next(iter(populations), 0) <= 0:
        errors.append("atom population changed")
    if time[-1] - time[0] < 9.95:
        errors.append("NVE trace is shorter than 10 ps within sampling tolerance")
    slope = float(np.polyfit(time, energy, 1)[0])
    drift = abs(slope) * 1000.0
    if drift > LIMIT_MEV_PER_ATOM_PS:
        errors.append("absolute energy drift exceeded the frozen threshold")
    mean_temperature = statistics.fmean(row["temperature_K"] for row in rows)
    temperature_tolerance = 0.10 * target_temperature
    if abs(mean_temperature - target_temperature) > temperature_tolerance:
        errors.append("mean NVE temperature differs by more than 10% from target")
    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "samples": len(rows),
        "duration_ps": float(time[-1] - time[0]),
        "natoms": next(iter(populations), None) if len(populations) == 1 else None,
        "target_temperature_K": target_temperature,
        "mean_temperature_K": mean_temperature,
        "energy_drift_eV_per_atom_ps": slope,
        "absolute_drift_meV_per_atom_ps": drift,
        "absolute_limit_meV_per_atom_ps": LIMIT_MEV_PER_ATOM_PS,
        "errors": errors,
        "samples_path": str(path.resolve()),
        "samples_sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = args.plan.resolve()
    rows = read_tsv(plan)
    if len(rows) != 6 or {row["stage"] for row in rows} != {"nve"}:
        raise ValueError("NVE gate requires the complete six-row plan")
    results = []
    for row in rows:
        root = Path(row["output_dir"])
        require_complete_receipt(root / "run.complete.json", root / "nve_final.data")
        result = analyze(root / "nve_energy_samples.dat", float(row["temperature_K"]))
        result.update({"task_id": int(row["task_id"]), "run_id": row["run_id"], "plan_sha256": sha256(plan)})
        gate = root / "nve_gate.json"
        atomic_json(gate, result)
        results.append({"task_id": int(row["task_id"]), "run_id": row["run_id"], "status": result["status"], "gate": str(gate), "gate_sha256": sha256(gate)})
    failed = sum(row["status"] != "passed" for row in results)
    payload = {
        "schema_version": 1, "status": "passed" if failed == 0 else "failed_nve",
        "plan": str(plan), "plan_sha256": sha256(plan), "rows": len(results),
        "passed": len(results) - failed, "failed": failed, "results": results,
        "validator": str(Path(__file__).resolve()), "validator_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
