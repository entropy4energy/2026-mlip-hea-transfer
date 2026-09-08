#!/usr/bin/env python3
"""Apply the frozen 20-ps block-stationarity gate to all NPT rows."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from common import atomic_json, load_json, read_tsv, require_complete_receipt, sha256


COLUMNS = (
    "step", "time_ps", "natoms", "temperature_K", "pressure_bar",
    "potential_eV_per_atom", "total_eV_per_atom", "volume_A3_per_atom", "density_g_cm3",
)


def read_samples(path: Path) -> list[dict[str, float]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != len(COLUMNS):
            raise ValueError(f"{path}:{number}: expected {len(COLUMNS)} fields")
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{number}: non-finite sample")
        rows.append(dict(zip(COLUMNS, values, strict=True)))
    if not rows:
        raise ValueError(f"{path}: no samples")
    return rows


def blocks(rows: list[dict[str, float]], field: str, start: float, stop: float) -> dict[str, float | int]:
    values: list[float] = []
    for block in range(5):
        low, high = start + 2.0 * block, start + 2.0 * (block + 1)
        sample = [row[field] for row in rows if low < row["time_ps"] <= high + 1.0e-10]
        if not sample:
            raise ValueError(f"empty block for {field} in ({low},{high}]")
        values.append(statistics.fmean(sample))
    return {
        "mean": statistics.fmean(values),
        "block_standard_error": statistics.stdev(values) / math.sqrt(len(values)),
        "blocks": len(values),
    }


def comparison(field: str, early: dict[str, float | int], late: dict[str, float | int], floor: float) -> dict[str, object]:
    difference = abs(float(late["mean"]) - float(early["mean"]))
    tolerance = max(
        2.0 * math.hypot(float(early["block_standard_error"]), float(late["block_standard_error"])),
        floor,
    )
    return {
        "metric": field,
        "early": early,
        "late": late,
        "absolute_window_difference": difference,
        "tolerance": tolerance,
        "passed": difference <= tolerance,
    }


def analyze(path: Path, target_temperature: float) -> dict[str, object]:
    rows = read_samples(path)
    times = [row["time_ps"] for row in rows]
    errors: list[str] = []
    if any(b <= a for a, b in zip(times, times[1:])):
        errors.append("times are not strictly increasing")
    if max(times) - min(times) < 39.8:
        errors.append("trace is shorter than 40 ps within sampling tolerance")
    populations = {int(round(row["natoms"])) for row in rows}
    if len(populations) != 1 or next(iter(populations), 0) <= 0:
        errors.append("atom population is not constant and positive")
    stop = max(times)
    early_start, split = stop - 20.0, stop - 10.0
    late_volume = statistics.fmean(row["volume_A3_per_atom"] for row in rows if split < row["time_ps"] <= stop)
    metrics = {}
    for field, floor in (
        ("temperature_K", 0.02 * target_temperature),
        ("potential_eV_per_atom", 0.005),
        ("volume_A3_per_atom", 0.005 * late_volume),
        ("pressure_bar", 2500.0),
    ):
        metrics[field] = comparison(
            field, blocks(rows, field, early_start, split), blocks(rows, field, split, stop), floor
        )
    late_temperature = metrics["temperature_K"]["late"]
    temperature_tolerance = max(
        2.0 * float(late_temperature["block_standard_error"]), 0.02 * target_temperature
    )
    temperature_difference = abs(float(late_temperature["mean"]) - target_temperature)
    late_pressure = metrics["pressure_bar"]["late"]
    pressure_tolerance = max(2.0 * float(late_pressure["block_standard_error"]), 2500.0)
    pressure_difference = abs(float(late_pressure["mean"]))
    targets = {
        "temperature": {
            "late_mean": late_temperature["mean"], "target": target_temperature,
            "absolute_difference": temperature_difference, "tolerance": temperature_tolerance,
            "passed": temperature_difference <= temperature_tolerance,
        },
        "pressure": {
            "late_mean": late_pressure["mean"], "target_bar": 0.0,
            "absolute_difference": pressure_difference, "tolerance": pressure_tolerance,
            "passed": pressure_difference <= pressure_tolerance,
        },
    }
    failed_metrics = [name for name, value in metrics.items() if not value["passed"]]
    failed_targets = [name for name, value in targets.items() if not value["passed"]]
    if failed_metrics:
        errors.append("nonstationary metrics: " + ", ".join(failed_metrics))
    if failed_targets:
        errors.append("target failures: " + ", ".join(failed_targets))
    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "samples": len(rows),
        "duration_ps": max(times) - min(times),
        "natoms": next(iter(populations), None) if len(populations) == 1 else None,
        "stationarity_window_ps": 20.0,
        "comparison_window_ps": 10.0,
        "block_length_ps": 2.0,
        "metrics": metrics,
        "target_checks": targets,
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
    if {row["stage"] for row in rows} != {"equilibrate"}:
        raise ValueError("gate requires an equilibrate-only plan")
    results = []
    for row in rows:
        root = Path(row["output_dir"])
        require_complete_receipt(root / "run.complete.json", root / "equilibrated.data")
        result = analyze(root / "equilibration_samples.dat", float(row["temperature_K"]))
        result.update({"task_id": int(row["task_id"]), "run_id": row["run_id"], "plan_sha256": sha256(plan)})
        gate = root / "stationarity_gate.json"
        atomic_json(gate, result)
        results.append({"task_id": int(row["task_id"]), "run_id": row["run_id"], "status": result["status"], "gate": str(gate), "gate_sha256": sha256(gate)})
    failed = sum(row["status"] != "passed" for row in results)
    payload = {
        "schema_version": 1,
        "status": "passed" if failed == 0 else "failed_stationarity",
        "plan": str(plan), "plan_sha256": sha256(plan),
        "rows": len(results), "passed": len(results) - failed, "failed": failed,
        "results": results,
        "gatekeeper": str(Path(__file__).resolve()), "gatekeeper_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
