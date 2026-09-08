#!/usr/bin/env python3
"""Apply the frozen stationarity gate to one NVT or NPT equilibration trace."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from pathlib import Path


COLUMNS = (
    "step",
    "time_ps",
    "natoms",
    "temperature_K",
    "pressure_bar",
    "potential_eV_per_atom",
    "total_eV_per_atom",
    "volume_A3_per_atom",
    "density_g_cm3",
)


def read_samples(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != len(COLUMNS):
            raise ValueError(
                f"{path}:{line_number}: expected {len(COLUMNS)} columns, found {len(fields)}"
            )
        values = [float(field) for field in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: non-finite value")
        rows.append(dict(zip(COLUMNS, values, strict=True)))
    if not rows:
        raise ValueError(f"no numeric samples in {path}")
    return rows


def block_summary(
    rows: list[dict[str, float]], field: str, start: float, stop: float, block_ps: float
) -> dict[str, float | int]:
    selected = [row for row in rows if start < row["time_ps"] <= stop + 1.0e-10]
    blocks: dict[int, list[float]] = {}
    for row in selected:
        block = min(int((row["time_ps"] - start) / block_ps), math.ceil((stop - start) / block_ps) - 1)
        blocks.setdefault(block, []).append(row[field])
    means = [statistics.fmean(blocks[index]) for index in sorted(blocks) if blocks[index]]
    if len(means) < 4:
        raise ValueError(f"fewer than four populated {block_ps:g} ps blocks for {field}")
    standard_error = statistics.stdev(means) / math.sqrt(len(means))
    return {
        "mean": statistics.fmean(means),
        "block_standard_error": standard_error,
        "blocks": len(means),
        "samples": len(selected),
    }


def compare_windows(
    name: str,
    early: dict[str, float | int],
    late: dict[str, float | int],
    floor: float,
) -> dict[str, float | bool | dict[str, float | int]]:
    difference = abs(float(late["mean"]) - float(early["mean"]))
    combined_se = math.hypot(
        float(early["block_standard_error"]), float(late["block_standard_error"])
    )
    tolerance = max(2.0 * combined_se, floor)
    return {
        "metric": name,
        "early": early,
        "late": late,
        "absolute_window_difference": difference,
        "tolerance": tolerance,
        "passed": difference <= tolerance,
    }


def analyze(
    rows: list[dict[str, float]],
    ensemble: str,
    target_temperature: float,
    target_pressure_bar: float,
) -> dict[str, object]:
    errors: list[str] = []
    times = [row["time_ps"] for row in rows]
    if any(second <= first for first, second in zip(times, times[1:])):
        errors.append("sample times are not strictly increasing")
    if max(times) - min(times) < 39.8:
        errors.append("equilibration trace is shorter than 40 ps within sampling tolerance")
    populations = {int(round(row["natoms"])) for row in rows}
    if len(populations) != 1 or next(iter(populations), 0) <= 0:
        errors.append("atom population is not constant and positive")

    stop = max(times)
    early_start, split = stop - 20.0, stop - 10.0
    metrics: dict[str, object] = {}
    for field, floor in (
        ("temperature_K", 0.02 * target_temperature),
        ("potential_eV_per_atom", 0.005),
    ):
        early = block_summary(rows, field, early_start, split, 2.0)
        late = block_summary(rows, field, split, stop, 2.0)
        metrics[field] = compare_windows(field, early, late, floor)

    late_temperature = metrics["temperature_K"]["late"]  # type: ignore[index]
    temperature_target_tolerance = max(
        2.0 * float(late_temperature["block_standard_error"]),
        0.02 * target_temperature,
    )
    temperature_target_difference = abs(float(late_temperature["mean"]) - target_temperature)
    target_checks: dict[str, object] = {
        "temperature": {
            "late_mean": late_temperature["mean"],
            "target": target_temperature,
            "absolute_difference": temperature_target_difference,
            "tolerance": temperature_target_tolerance,
            "passed": temperature_target_difference <= temperature_target_tolerance,
        }
    }

    if ensemble == "npt":
        late_volume_reference = statistics.fmean(
            row["volume_A3_per_atom"] for row in rows if split < row["time_ps"] <= stop
        )
        for field, floor in (
            ("volume_A3_per_atom", 0.005 * late_volume_reference),
            ("pressure_bar", 2500.0),
        ):
            early = block_summary(rows, field, early_start, split, 2.0)
            late = block_summary(rows, field, split, stop, 2.0)
            metrics[field] = compare_windows(field, early, late, floor)
        late_pressure = metrics["pressure_bar"]["late"]  # type: ignore[index]
        pressure_target_tolerance = max(
            2.0 * float(late_pressure["block_standard_error"]), 2500.0
        )
        pressure_target_difference = abs(float(late_pressure["mean"]) - target_pressure_bar)
        target_checks["pressure"] = {
            "late_mean": late_pressure["mean"],
            "target": target_pressure_bar,
            "absolute_difference": pressure_target_difference,
            "tolerance": pressure_target_tolerance,
            "passed": pressure_target_difference <= pressure_target_tolerance,
        }

    failed_metrics = [
        name for name, result in metrics.items() if not bool(result["passed"])  # type: ignore[index]
    ]
    failed_targets = [
        name for name, result in target_checks.items() if not bool(result["passed"])  # type: ignore[index]
    ]
    if failed_metrics:
        errors.append(f"nonstationary window metrics: {', '.join(failed_metrics)}")
    if failed_targets:
        errors.append(f"target checks failed: {', '.join(failed_targets)}")
    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "ensemble": ensemble,
        "samples": len(rows),
        "duration_ps": max(times) - min(times),
        "natoms": next(iter(populations), None) if len(populations) == 1 else None,
        "stationarity_window_ps": 20.0,
        "comparison_window_ps": 10.0,
        "block_length_ps": 2.0,
        "metrics": metrics,
        "target_checks": target_checks,
        "errors": errors,
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--ensemble", choices=("nvt", "npt"), required=True)
    parser.add_argument("--target-temperature", type=float, required=True)
    parser.add_argument("--target-pressure-bar", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        read_samples(args.samples),
        args.ensemble,
        args.target_temperature,
        args.target_pressure_bar,
    )
    result["samples_path"] = str(args.samples.resolve())
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
