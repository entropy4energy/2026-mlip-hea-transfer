#!/usr/bin/env python3
"""Independently replay finite-rate tensile endpoints from raw curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


T95_DF2 = 4.302652729911275
CURVE_COLUMNS = (
    "step", "time_ps", "strain", "lateral_y", "lateral_z", "temperature_K",
    "sxx_GPa", "syy_GPa", "szz_GPa", "sxy_GPa", "sxz_GPa", "syz_GPa",
    "potential_eV_per_atom", "volume_A3_per_atom",
)
ENDPOINTS = (
    "tangent_modulus_GPa", "yield_0p2_GPa", "uts_GPa", "strain_at_uts",
    "work_to_20pct_GJ_m3",
)
ABS_TOL = 2.0e-8
REL_TOL = 2.0e-10


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_table(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"{path}: empty table")
    return rows


def write_json(path: Path, value: dict[str, object]) -> None:
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


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def parse_curve(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != len(CURVE_COLUMNS):
            raise ValueError(f"{path}:{number}: expected {len(CURVE_COLUMNS)} values")
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{number}: non-finite value")
        rows.append(values)
    data = np.asarray(rows, dtype=float)
    if data.shape[0] < 1000:
        raise ValueError(f"{path}: only {data.shape[0]} curve samples")
    if np.any(np.diff(data[:, 0]) <= 0.0) or np.any(np.diff(data[:, 2]) < -1.0e-12):
        raise ValueError(f"{path}: step/strain order is not monotonic")
    return data


def replay_curve(path: Path) -> dict[str, float | str]:
    data = parse_curve(path)
    strain = data[:, 2]
    stress = data[:, 6]

    elastic = (strain >= 0.0) & (strain <= 0.005)
    if int(np.count_nonzero(elastic)) < 20:
        raise ValueError(f"{path}: insufficient elastic-window samples")
    design = np.column_stack((strain[elastic], np.ones(int(np.count_nonzero(elastic)))))
    slope, intercept = np.linalg.lstsq(design, stress[elastic], rcond=None)[0]

    bin_index = np.floor(strain / 0.001 + 1.0e-10).astype(np.int64)
    offset = int(bin_index.min())
    shifted = bin_index - offset
    counts = np.bincount(shifted)
    present = counts > 0
    bstrain = np.bincount(shifted, weights=strain)[present] / counts[present]
    bstress = np.bincount(shifted, weights=stress)[present] / counts[present]
    difference = bstress - (intercept + slope * (bstrain - 0.002))
    eligible = np.flatnonzero(bstrain >= 0.002)
    yield_strain: float | None = None
    for left, right in zip(eligible[:-1], eligible[1:]):
        if difference[left] > 0.0 and difference[right] <= 0.0:
            fraction = difference[left] / (difference[left] - difference[right])
            yield_strain = float(bstrain[left] + fraction * (bstrain[right] - bstrain[left]))
            break
    yield_stress = (
        float(np.interp(yield_strain, bstrain, bstress))
        if yield_strain is not None else math.nan
    )

    target = bstrain <= 0.2002
    if not np.any(target) or float(strain[-1]) < 0.199:
        raise ValueError(f"{path}: curve does not reach the predeclared 20% target")
    selected_x = bstrain[target]
    selected_y = bstress[target]
    maximum = int(np.argmax(selected_y))
    fitted = design @ np.asarray((slope, intercept))
    residual = float(np.sum((stress[elastic] - fitted) ** 2))
    total = float(np.sum((stress[elastic] - np.mean(stress[elastic])) ** 2))
    return {
        "tangent_modulus_GPa": float(slope),
        "elastic_fit_intercept_GPa": float(intercept),
        "elastic_fit_R2": 1.0 - residual / total if total > 0.0 else math.nan,
        "yield_0p2_status": "resolved" if yield_strain is not None else "unresolved_no_crossing",
        "yield_0p2_GPa": yield_stress,
        "yield_0p2_strain": yield_strain if yield_strain is not None else math.nan,
        "uts_GPa": float(selected_y[maximum]),
        "strain_at_uts": float(selected_x[maximum]),
        "work_to_20pct_GJ_m3": float(np.trapezoid(selected_y, selected_x)),
        "mean_temperature_K": float(np.mean(data[:, 5])),
        "final_lateral_y_strain": float(data[-1, 3]),
        "final_lateral_z_strain": float(data[-1, 4]),
    }


def compare_numeric(errors: list[str], label: str, observed: str, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        errors.append(f"{label}: missing/non-numeric value")
        return
    if not math.isfinite(value) or not close(value, expected):
        errors.append(f"{label}: table={value!r}, replay={expected!r}")


def mean_interval(values: list[float]) -> tuple[float, float, float, float]:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sd = math.sqrt(variance)
    half_width = T95_DF2 * sd / math.sqrt(len(values))
    return mean, sd, mean - half_width, mean + half_width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-validation", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = args.plan.resolve()
    execution_path = args.execution_validation.resolve()
    analysis_dir = args.analysis_dir.resolve()
    receipt_path = analysis_dir / "analysis_receipt.json"
    replica_path = analysis_dir / "tension_replica_endpoints.csv"
    group_path = analysis_dir / "tension_group_summaries.csv"
    rate_path = analysis_dir / "tension_rate_sensitivity.csv"
    errors: list[str] = []
    checks = 0

    execution = load_json(execution_path)
    checks += 2
    if execution.get("status") != "passed":
        errors.append("execution validation is not passed")
    if execution.get("plan_sha256") != digest(plan):
        errors.append("execution validation is not bound to the plan")

    receipt = load_json(receipt_path)
    bindings = {
        "plan_sha256": digest(plan),
        "execution_validation_sha256": digest(execution_path),
        "replica_table_sha256": digest(replica_path),
        "group_table_sha256": digest(group_path),
        "rate_table_sha256": digest(rate_path),
    }
    for field, expected in bindings.items():
        checks += 1
        if receipt.get(field) != expected:
            errors.append(f"analysis receipt {field} mismatch")
    checks += 1
    if receipt.get("status") != "passed":
        errors.append("analysis receipt is not passed")

    plan_rows = read_table(plan, "\t")
    replica_rows = read_table(replica_path)
    group_rows = read_table(group_path)
    rate_rows = read_table(rate_path)
    checks += 4
    if len(plan_rows) != 18 or len(replica_rows) != 18:
        errors.append(f"expected 18 plan/replica rows; found {len(plan_rows)}/{len(replica_rows)}")
    if len(group_rows) != 6:
        errors.append(f"expected six group rows; found {len(group_rows)}")

    plan_by_id = {row["run_id"]: row for row in plan_rows}
    replica_by_id = {row["run_id"]: row for row in replica_rows}
    checks += 2
    if len(plan_by_id) != len(plan_rows) or set(plan_by_id) != set(replica_by_id):
        errors.append("plan and replica run-ID populations differ or are not unique")

    replayed: dict[str, dict[str, float | str]] = {}
    raw_curve_hashes: dict[str, str] = {}
    numeric_fields = (
        "tangent_modulus_GPa", "elastic_fit_intercept_GPa", "elastic_fit_R2",
        "uts_GPa", "strain_at_uts", "work_to_20pct_GJ_m3", "mean_temperature_K",
        "final_lateral_y_strain", "final_lateral_z_strain",
    )
    for run_id, plan_row in sorted(plan_by_id.items()):
        curve = Path(plan_row["output_dir"]) / "stress_strain.dat"
        try:
            replay = replay_curve(curve)
        except Exception as exc:  # report every failed run in one receipt
            errors.append(f"{run_id}: {exc}")
            continue
        replayed[run_id] = replay
        raw_curve_hashes[run_id] = digest(curve)
        table_row = replica_by_id[run_id]
        checks += len(numeric_fields) + 4
        for field in numeric_fields:
            compare_numeric(errors, f"{run_id}:{field}", table_row[field], float(replay[field]))
        if table_row["yield_0p2_status"] != replay["yield_0p2_status"]:
            errors.append(f"{run_id}: yield status differs")
        if replay["yield_0p2_status"] == "resolved":
            compare_numeric(errors, f"{run_id}:yield_0p2_GPa", table_row["yield_0p2_GPa"], float(replay["yield_0p2_GPa"]))
            compare_numeric(errors, f"{run_id}:yield_0p2_strain", table_row["yield_0p2_strain"], float(replay["yield_0p2_strain"]))
        elif table_row["yield_0p2_GPa"] or table_row["yield_0p2_strain"]:
            errors.append(f"{run_id}: unresolved yield has populated numeric fields")
        if table_row["curve_sha256"] != raw_curve_hashes[run_id]:
            errors.append(f"{run_id}: raw curve hash differs")

    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    rates: set[float] = set()
    for run_id, row in plan_by_id.items():
        grouped[(row["orientation"], int(row["temperature_K"]))].append(run_id)
        rates.add(float(row["strain_rate_ps_inverse"]))
    checks += 3
    if len(grouped) != 6 or {len(group) for group in grouped.values()} != {3}:
        errors.append("inferential matrix is not six orientation/temperature groups of three")
    if rates != {0.005}:
        errors.append(f"unexpected core strain-rate population: {sorted(rates)}")
    if len(rate_rows) != 6 or {int(row["chemical_seed"]) for row in rate_rows} != {20260825}:
        errors.append("rate-reference table population is not the six core 20260825 rows")

    summary_by_key = {(row["orientation"], int(row["temperature_K"])): row for row in group_rows}
    if set(summary_by_key) != set(grouped):
        errors.append("group-summary keys differ from the plan population")
    for key, run_ids in sorted(grouped.items()):
        if key not in summary_by_key or any(run_id not in replayed for run_id in run_ids):
            continue
        summary = summary_by_key[key]
        for endpoint in ENDPOINTS:
            values = [float(replayed[run_id][endpoint]) for run_id in run_ids]
            resolved = all(math.isfinite(value) for value in values)
            prefix = f"{key[0]}/{key[1]}:{endpoint}"
            checks += 5
            if not resolved:
                if summary[f"{endpoint}_status"] != "unresolved_in_at_least_one_realization":
                    errors.append(f"{prefix}: unresolved group status differs")
                for suffix in ("mean", "sd", "ci95_low", "ci95_high"):
                    if summary[f"{endpoint}_{suffix}"]:
                        errors.append(f"{prefix}: unresolved group has populated {suffix}")
                continue
            expected = mean_interval(values)
            if summary[f"{endpoint}_status"] != "estimated":
                errors.append(f"{prefix}: expected estimated status")
            for suffix, value in zip(("mean", "sd", "ci95_low", "ci95_high"), expected):
                compare_numeric(errors, f"{prefix}:{suffix}", summary[f"{endpoint}_{suffix}"], value)

    result = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "errors": errors,
        "plan": str(plan),
        "plan_sha256": digest(plan),
        "execution_validation": str(execution_path),
        "execution_validation_sha256": digest(execution_path),
        "analysis_receipt": str(receipt_path),
        "analysis_receipt_sha256": digest(receipt_path),
        "replica_rows_replayed": len(replayed),
        "group_rows_replayed": len(grouped),
        "inferential_unit": "chemical/velocity realization pair",
        "interval_replay": "two-sided Student-t, df=2, unadjusted",
        "observed_strain_rates_ps_inverse": sorted(rates),
        "rate_sensitivity_eligible": len(rates) > 1,
        "rate_sensitivity_note": "The selected core contains one rate; the six-row rate table is a reference subset, not a rate-sensitivity result.",
        "raw_curve_sha256": raw_curve_hashes,
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": digest(Path(__file__).resolve()),
    }
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
