#!/usr/bin/env python3
"""Reconstruct predeclared tensile endpoints and realization-level summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import atomic_json, load_json, read_tsv, sha256


T95_DF2 = 4.302652729911275
COLUMNS = (
    "step", "time_ps", "strain", "lateral_y", "lateral_z", "temperature_K",
    "sxx_GPa", "syy_GPa", "szz_GPa", "sxy_GPa", "sxz_GPa", "syz_GPa",
    "potential_eV_per_atom", "volume_A3_per_atom",
)
ENDPOINTS = ("tangent_modulus_GPa", "yield_0p2_GPa", "uts_GPa", "strain_at_uts", "work_to_20pct_GJ_m3")


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
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


def read_curve(path: Path) -> dict[str, np.ndarray]:
    values = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != len(COLUMNS):
            raise ValueError(f"{path}:{number}: expected {len(COLUMNS)} values")
        row = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{path}:{number}: non-finite value")
        values.append(row)
    data = np.asarray(values)
    if data.shape[0] < 1000:
        raise ValueError(f"{path}: insufficient curve samples")
    return {name: data[:, index] for index, name in enumerate(COLUMNS)}


def binned_curve(strain: np.ndarray, stress: np.ndarray, width: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    indices = np.floor(strain / width + 1.0e-10).astype(int)
    output_x, output_y = [], []
    for index in sorted(set(indices)):
        mask = indices == index
        output_x.append(float(np.mean(strain[mask])))
        output_y.append(float(np.mean(stress[mask])))
    return np.asarray(output_x), np.asarray(output_y)


def interpolate_crossing(strain: np.ndarray, difference: np.ndarray, start: float) -> float | None:
    eligible = np.flatnonzero(strain >= start)
    for left, right in zip(eligible, eligible[1:]):
        if difference[left] > 0.0 and difference[right] <= 0.0:
            fraction = difference[left] / (difference[left] - difference[right])
            return float(strain[left] + fraction * (strain[right] - strain[left]))
    return None


def analyze_one(row: dict[str, str]) -> dict[str, object]:
    path = Path(row["output_dir"]) / "stress_strain.dat"
    curve = read_curve(path)
    strain = curve["strain"]
    stress = curve["sxx_GPa"]
    elastic = (strain >= 0.0) & (strain <= 0.005)
    if int(np.sum(elastic)) < 20:
        raise ValueError(f"{path}: too few samples in fixed elastic-fit window")
    slope, intercept = np.polyfit(strain[elastic], stress[elastic], 1)
    fitted = slope * strain[elastic] + intercept
    ss_res = float(np.sum((stress[elastic] - fitted) ** 2))
    ss_tot = float(np.sum((stress[elastic] - np.mean(stress[elastic])) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    bstrain, bstress = binned_curve(strain, stress)
    offset_line = intercept + slope * (bstrain - 0.002)
    yield_strain = interpolate_crossing(bstrain, bstress - offset_line, 0.002)
    yield_stress = (
        float(np.interp(yield_strain, bstrain, bstress)) if yield_strain is not None else math.nan
    )
    through_target = bstrain <= 0.2002
    max_index = int(np.argmax(bstress[through_target]))
    selected_strain = bstrain[through_target]
    selected_stress = bstress[through_target]
    uts = float(selected_stress[max_index])
    strain_at_uts = float(selected_strain[max_index])
    work = float(np.trapezoid(selected_stress, selected_strain))
    return {
        "task_id": int(row["task_id"]),
        "run_id": row["run_id"],
        "profile": row["profile"],
        "orientation": row["orientation"],
        "temperature_K": int(row["temperature_K"]),
        "chemical_seed": int(row["chemical_seed"]),
        "velocity_seed": int(row["velocity_seed"]),
        "strain_rate_ps_inverse": float(row["strain_rate_ps_inverse"]),
        "strain_rate_s_inverse": float(row["strain_rate_ps_inverse"]) * 1.0e12,
        "tangent_modulus_GPa": float(slope),
        "elastic_fit_intercept_GPa": float(intercept),
        "elastic_fit_R2": r_squared,
        "yield_0p2_status": "resolved" if yield_strain is not None else "unresolved_no_crossing",
        "yield_0p2_GPa": yield_stress if yield_strain is not None else "",
        "yield_0p2_strain": yield_strain if yield_strain is not None else "",
        "uts_GPa": uts,
        "strain_at_uts": strain_at_uts,
        "work_to_20pct_GJ_m3": work,
        "mean_temperature_K": float(np.mean(curve["temperature_K"])),
        "final_lateral_y_strain": float(curve["lateral_y"][-1]),
        "final_lateral_z_strain": float(curve["lateral_z"][-1]),
        "curve": str(path.resolve()),
        "curve_sha256": sha256(path),
        "endpoint_curve_rule": "0.001-strain bin means; 0-0.005 raw tangent fit",
    }


def mean_interval(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    half = T95_DF2 * sd / math.sqrt(array.size)
    return mean, sd, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = args.plan.resolve()
    validation_path = args.validation.resolve()
    validation = load_json(validation_path)
    if validation.get("status") != "passed" or validation.get("plan_sha256") != sha256(plan):
        raise ValueError("tension execution validation is not passed/bound to this plan")
    rows = read_tsv(plan)
    per_run = [analyze_one(row) for row in rows]
    core = [row for row in per_run if math.isclose(float(row["strain_rate_ps_inverse"]), 0.005)]
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in core:
        grouped[(str(row["orientation"]), int(row["temperature_K"]))].append(row)
    if set(len(rows) for rows in grouped.values()) != {3} or len(grouped) != 6:
        raise ValueError("core inference matrix is not six groups of three realization pairs")
    summaries = []
    for (orientation, temperature), group in sorted(grouped.items()):
        summary: dict[str, object] = {
            "orientation": orientation,
            "temperature_K": temperature,
            "strain_rate_ps_inverse": 0.005,
            "strain_rate_s_inverse": 5.0e9,
            "n_realization_pairs": 3,
        }
        for endpoint in ENDPOINTS:
            values = [float(row[endpoint]) for row in group if row[endpoint] != ""]
            if len(values) != 3:
                summary[f"{endpoint}_mean"] = ""
                summary[f"{endpoint}_sd"] = ""
                summary[f"{endpoint}_ci95_low"] = ""
                summary[f"{endpoint}_ci95_high"] = ""
                summary[f"{endpoint}_status"] = "unresolved_in_at_least_one_realization"
            else:
                mean, sd, low, high = mean_interval(values)
                summary[f"{endpoint}_mean"] = mean
                summary[f"{endpoint}_sd"] = sd
                summary[f"{endpoint}_ci95_low"] = low
                summary[f"{endpoint}_ci95_high"] = high
                summary[f"{endpoint}_status"] = "estimated"
        summaries.append(summary)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_run_path = output / "tension_replica_endpoints.csv"
    summary_path = output / "tension_group_summaries.csv"
    rate_path = output / "tension_rate_sensitivity.csv"
    atomic_csv(per_run_path, per_run)
    atomic_csv(summary_path, summaries)
    rate_rows = [row for row in per_run if int(row["chemical_seed"]) == 20260825]
    atomic_csv(rate_path, rate_rows)
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "plan": str(plan), "plan_sha256": sha256(plan),
        "execution_validation": str(validation_path), "execution_validation_sha256": sha256(validation_path),
        "replica_rows": len(per_run), "core_rows": len(core), "group_rows": len(summaries),
        "replica_table": str(per_run_path), "replica_table_sha256": sha256(per_run_path),
        "group_table": str(summary_path), "group_table_sha256": sha256(summary_path),
        "rate_table": str(rate_path), "rate_table_sha256": sha256(rate_path),
        "analysis": str(Path(__file__).resolve()), "analysis_sha256": sha256(Path(__file__).resolve()),
        "inferential_unit": "chemical/velocity realization pair",
        "interval": "two-sided Student-t, df=2, unadjusted",
    }
    atomic_json(output / "analysis_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
