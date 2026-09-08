#!/usr/bin/env python3
"""Create manuscript-facing H100 tensile tables after the raw replay passes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


T95_DF2 = 4.302652729911275
ORIENTATIONS = ("100", "110", "111")
TEMPERATURES = (300, 1200)
ENDPOINTS = (
    "tangent_modulus_GPa",
    "yield_0p2_GPa",
    "uts_GPa",
    "strain_at_uts",
    "work_to_20pct_GJ_m3",
)
CURVE_COLUMNS = (
    "step",
    "time_ps",
    "strain",
    "lateral_y",
    "lateral_z",
    "temperature_K",
    "sxx_GPa",
    "syy_GPa",
    "szz_GPa",
    "sxy_GPa",
    "sxz_GPa",
    "syz_GPa",
    "potential_eV_per_atom",
    "volume_A3_per_atom",
)
ORIENTATION_CONTRASTS = (("100", "110"), ("100", "111"), ("110", "111"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_table(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"{path}: empty table")
    return rows


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
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


def finite_or_none(value: str) -> float | None:
    if value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite endpoint {value!r}")
    return number


def mean_interval(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 3:
        raise ValueError("Student-t summary requires exactly three realization pairs")
    mean = math.fsum(values) / 3.0
    variance = math.fsum((value - mean) ** 2 for value in values) / 2.0
    sd = math.sqrt(variance)
    half = T95_DF2 * sd / math.sqrt(3.0)
    return mean, sd, mean - half, mean + half


def _key(row: dict[str, str]) -> tuple[str, int, int, int]:
    return (
        row["orientation"],
        int(row["temperature_K"]),
        int(row["chemical_seed"]),
        int(row["velocity_seed"]),
    )


def validate_replica_population(rows: list[dict[str, str]]) -> dict[tuple[str, int, int, int], dict[str, str]]:
    indexed = {_key(row): row for row in rows}
    if len(rows) != 18 or len(indexed) != 18:
        raise ValueError("replica table is not 18 unique realization rows")
    pairings = {
        (int(row["chemical_seed"]), int(row["velocity_seed"])) for row in rows
    }
    if len(pairings) != 3:
        raise ValueError("replica table does not contain three fixed realization pairs")
    expected = {
        (orientation, temperature, chemical, velocity)
        for orientation in ORIENTATIONS
        for temperature in TEMPERATURES
        for chemical, velocity in pairings
    }
    if set(indexed) != expected:
        raise ValueError("replica table is not the complete orientation/temperature/pair matrix")
    return indexed


def build_paired_contrasts(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    indexed = validate_replica_population(rows)
    pairs = sorted({(key[2], key[3]) for key in indexed})
    definitions: list[tuple[str, str, str, int | str, tuple[str, int], tuple[str, int]]] = []
    for orientation in ORIENTATIONS:
        definitions.append(
            (
                "temperature",
                f"T1200_minus_T300__{orientation}",
                "1200 K - 300 K",
                orientation,
                (orientation, 300),
                (orientation, 1200),
            )
        )
    for temperature in TEMPERATURES:
        for first, second in ORIENTATION_CONTRASTS:
            definitions.append(
                (
                    "orientation",
                    f"{second}_minus_{first}__T{temperature}",
                    f"[{second}] - [{first}]",
                    temperature,
                    (first, temperature),
                    (second, temperature),
                )
            )

    values: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for family, contrast_id, definition, fixed_level, first, second in definitions:
        for endpoint in ENDPOINTS:
            differences: list[float] = []
            for pair_index, (chemical, velocity) in enumerate(pairs, 1):
                left = finite_or_none(indexed[(first[0], first[1], chemical, velocity)][endpoint])
                right = finite_or_none(indexed[(second[0], second[1], chemical, velocity)][endpoint])
                resolved = left is not None and right is not None
                difference = right - left if resolved else None
                if difference is not None:
                    differences.append(difference)
                values.append(
                    {
                        "contrast_family": family,
                        "contrast_id": contrast_id,
                        "difference_definition": definition,
                        "fixed_orientation": fixed_level if family == "temperature" else "",
                        "fixed_temperature_K": fixed_level if family == "orientation" else "",
                        "endpoint": endpoint,
                        "pair_index": pair_index,
                        "chemical_seed": chemical,
                        "velocity_seed": velocity,
                        "level_a_value": "" if left is None else left,
                        "level_b_value": "" if right is None else right,
                        "paired_difference_b_minus_a": "" if difference is None else difference,
                        "status": "resolved" if resolved else "unresolved_endpoint",
                    }
                )
            summary: dict[str, object] = {
                "contrast_family": family,
                "contrast_id": contrast_id,
                "difference_definition": definition,
                "fixed_orientation": fixed_level if family == "temperature" else "",
                "fixed_temperature_K": fixed_level if family == "orientation" else "",
                "endpoint": endpoint,
                "n_pairs_planned": 3,
                "n_pairs_eligible": len(differences),
            }
            if len(differences) == 3:
                mean, sd, low, high = mean_interval(differences)
                summary.update(
                    {
                        "status": "estimated",
                        "mean_difference_b_minus_a": mean,
                        "sd_difference": sd,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
            else:
                summary.update(
                    {
                        "status": "unresolved_in_at_least_one_realization",
                        "mean_difference_b_minus_a": "",
                        "sd_difference": "",
                        "ci95_low": "",
                        "ci95_high": "",
                    }
                )
            summaries.append(summary)
    if len(values) != 135 or len(summaries) != 45:
        raise AssertionError("unexpected paired-contrast population")
    return values, summaries


def parse_curve(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != len(CURVE_COLUMNS):
            raise ValueError(f"{path}:{number}: expected {len(CURVE_COLUMNS)} fields")
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{number}: non-finite curve value")
        rows.append(values)
    data = np.asarray(rows, dtype=float)
    if data.shape[0] < 1000 or np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"{path}: insufficient or unordered curve")
    if float(data[-1, 2]) < 0.1998:
        raise ValueError(f"{path}: curve does not reach 20% strain")
    return data


def bin_curve(path: Path) -> list[dict[str, object]]:
    data = parse_curve(path)
    strain = data[:, 2]
    stress = data[:, 6]
    indices = np.floor(strain / 0.001 + 1.0e-10).astype(int)
    rows: list[dict[str, object]] = []
    for index in sorted(set(indices)):
        mask = indices == index
        rows.append(
            {
                "strain_bin": int(index),
                "strain_mean": float(np.mean(strain[mask])),
                "stress_GPa_mean": float(np.mean(stress[mask])),
                "raw_samples": int(np.count_nonzero(mask)),
            }
        )
    if len(rows) < 200:
        raise ValueError(f"{path}: fewer than 200 strain bins")
    return rows


def build_curve_tables(
    replica_rows: list[dict[str, str]], curve_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_run: list[dict[str, object]] = []
    for replica in sorted(replica_rows, key=lambda row: int(row["task_id"])):
        curve = curve_root / replica["run_id"] / "stress_strain.dat"
        if not curve.is_file():
            raise FileNotFoundError(curve)
        actual_hash = sha256(curve)
        if actual_hash != replica["curve_sha256"]:
            raise ValueError(f"{replica['run_id']}: local curve hash differs from validated endpoint table")
        for row in bin_curve(curve):
            per_run.append(
                {
                    "run_id": replica["run_id"],
                    "orientation": replica["orientation"],
                    "temperature_K": int(replica["temperature_K"]),
                    "chemical_seed": int(replica["chemical_seed"]),
                    "velocity_seed": int(replica["velocity_seed"]),
                    **row,
                    "curve_sha256": actual_hash,
                }
            )

    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in per_run:
        grouped[(str(row["orientation"]), int(row["temperature_K"]), int(row["strain_bin"]))].append(row)
    group_rows: list[dict[str, object]] = []
    for (orientation, temperature, strain_bin), members in sorted(grouped.items()):
        if len(members) != 3:
            raise ValueError(
                f"curve group {orientation}/{temperature}/bin{strain_bin} has {len(members)} rows"
            )
        strain_values = [float(row["strain_mean"]) for row in members]
        stress_values = [float(row["stress_GPa_mean"]) for row in members]
        group_rows.append(
            {
                "orientation": orientation,
                "temperature_K": temperature,
                "strain_bin": strain_bin,
                "n_realization_pairs": 3,
                "strain_mean": math.fsum(strain_values) / 3.0,
                "stress_GPa_mean": math.fsum(stress_values) / 3.0,
                "stress_GPa_sd": float(np.std(stress_values, ddof=1)),
                "stress_GPa_min": min(stress_values),
                "stress_GPa_max": max(stress_values),
                "curve_role": "descriptive_equal_realization_mean_not_inferential_frames",
            }
        )
    return per_run, group_rows


def copy_table(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--analysis-receipt", type=Path, required=True)
    parser.add_argument("--analysis-validation", type=Path, required=True)
    parser.add_argument("--replica-table", type=Path, required=True)
    parser.add_argument("--group-table", type=Path, required=True)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = args.plan.resolve()
    receipt_path = args.analysis_receipt.resolve()
    validation_path = args.analysis_validation.resolve()
    replica_path = args.replica_table.resolve()
    group_path = args.group_table.resolve()
    curve_root = args.curve_root.resolve()
    output = args.output_dir.resolve()

    validation = read_json(validation_path)
    if validation.get("status") != "passed" or validation.get("errors") not in ([], None):
        raise ValueError("independent raw tension analysis did not pass cleanly")
    if validation.get("plan_sha256") != sha256(plan):
        raise ValueError("independent tension validation is not bound to the supplied plan")
    if validation.get("analysis_receipt_sha256") != sha256(receipt_path):
        raise ValueError("independent tension validation is not bound to the supplied receipt")
    if validation.get("replica_rows_replayed") != 18 or validation.get("group_rows_replayed") != 6:
        raise ValueError("independent tension validation did not replay the full population")

    receipt = read_json(receipt_path)
    expected_bindings = {
        "plan_sha256": sha256(plan),
        "replica_table_sha256": sha256(replica_path),
        "group_table_sha256": sha256(group_path),
    }
    for field, expected in expected_bindings.items():
        if receipt.get(field) != expected:
            raise ValueError(f"analysis receipt {field} mismatch")
    replica_rows = read_table(replica_path)
    group_rows = read_table(group_path)
    if len(group_rows) != 6:
        raise ValueError("validated group table does not contain six rows")
    validate_replica_population(replica_rows)

    contrast_values, contrast_summaries = build_paired_contrasts(replica_rows)
    curve_replicas, curve_groups = build_curve_tables(replica_rows, curve_root)

    paths = {
        "replica_endpoints": output / "lammps_h100_tension_replica_endpoints.csv",
        "group_summaries": output / "lammps_h100_tension_group_summaries.csv",
        "paired_contrast_values": output / "lammps_h100_tension_paired_contrast_values.csv",
        "paired_contrasts": output / "lammps_h100_tension_paired_contrasts.csv",
        "binned_replica_curves": output / "lammps_h100_tension_binned_replica_curves.csv",
        "group_curves": output / "lammps_h100_tension_group_curves.csv",
    }
    copy_table(replica_path, paths["replica_endpoints"])
    copy_table(group_path, paths["group_summaries"])
    atomic_csv(paths["paired_contrast_values"], contrast_values)
    atomic_csv(paths["paired_contrasts"], contrast_summaries)
    atomic_csv(paths["binned_replica_curves"], curve_replicas)
    atomic_csv(paths["group_curves"], curve_groups)
    output_records = {
        name: {
            "path": str(path),
            "sha256": sha256(path),
            "rows": len(read_table(path)),
        }
        for name, path in paths.items()
    }
    result = {
        "schema_version": 1,
        "status": "passed",
        "plan": str(plan),
        "plan_sha256": sha256(plan),
        "source_analysis_receipt": str(receipt_path),
        "source_analysis_receipt_sha256": sha256(receipt_path),
        "source_analysis_validation": str(validation_path),
        "source_analysis_validation_sha256": sha256(validation_path),
        "curve_root": str(curve_root),
        "replica_rows": len(replica_rows),
        "group_rows": len(group_rows),
        "paired_contrast_value_rows": len(contrast_values),
        "paired_contrast_rows": len(contrast_summaries),
        "binned_replica_curve_rows": len(curve_replicas),
        "group_curve_rows": len(curve_groups),
        "outputs": output_records,
        "inferential_unit": "fixed chemical/velocity realization pair",
        "interval": "two-sided Student-t, df=2, unadjusted",
        "curve_role": "descriptive; MD frames and strain bins are not inferential replicates",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(output / "lammps_h100_tension_promotion_receipt.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
