#!/usr/bin/env python3
"""Independently replay manuscript-facing H100 tensile tables from raw curves."""

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


T_CRITICAL = 4.302652729911275
ABS_TOL = 2.0e-8
REL_TOL = 2.0e-10
ORIENTATIONS = ("100", "110", "111")
TEMPERATURES = (300, 1200)
ENDPOINTS = (
    "tangent_modulus_GPa",
    "yield_0p2_GPa",
    "uts_GPa",
    "strain_at_uts",
    "work_to_20pct_GJ_m3",
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
        raise ValueError(f"{path}: expected object")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: empty CSV")
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
    return math.isclose(left, right, abs_tol=ABS_TOL, rel_tol=REL_TOL)


def number(value: str) -> float | None:
    if value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite table value {value!r}")
    return result


def compare(errors: list[str], label: str, observed: str, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        errors.append(f"{label}: missing numeric value")
        return
    if not math.isfinite(value) or not close(value, expected):
        errors.append(f"{label}: table={value!r}, replay={expected!r}")


def interval(values: list[float]) -> tuple[float, float, float, float]:
    mean = math.fsum(values) / 3.0
    variance = math.fsum((item - mean) ** 2 for item in values) / 2.0
    deviation = math.sqrt(variance)
    margin = T_CRITICAL * deviation / math.sqrt(3.0)
    return mean, deviation, mean - margin, mean + margin


def curve_bins(path: Path) -> list[tuple[int, float, float, int]]:
    data = np.loadtxt(path, comments="#", dtype=float)
    if data.ndim != 2 or data.shape[1] != 14 or data.shape[0] < 1000:
        raise ValueError(f"{path}: unexpected raw curve shape {data.shape}")
    if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"{path}: non-finite or unordered raw curve")
    if data[-1, 2] < 0.1998:
        raise ValueError(f"{path}: incomplete final strain")
    index = np.floor(data[:, 2] / 0.001 + 1.0e-10).astype(np.int64)
    output: list[tuple[int, float, float, int]] = []
    for value in np.unique(index):
        selected = index == value
        output.append(
            (
                int(value),
                float(math.fsum(data[selected, 2]) / int(np.count_nonzero(selected))),
                float(math.fsum(data[selected, 6]) / int(np.count_nonzero(selected))),
                int(np.count_nonzero(selected)),
            )
        )
    return output


def expected_contrasts(replica: list[dict[str, str]]):
    keyed = {
        (
            row["orientation"],
            int(row["temperature_K"]),
            int(row["chemical_seed"]),
            int(row["velocity_seed"]),
        ): row
        for row in replica
    }
    pairs = sorted({(key[2], key[3]) for key in keyed})
    comparisons: list[tuple[str, str, str, str, str, int, int]] = []
    for orientation in ORIENTATIONS:
        comparisons.append(
            (
                "temperature",
                f"T1200_minus_T300__{orientation}",
                "1200 K - 300 K",
                orientation,
                orientation,
                300,
                1200,
            )
        )
    for temperature in TEMPERATURES:
        for first, second in (("100", "110"), ("100", "111"), ("110", "111")):
            comparisons.append(
                (
                    "orientation",
                    f"{second}_minus_{first}__T{temperature}",
                    f"[{second}] - [{first}]",
                    first,
                    second,
                    temperature,
                    temperature,
                )
            )
    value_rows: dict[tuple[str, str, int], dict[str, object]] = {}
    summary_rows: dict[tuple[str, str], dict[str, object]] = {}
    for family, identifier, definition, first_orientation, second_orientation, first_temp, second_temp in comparisons:
        for endpoint in ENDPOINTS:
            differences: list[float] = []
            for pair_index, (chemical, velocity) in enumerate(pairs, 1):
                left = number(keyed[(first_orientation, first_temp, chemical, velocity)][endpoint])
                right = number(keyed[(second_orientation, second_temp, chemical, velocity)][endpoint])
                difference = right - left if left is not None and right is not None else None
                if difference is not None:
                    differences.append(difference)
                value_rows[(identifier, endpoint, pair_index)] = {
                    "family": family,
                    "definition": definition,
                    "chemical": chemical,
                    "velocity": velocity,
                    "left": left,
                    "right": right,
                    "difference": difference,
                }
            summary_rows[(identifier, endpoint)] = {
                "family": family,
                "definition": definition,
                "n": len(differences),
                "statistics": interval(differences) if len(differences) == 3 else None,
            }
    return value_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.promotion_receipt.resolve()
    receipt = load_json(receipt_path)
    errors: list[str] = []
    checks = 0

    if receipt.get("status") != "passed":
        errors.append("promotion receipt is not passed")
    source_validation_path = Path(str(receipt.get("source_analysis_validation", "")))
    source_receipt_path = Path(str(receipt.get("source_analysis_receipt", "")))
    for label, path, field in (
        ("source analysis validation", source_validation_path, "source_analysis_validation_sha256"),
        ("source analysis receipt", source_receipt_path, "source_analysis_receipt_sha256"),
    ):
        checks += 1
        if not path.is_file() or digest(path) != receipt.get(field):
            errors.append(f"{label} binding mismatch")
    if source_validation_path.is_file():
        source_validation = load_json(source_validation_path)
        checks += 4
        if source_validation.get("status") != "passed" or source_validation.get("errors") not in ([], None):
            errors.append("source independent analysis validation did not pass cleanly")
        if source_validation.get("replica_rows_replayed") != 18:
            errors.append("source independent analysis did not replay 18 replicas")
        if source_validation.get("group_rows_replayed") != 6:
            errors.append("source independent analysis did not replay six groups")

    output_records = receipt.get("outputs")
    if not isinstance(output_records, dict):
        raise ValueError("promotion receipt has no output records")
    paths: dict[str, Path] = {}
    for name, record in output_records.items():
        if not isinstance(record, dict):
            errors.append(f"invalid output record {name}")
            continue
        path = Path(str(record.get("path", "")))
        paths[name] = path
        checks += 2
        if not path.is_file():
            errors.append(f"missing output {name}")
        elif digest(path) != record.get("sha256"):
            errors.append(f"output hash mismatch {name}")

    required = {
        "replica_endpoints",
        "group_summaries",
        "paired_contrast_values",
        "paired_contrasts",
        "binned_replica_curves",
        "group_curves",
    }
    if set(paths) != required or errors:
        result = {
            "schema_version": 1,
            "status": "failed",
            "checks": checks,
            "errors": errors,
            "promotion_receipt": str(receipt_path),
            "promotion_receipt_sha256": digest(receipt_path),
            "validator": str(Path(__file__).resolve()),
            "validator_sha256": digest(Path(__file__).resolve()),
        }
        write_json(args.output.resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(1)

    replica = read_csv(paths["replica_endpoints"])
    groups = read_csv(paths["group_summaries"])
    values = read_csv(paths["paired_contrast_values"])
    summaries = read_csv(paths["paired_contrasts"])
    binned = read_csv(paths["binned_replica_curves"])
    group_curves = read_csv(paths["group_curves"])
    checks += 6
    expected_counts = (18, 6, 135, 45)
    observed_counts = (len(replica), len(groups), len(values), len(summaries))
    if observed_counts != expected_counts:
        errors.append(f"core promoted populations {observed_counts} != {expected_counts}")

    expected_values, expected_summaries = expected_contrasts(replica)
    actual_values = {
        (row["contrast_id"], row["endpoint"], int(row["pair_index"])): row for row in values
    }
    if set(actual_values) != set(expected_values):
        errors.append("paired contrast-value keys differ from independent replay")
    for key, expected in expected_values.items():
        row = actual_values.get(key)
        if row is None:
            continue
        checks += 8
        if row["contrast_family"] != expected["family"] or row["difference_definition"] != expected["definition"]:
            errors.append(f"{key}: contrast metadata differs")
        if int(row["chemical_seed"]) != expected["chemical"] or int(row["velocity_seed"]) != expected["velocity"]:
            errors.append(f"{key}: realization mapping differs")
        for field, item in (
            ("level_a_value", expected["left"]),
            ("level_b_value", expected["right"]),
            ("paired_difference_b_minus_a", expected["difference"]),
        ):
            if item is None:
                if row[field] != "":
                    errors.append(f"{key}:{field}: expected blank")
            else:
                compare(errors, f"{key}:{field}", row[field], float(item))

    actual_summaries = {(row["contrast_id"], row["endpoint"]): row for row in summaries}
    if set(actual_summaries) != set(expected_summaries):
        errors.append("paired contrast-summary keys differ from independent replay")
    for key, expected in expected_summaries.items():
        row = actual_summaries.get(key)
        if row is None:
            continue
        checks += 7
        if int(row["n_pairs_eligible"]) != expected["n"]:
            errors.append(f"{key}: eligible-pair count differs")
        statistics = expected["statistics"]
        if statistics is None:
            if row["status"] != "unresolved_in_at_least_one_realization":
                errors.append(f"{key}: unresolved status differs")
            for field in ("mean_difference_b_minus_a", "sd_difference", "ci95_low", "ci95_high"):
                if row[field] != "":
                    errors.append(f"{key}:{field}: unresolved summary is populated")
        else:
            if row["status"] != "estimated":
                errors.append(f"{key}: estimated status differs")
            for field, expected_value in zip(
                ("mean_difference_b_minus_a", "sd_difference", "ci95_low", "ci95_high"),
                statistics,
            ):
                compare(errors, f"{key}:{field}", row[field], expected_value)

    binned_by_run: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in binned:
        binned_by_run[row["run_id"]][int(row["strain_bin"])] = row
    replayed_groups: dict[tuple[str, int, int], list[tuple[float, float]]] = defaultdict(list)
    curve_root = Path(str(receipt.get("curve_root", "")))
    for row in replica:
        run_id = row["run_id"]
        curve = curve_root / run_id / "stress_strain.dat"
        checks += 2
        if not curve.is_file() or digest(curve) != row["curve_sha256"]:
            errors.append(f"{run_id}: raw curve missing or hash mismatch")
            continue
        expected_bins = curve_bins(curve)
        actual_bins = binned_by_run.get(run_id, {})
        if set(actual_bins) != {item[0] for item in expected_bins}:
            errors.append(f"{run_id}: strain-bin population differs")
            continue
        for strain_bin, strain_mean, stress_mean, samples in expected_bins:
            table = actual_bins[strain_bin]
            checks += 3
            compare(errors, f"{run_id}/bin{strain_bin}:strain", table["strain_mean"], strain_mean)
            compare(errors, f"{run_id}/bin{strain_bin}:stress", table["stress_GPa_mean"], stress_mean)
            if int(table["raw_samples"]) != samples:
                errors.append(f"{run_id}/bin{strain_bin}: sample count differs")
            replayed_groups[(row["orientation"], int(row["temperature_K"]), strain_bin)].append(
                (strain_mean, stress_mean)
            )

    actual_group_curves = {
        (row["orientation"], int(row["temperature_K"]), int(row["strain_bin"])): row
        for row in group_curves
    }
    if set(actual_group_curves) != set(replayed_groups):
        errors.append("group-curve keys differ from raw-curve replay")
    for key, members in replayed_groups.items():
        row = actual_group_curves.get(key)
        if row is None:
            continue
        checks += 6
        if len(members) != 3 or int(row["n_realization_pairs"]) != 3:
            errors.append(f"{key}: group curve does not contain three realizations")
            continue
        strain_values = [member[0] for member in members]
        stress_values = [member[1] for member in members]
        expected = (
            math.fsum(strain_values) / 3.0,
            math.fsum(stress_values) / 3.0,
            float(np.std(stress_values, ddof=1)),
            min(stress_values),
            max(stress_values),
        )
        for field, expected_value in zip(
            ("strain_mean", "stress_GPa_mean", "stress_GPa_sd", "stress_GPa_min", "stress_GPa_max"),
            expected,
        ):
            compare(errors, f"{key}:{field}", row[field], expected_value)

    result = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "errors": errors,
        "promotion_receipt": str(receipt_path),
        "promotion_receipt_sha256": digest(receipt_path),
        "replica_rows_replayed": len(replica),
        "paired_contrast_value_rows_replayed": len(expected_values),
        "paired_contrast_rows_replayed": len(expected_summaries),
        "binned_replica_curve_rows_replayed": len(binned),
        "group_curve_rows_replayed": len(replayed_groups),
        "interval_replay": "two-sided Student-t, df=2, unadjusted",
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": digest(Path(__file__).resolve()),
    }
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
