#!/usr/bin/env python3
"""Independently reconstruct the A5 phase-native static lattice table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
T975_DF4 = 2.7764451051977987
BRANCHES = tuple(
    (f"{architecture}__{phase}_parent", architecture, phase)
    for architecture in ("DPA2", "DPA3", "DPA4")
    for phase in ("bcc", "fcc")
)
NUMERIC_FIELDS = (
    "lattice_parameter_A_mean",
    "lattice_parameter_A_sample_standard_deviation",
    "lattice_parameter_A_ci95_low",
    "lattice_parameter_A_ci95_high",
    "volume_A3_per_atom_mean",
    "volume_A3_per_atom_ci95_low",
    "volume_A3_per_atom_ci95_high",
    "density_g_cm3_mean",
    "density_g_cm3_ci95_low",
    "density_g_cm3_ci95_high",
)


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


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def bound(record: dict[str, object], path_field: str, hash_field: str) -> Path:
    path = Path(str(record.get(path_field, ""))).resolve()
    if not path.is_file() or record.get(hash_field) != sha256(path):
        raise ValueError(f"hash binding failed for {path_field}")
    return path


def summarize(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = T975_DF4 * standard_deviation / math.sqrt(5.0)
    return mean, standard_deviation, mean - half_width, mean + half_width


def gate_class(row: dict[str, str]) -> str:
    reason = (row.get("eligibility_reason") or row.get("failure_reason") or row["status"]).lower()
    if "deviatoric stress" in reason:
        return "relaxation_deviatoric_stress_gate"
    if "force component" in reason:
        return "relaxation_force_gate"
    if "wall-time" in reason or "timeout" in reason:
        return "technical_wall_time_failure"
    if "dominant" in reason or row.get("status") == "failed_validation":
        return "post_run_structure_or_validation_gate"
    return "other_relaxation_failure"


def close(observed: str, expected: float) -> bool:
    try:
        return math.isclose(float(observed), expected, rel_tol=2.0e-13, abs_tol=2.0e-13)
    except (TypeError, ValueError):
        return False


def validate(table_path: Path, receipt_path: Path) -> dict[str, object]:
    table_path = table_path.resolve()
    receipt_path = receipt_path.resolve()
    receipt = read_json(receipt_path)
    errors: list[str] = []
    checks = 0
    checks += 4
    if receipt.get("status") != "passed":
        errors.append("analysis receipt did not pass")
    if Path(str(receipt.get("table", ""))).resolve() != table_path:
        errors.append("analysis receipt table path mismatch")
    if receipt.get("table_sha256") != sha256(table_path):
        errors.append("analysis receipt table hash mismatch")
    if receipt.get("external_reference_comparison") is not False:
        errors.append("static table improperly claims an external comparison")
    validation_path = bound(
        receipt, "relaxation_validation", "relaxation_validation_sha256"
    )
    analysis_receipt_path = bound(
        receipt, "relaxation_analysis_receipt", "relaxation_analysis_receipt_sha256"
    )
    replicas_path = bound(
        receipt, "relaxation_replicas", "relaxation_replicas_sha256"
    )
    checks += 3
    validation = read_json(validation_path)
    analysis_receipt = read_json(analysis_receipt_path)
    checks += 4
    if validation.get("status") != "passed" or validation.get("errors"):
        errors.append("source relaxation validation did not pass")
    if validation.get("analysis_receipt_sha256") != sha256(analysis_receipt_path):
        errors.append("source validation does not bind the relaxation receipt")
    if analysis_receipt.get("status") != "passed":
        errors.append("source relaxation analysis did not pass")
    if analysis_receipt.get("replicas_sha256") != sha256(replicas_path):
        errors.append("source analysis does not bind the replica table")

    replica_values = rows(replicas_path)
    native = [row for row in replica_values if row["training_relation"] == "phase_native"]
    output_values = rows(table_path)
    output = {(row["model_id"], row["simulated_phase"]): row for row in output_values}
    expected_keys = {(model_id, phase) for model_id, _, phase in BRANCHES}
    checks += 4
    if len(replica_values) != 60 or len(native) != 30:
        errors.append("relaxation replica source population mismatch")
    if len(output_values) != 6 or len(output) != 6 or set(output) != expected_keys:
        errors.append("static lattice table is not the exact six-branch population")

    status_counts: Counter[str] = Counter()
    for model_id, architecture, phase in BRANCHES:
        group = [
            row for row in native if row["model_id"] == model_id and row["phase"] == phase
        ]
        actual = output.get((model_id, phase))
        if actual is None:
            continue
        checks += 3
        if len(group) != 5 or len({row["chemical_seed"] for row in group}) != 5:
            errors.append(f"{model_id}/{phase}: native population mismatch")
            continue
        eligible = [row for row in group if row["eligible"] == "true"]
        balanced = len(eligible) == 5 and all(row["status"] == "passed" for row in eligible)
        expected_status = (
            "eligible_balanced" if balanced else "blocked_incomplete_or_failed_relaxation"
        )
        status_counts[expected_status] += 1
        expected_metadata = {
            "architecture": architecture,
            "training_parent": phase,
            "simulated_phase": phase,
            "natoms": "2000",
            "planned_chemical_replicas": "5",
            "eligible_chemical_replicas": str(len(eligible)),
            "status": expected_status,
            "cubic_equivalent_definition": (
                "(2*volume_A3_per_atom)^(1/3)"
                if phase == "bcc"
                else "(4*volume_A3_per_atom)^(1/3)"
            ),
            "failure_gate_class_counts": (
                ""
                if balanced
                else json.dumps(
                    dict(Counter(gate_class(item) for item in group if item["eligible"] != "true")),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "ci_method": "two-sided Student-t, df=4" if balanced else "",
            "interpretation_scope": "phase-native 0 K isotropic-cell relaxation; external comparison not performed",
        }
        for field, expected in expected_metadata.items():
            checks += 1
            if actual.get(field, "") != expected:
                errors.append(
                    f"{model_id}/{phase}: {field}={actual.get(field, '')!r}, expected {expected!r}"
                )
        if not balanced:
            for field in NUMERIC_FIELDS:
                checks += 1
                if actual.get(field, ""):
                    errors.append(f"{model_id}/{phase}: blocked row leaks {field}")
            continue

        volumes = [float(row["volume_A3_per_atom"]) for row in eligible]
        densities = [float(row["density_g_cm3"]) for row in eligible]
        multiplier = 2.0 if phase == "bcc" else 4.0
        lattice = [(multiplier * value) ** (1.0 / 3.0) for value in volumes]
        lattice_stats = summarize(lattice)
        volume_stats = summarize(volumes)
        density_stats = summarize(densities)
        expected_numeric = {
            "lattice_parameter_A_mean": lattice_stats[0],
            "lattice_parameter_A_sample_standard_deviation": lattice_stats[1],
            "lattice_parameter_A_ci95_low": lattice_stats[2],
            "lattice_parameter_A_ci95_high": lattice_stats[3],
            "volume_A3_per_atom_mean": volume_stats[0],
            "volume_A3_per_atom_ci95_low": volume_stats[2],
            "volume_A3_per_atom_ci95_high": volume_stats[3],
            "density_g_cm3_mean": density_stats[0],
            "density_g_cm3_ci95_low": density_stats[2],
            "density_g_cm3_ci95_high": density_stats[3],
        }
        for field, expected in expected_numeric.items():
            checks += 1
            if not close(actual.get(field, ""), expected):
                errors.append(
                    f"{model_id}/{phase}: {field}={actual.get(field, '')!r}, expected {expected!r}"
                )

    checks += 3
    if receipt.get("rows") != 6:
        errors.append("analysis receipt row count mismatch")
    if receipt.get("status_counts") != dict(status_counts):
        errors.append("analysis receipt status counts mismatch")
    expected_scientific = (
        "complete" if status_counts == {"eligible_balanced": 6} else "partial_with_explicit_blocks"
    )
    if receipt.get("scientific_status") != expected_scientific:
        errors.append("analysis receipt scientific status mismatch")

    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "errors": errors,
        "table": str(table_path),
        "table_sha256": sha256(table_path),
        "analysis_receipt": str(receipt_path),
        "analysis_receipt_sha256": sha256(receipt_path),
        "source_relaxation_replicas": str(replicas_path),
        "source_relaxation_replicas_sha256": sha256(replicas_path),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        type=Path,
        default=PROJECT / "tables" / "lammps_a5_static_lattice.csv",
    )
    parser.add_argument(
        "--analysis-receipt",
        type=Path,
        default=PROJECT / "work" / "lammps" / "lammps_a5_static_lattice.receipt.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "work" / "lammps" / "lammps_a5_static_lattice.validation.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = validate(args.table, args.analysis_receipt)
    atomic_json(output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
