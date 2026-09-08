#!/usr/bin/env python3
"""Summarize phase-native static lattice parameters from five decorations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
T975_DF4 = 2.7764451051977987
BRANCHES = tuple(
    (f"{architecture}__{phase}_parent", architecture, phase)
    for architecture in ("DPA2", "DPA3", "DPA4")
    for phase in ("bcc", "fcc")
)
FIELDS = (
    "model_id",
    "architecture",
    "training_parent",
    "simulated_phase",
    "natoms",
    "planned_chemical_replicas",
    "eligible_chemical_replicas",
    "status",
    "cubic_equivalent_definition",
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
    "failure_gate_class_counts",
    "ci_method",
    "interpretation_scope",
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


def atomic_csv(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


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


def bind_replicas(validation_path: Path) -> tuple[Path, Path]:
    validation = read_json(validation_path)
    if validation.get("status") != "passed" or validation.get("errors"):
        raise ValueError("relaxation tables did not pass independent validation")
    receipt_path = Path(str(validation.get("analysis_receipt", ""))).resolve()
    if not receipt_path.is_file() or validation.get("analysis_receipt_sha256") != sha256(
        receipt_path
    ):
        raise ValueError("relaxation-validation receipt binding failed")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "passed":
        raise ValueError("relaxation analysis receipt did not pass")
    replicas = Path(str(receipt.get("replicas", ""))).resolve()
    if not replicas.is_file() or receipt.get("replicas_sha256") != sha256(replicas):
        raise ValueError("relaxation replica-table binding failed")
    return receipt_path, replicas


def lattice_parameter(volume_per_atom: float, phase: str) -> float:
    atoms_per_conventional_cell = 2.0 if phase == "bcc" else 4.0
    return (atoms_per_conventional_cell * volume_per_atom) ** (1.0 / 3.0)


def estimate(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 5 or not all(math.isfinite(value) for value in values):
        raise ValueError("balanced static estimate requires five finite chemical replicas")
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    half_width = T975_DF4 * standard_deviation / math.sqrt(5.0)
    return mean, standard_deviation, mean - half_width, mean + half_width


def failure_class(row: dict[str, str]) -> str:
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


def build_rows(replica_values: list[dict[str, str]]) -> list[dict[str, object]]:
    native = [row for row in replica_values if row["training_relation"] == "phase_native"]
    output: list[dict[str, object]] = []
    for model_id, architecture, phase in BRANCHES:
        group = [
            row
            for row in native
            if row["model_id"] == model_id and row["phase"] == phase
        ]
        if len(group) != 5 or len({row["chemical_seed"] for row in group}) != 5:
            raise ValueError(f"{model_id}/{phase}: not the exact five-chemistry population")
        eligible = [row for row in group if row["eligible"] == "true"]
        balanced = len(eligible) == 5 and all(row["status"] == "passed" for row in eligible)
        definition = "(2*volume_A3_per_atom)^(1/3)" if phase == "bcc" else "(4*volume_A3_per_atom)^(1/3)"
        row: dict[str, object] = {
            "model_id": model_id,
            "architecture": architecture,
            "training_parent": phase,
            "simulated_phase": phase,
            "natoms": 2000,
            "planned_chemical_replicas": 5,
            "eligible_chemical_replicas": len(eligible),
            "status": "eligible_balanced" if balanced else "blocked_incomplete_or_failed_relaxation",
            "cubic_equivalent_definition": definition,
            "failure_gate_class_counts": json.dumps(
                dict(Counter(failure_class(item) for item in group if item["eligible"] != "true")),
                sort_keys=True,
                separators=(",", ":"),
            )
            if not balanced
            else "",
            "ci_method": "two-sided Student-t, df=4" if balanced else "",
            "interpretation_scope": "phase-native 0 K isotropic-cell relaxation; external comparison not performed",
        }
        numeric_fields = (
            "lattice_parameter_A",
            "volume_A3_per_atom",
            "density_g_cm3",
        )
        if balanced:
            values = {
                "lattice_parameter_A": [
                    lattice_parameter(float(item["volume_A3_per_atom"]), phase)
                    for item in eligible
                ],
                "volume_A3_per_atom": [
                    float(item["volume_A3_per_atom"]) for item in eligible
                ],
                "density_g_cm3": [float(item["density_g_cm3"]) for item in eligible],
            }
            for field in numeric_fields:
                mean, standard_deviation, lower, upper = estimate(values[field])
                row[f"{field}_mean"] = mean
                if field == "lattice_parameter_A":
                    row[f"{field}_sample_standard_deviation"] = standard_deviation
                row[f"{field}_ci95_low"] = lower
                row[f"{field}_ci95_high"] = upper
        else:
            for field in numeric_fields:
                row[f"{field}_mean"] = ""
                if field == "lattice_parameter_A":
                    row[f"{field}_sample_standard_deviation"] = ""
                row[f"{field}_ci95_low"] = ""
                row[f"{field}_ci95_high"] = ""
        output.append(row)
    return output


def build(validation_path: Path, output_path: Path, receipt_path: Path) -> dict[str, object]:
    validation_path = validation_path.resolve()
    output_path = output_path.resolve()
    receipt_path = receipt_path.resolve()
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite a static-lattice table or receipt")
    analysis_receipt, replicas = bind_replicas(validation_path)
    output_rows = build_rows(rows(replicas))
    atomic_csv(output_path, output_rows)
    builder = Path(__file__).resolve()
    status_counts = Counter(str(row["status"]) for row in output_rows)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "scientific_status": (
            "complete" if status_counts == {"eligible_balanced": 6} else "partial_with_explicit_blocks"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "relaxation_validation": str(validation_path),
        "relaxation_validation_sha256": sha256(validation_path),
        "relaxation_analysis_receipt": str(analysis_receipt),
        "relaxation_analysis_receipt_sha256": sha256(analysis_receipt),
        "relaxation_replicas": str(replicas),
        "relaxation_replicas_sha256": sha256(replicas),
        "table": str(output_path),
        "table_sha256": sha256(output_path),
        "rows": 6,
        "status_counts": dict(status_counts),
        "external_reference_comparison": False,
        "builder": str(builder),
        "builder_sha256": sha256(builder),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relaxation-validation",
        type=Path,
        default=PROJECT / "work" / "lammps" / "lammps_relaxation_tables.validation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "tables" / "lammps_a5_static_lattice.csv",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=PROJECT / "work" / "lammps" / "lammps_a5_static_lattice.receipt.json",
    )
    args = parser.parse_args()
    receipt = build(args.relaxation_validation, args.output, args.receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
