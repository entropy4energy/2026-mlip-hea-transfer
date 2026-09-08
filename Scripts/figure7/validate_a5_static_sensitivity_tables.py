#!/usr/bin/env python3
"""Independently reconstruct A5 static size/triclinic sensitivity tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path


ARCHITECTURES = ("DPA2", "DPA3", "DPA4")
PHASES = ("bcc", "fcc")
MODEL_PHASE = {
    f"{architecture}__{phase}_parent": phase
    for architecture in ARCHITECTURES
    for phase in PHASES
}
SEEDS = (20260825, 20261834, 20262843, 20263852, 20264861)
SENSITIVITY_SEED = 20260825
RELATIVE_LIMIT = 0.02
NUMERIC_TOLERANCE = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


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


def row_key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["model_id"], row["phase"], int(row["chemical_seed"])


def plan_key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["model_id"], row["simulated_phase"], int(row["chemical_seed"])


def table_key(row: dict[str, str]) -> tuple[str, str]:
    return row["model_id"], row["simulated_phase"]


def load_validation(
    path: Path, calculation: str, expected: int, errors: list[str]
) -> tuple[dict[tuple[str, str, int], dict[str, str]], dict[tuple[str, str, int], dict[str, str]]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    plan_path = Path(str(record["plan"])).resolve()
    summary_path = Path(str(record["summary_csv"])).resolve()
    if record.get("plan_sha256") != sha256(plan_path):
        errors.append(f"{calculation}: validation plan hash mismatch")
    if record.get("summary_csv_sha256") != sha256(summary_path):
        errors.append(f"{calculation}: validation summary hash mismatch")
    if record.get("planned_tasks") != expected or record.get("validated_tasks") != expected:
        errors.append(f"{calculation}: validation population mismatch")
    plan_values = rows(plan_path, "\t")
    summary_values = rows(summary_path)
    plan = {plan_key(row): row for row in plan_values}
    summary = {row_key(row): row for row in summary_values}
    if (
        len(plan) != expected
        or len(summary) != expected
        or set(plan) != set(summary)
        or {row.get("calculation") for row in plan_values} != {calculation}
        or {row.get("calculation") for row in summary_values} != {calculation}
    ):
        errors.append(f"{calculation}: plan/summary rows are invalid")
    return plan, summary


def number(row: dict[str, str], field: str, expected: float, errors: list[str], label: str) -> None:
    try:
        value = float(row.get(field, ""))
    except ValueError:
        errors.append(f"{label}: {field} is not numeric")
        return
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=NUMERIC_TOLERANCE, abs_tol=NUMERIC_TOLERANCE
    ):
        errors.append(f"{label}: {field}={value!r}, expected {expected!r}")


def blank(row: dict[str, str], fields: tuple[str, ...], errors: list[str], label: str) -> None:
    for field in fields:
        if row.get(field, "") != "":
            errors.append(f"{label}: blocked row contains {field}")


def read_bound_raw(plan: dict[str, str], errors: list[str]) -> dict[str, str] | None:
    root = Path(plan["output_dir"])
    raw = root / "relax_summary.csv"
    index = root / "run_outputs.tsv"
    receipt_path = root / "completion_receipt.json"
    marker_path = root / "run.complete"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        indexed = {row["relative_path"]: row for row in rows(index, "\t")}
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"{plan['task_id']}: cannot replay raw output binding: {error}")
        return None
    if receipt.get("status") != "passed" or receipt.get("lammps_return_code") != 0:
        errors.append(f"{plan['task_id']}: completion receipt did not pass")
    if receipt.get("output_index_sha256") != sha256(index):
        errors.append(f"{plan['task_id']}: output index is not receipt-bound")
    if marker.get("completion_receipt_sha256") != sha256(receipt_path):
        errors.append(f"{plan['task_id']}: run marker is not receipt-bound")
    if indexed.get("relax_summary.csv", {}).get("sha256") != sha256(raw):
        errors.append(f"{plan['task_id']}: raw summary is not output-index-bound")
    values = rows(raw)
    if len(values) != 1:
        errors.append(f"{plan['task_id']}: raw summary does not contain one row")
        return None
    return values[0]


def reconstruct_cell(raw: dict[str, str]) -> dict[str, float]:
    lx, ly, lz = (float(raw[field]) for field in ("lx_A", "ly_A", "lz_A"))
    xy, xz, yz = (float(raw[field]) for field in ("xy_A", "xz_A", "yz_A"))
    vectors = ((lx, 0.0, 0.0), (xy, ly, 0.0), (xz, yz, lz))
    lengths = [math.sqrt(sum(component * component for component in vector)) for vector in vectors]

    def vector_angle(i: int, j: int) -> float:
        dot = sum(a * b for a, b in zip(vectors[i], vectors[j], strict=True))
        cosine = dot / (lengths[i] * lengths[j])
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    return {
        "cell_a_A": lengths[0],
        "cell_b_A": lengths[1],
        "cell_c_A": lengths[2],
        "cell_alpha_deg": vector_angle(1, 2),
        "cell_beta_deg": vector_angle(0, 2),
        "cell_gamma_deg": vector_angle(0, 1),
        "maximum_tilt_over_box_length": max(abs(xy) / lx, abs(xz) / lx, abs(yz) / ly),
        "length_anisotropy_fraction": max(lengths) / min(lengths) - 1.0,
    }


def validate_size(
    actual_values: list[dict[str, str]],
    primary: dict[tuple[str, str, int], dict[str, str]],
    size: dict[tuple[str, str, int], dict[str, str]],
    errors: list[str],
) -> Counter[str]:
    actual = {table_key(row): row for row in actual_values}
    expected_keys = set(MODEL_PHASE.items())
    if len(actual_values) != 6 or len(actual) != 6 or set(actual) != expected_keys:
        errors.append("size table is not the exact six phase-native rows")
    statuses: Counter[str] = Counter()
    numeric_fields = (
        "primary_energy_eV_per_atom",
        "sensitivity_energy_eV_per_atom",
        "energy_250_minus_2000_eV_per_atom",
        "five_decoration_energy_range_eV_per_atom",
        "primary_volume_A3_per_atom",
        "sensitivity_volume_A3_per_atom",
        "volume_relative_difference",
        "primary_density_g_cm3",
        "sensitivity_density_g_cm3",
        "density_relative_difference",
    )
    for model_id, phase in sorted(expected_keys):
        if (model_id, phase) not in actual:
            continue
        row = actual[(model_id, phase)]
        label = f"size {model_id}/{phase}"
        group = [primary[(model_id, phase, seed)] for seed in SEEDS]
        reference = primary[(model_id, phase, SENSITIVITY_SEED)]
        sensitivity = size[(model_id, phase, SENSITIVITY_SEED)]
        if row.get("primary_status") != reference["status"] or row.get(
            "sensitivity_status"
        ) != sensitivity["status"]:
            errors.append(f"{label}: source status mismatch")
        complete = all(item["status"] == "passed" for item in group)
        pair_passed = reference["status"] == "passed" and sensitivity["status"] == "passed"
        if row.get("primary_five_decoration_status") != (
            "complete" if complete else "blocked_incomplete"
        ):
            errors.append(f"{label}: five-decoration status mismatch")
        if row.get("finite_temperature_size_status") != "not_evaluated_static_only":
            errors.append(f"{label}: static table overstates finite-temperature size validation")
        if complete and pair_passed:
            energies = [float(item["energy_eV_per_atom"]) for item in group]
            energy_range = max(energies) - min(energies)
            e0, e1 = float(reference["energy_eV_per_atom"]), float(sensitivity["energy_eV_per_atom"])
            v0, v1 = float(reference["volume_A3_per_atom"]), float(sensitivity["volume_A3_per_atom"])
            d0, d1 = float(reference["density_g_cm3"]), float(sensitivity["density_g_cm3"])
            values = {
                "primary_energy_eV_per_atom": e0,
                "sensitivity_energy_eV_per_atom": e1,
                "energy_250_minus_2000_eV_per_atom": e1 - e0,
                "five_decoration_energy_range_eV_per_atom": energy_range,
                "primary_volume_A3_per_atom": v0,
                "sensitivity_volume_A3_per_atom": v1,
                "volume_relative_difference": (v1 - v0) / abs(v0),
                "primary_density_g_cm3": d0,
                "sensitivity_density_g_cm3": d1,
                "density_relative_difference": (d1 - d0) / abs(d0),
            }
            for field, expected in values.items():
                number(row, field, expected, errors, label)
            gates = {
                "energy_size_gate_passed": abs(e1 - e0) <= energy_range + 1.0e-15,
                "volume_size_gate_passed": abs((v1 - v0) / abs(v0)) <= RELATIVE_LIMIT + 1.0e-15,
                "density_size_gate_passed": abs((d1 - d0) / abs(d0)) <= RELATIVE_LIMIT + 1.0e-15,
            }
            for field, expected in gates.items():
                if row.get(field) != str(expected).lower():
                    errors.append(f"{label}: {field} mismatch")
            status = (
                "eligible_size_insensitive"
                if all(gates.values())
                else "blocked_detected_size_effect"
            )
        else:
            blank(row, numeric_fields, errors, label)
            status = "blocked_incomplete_primary_or_sensitivity"
        if row.get("static_size_claim_status") != status:
            errors.append(f"{label}: static size status mismatch")
        statuses[status] += 1
    return statuses


def validate_triclinic(
    actual_values: list[dict[str, str]],
    primary: dict[tuple[str, str, int], dict[str, str]],
    tri: dict[tuple[str, str, int], dict[str, str]],
    tri_plans: dict[tuple[str, str, int], dict[str, str]],
    errors: list[str],
) -> Counter[str]:
    actual = {table_key(row): row for row in actual_values}
    expected_keys = {
        (f"{architecture}__{parent}_parent", phase)
        for architecture in ARCHITECTURES
        for parent in PHASES
        for phase in PHASES
    }
    if len(actual_values) != 12 or len(actual) != 12 or set(actual) != expected_keys:
        errors.append("triclinic table is not the exact 12 model/phase rows")
    statuses: Counter[str] = Counter()
    promoted_fields = (
        "isotropic_energy_eV_per_atom",
        "triclinic_energy_eV_per_atom",
        "triclinic_minus_isotropic_energy_eV_per_atom",
        "isotropic_volume_A3_per_atom",
        "triclinic_volume_A3_per_atom",
        "triclinic_minus_isotropic_volume_relative",
        "isotropic_expected_phase_fraction",
        "triclinic_expected_phase_fraction",
        "cell_a_A",
        "cell_b_A",
        "cell_c_A",
        "cell_alpha_deg",
        "cell_beta_deg",
        "cell_gamma_deg",
        "maximum_tilt_over_box_length",
        "length_anisotropy_fraction",
    )
    for model_id, phase in sorted(expected_keys):
        if (model_id, phase) not in actual:
            continue
        item = (model_id, phase, SENSITIVITY_SEED)
        row = actual[(model_id, phase)]
        reference, sensitivity = primary[item], tri[item]
        label = f"triclinic {model_id}/{phase}"
        if row.get("isotropic_status") != reference["status"] or row.get(
            "triclinic_status"
        ) != sensitivity["status"]:
            errors.append(f"{label}: source status mismatch")
        eligible = reference["status"] == "passed" and sensitivity["status"] == "passed"
        if eligible:
            raw = read_bound_raw(tri_plans[item], errors)
            if raw is not None:
                e0, e1 = float(reference["energy_eV_per_atom"]), float(sensitivity["energy_eV_per_atom"])
                v0, v1 = float(reference["volume_A3_per_atom"]), float(sensitivity["volume_A3_per_atom"])
                values = {
                    "isotropic_energy_eV_per_atom": e0,
                    "triclinic_energy_eV_per_atom": e1,
                    "triclinic_minus_isotropic_energy_eV_per_atom": e1 - e0,
                    "isotropic_volume_A3_per_atom": v0,
                    "triclinic_volume_A3_per_atom": v1,
                    "triclinic_minus_isotropic_volume_relative": (v1 - v0) / abs(v0),
                    "isotropic_expected_phase_fraction": float(reference[f"fraction_{phase}"]),
                    "triclinic_expected_phase_fraction": float(sensitivity[f"fraction_{phase}"]),
                    **reconstruct_cell(raw),
                }
                for field, expected in values.items():
                    number(row, field, expected, errors, label)
            status = "eligible_diagnostic"
        else:
            blank(row, promoted_fields, errors, label)
            status = "blocked_primary_or_triclinic_failure"
        if row.get("comparison_status") != status:
            errors.append(f"{label}: comparison status mismatch")
        if row.get("sensitivity_role") != "diagnostic_not_a_substitute_for_failed_isotropic_reference":
            errors.append(f"{label}: sensitivity role mismatch")
        statuses[status] += 1
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-table", type=Path, required=True)
    parser.add_argument("--triclinic-table", type=Path, required=True)
    parser.add_argument("--analysis-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    size_path = args.size_table.resolve()
    tri_path = args.triclinic_table.resolve()
    receipt_path = args.analysis_receipt.resolve()
    output = args.output.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if receipt.get("status") != "passed":
        errors.append("analysis receipt status is not passed")
    if Path(str(receipt.get("size_output", ""))).resolve() != size_path or receipt.get(
        "size_output_sha256"
    ) != sha256(size_path):
        errors.append("size table is not analysis-receipt-bound")
    if Path(str(receipt.get("triclinic_output", ""))).resolve() != tri_path or receipt.get(
        "triclinic_output_sha256"
    ) != sha256(tri_path):
        errors.append("triclinic table is not analysis-receipt-bound")
    analysis_program = Path(str(receipt.get("analysis_program", ""))).resolve()
    if not analysis_program.is_file() or receipt.get("analysis_program_sha256") != sha256(
        analysis_program
    ):
        errors.append("analysis program hash binding failed")
    primary_path = Path(str(receipt["primary_validation"])).resolve()
    size_validation_path = Path(str(receipt["size_validation"])).resolve()
    tri_validation_path = Path(str(receipt["triclinic_validation"])).resolve()
    for path, field in (
        (primary_path, "primary_validation_sha256"),
        (size_validation_path, "size_validation_sha256"),
        (tri_validation_path, "triclinic_validation_sha256"),
    ):
        if receipt.get(field) != sha256(path):
            errors.append(f"receipt source binding failed: {field}")
    _, primary = load_validation(primary_path, "relax_iso_coupled", 60, errors)
    _, size = load_validation(size_validation_path, "relax_size_coupled", 6, errors)
    tri_plans, tri = load_validation(tri_validation_path, "relax_tri_coupled", 12, errors)
    expected_primary = {
        (f"{architecture}__{parent}_parent", phase, seed)
        for architecture in ARCHITECTURES
        for parent in PHASES
        for phase in PHASES
        for seed in SEEDS
    }
    expected_size = {(model_id, phase, SENSITIVITY_SEED) for model_id, phase in MODEL_PHASE.items()}
    expected_tri = {
        (f"{architecture}__{parent}_parent", phase, SENSITIVITY_SEED)
        for architecture in ARCHITECTURES
        for parent in PHASES
        for phase in PHASES
    }
    if set(primary) != expected_primary or set(size) != expected_size or set(tri) != expected_tri:
        errors.append("source validation populations are not exact")
    size_statuses = validate_size(rows(size_path), primary, size, errors)
    tri_statuses = validate_triclinic(rows(tri_path), primary, tri, tri_plans, errors)
    if receipt.get("size_rows") != 6 or receipt.get("size_status_counts") != dict(size_statuses):
        errors.append("analysis receipt size counts mismatch")
    if receipt.get("triclinic_rows") != 12 or receipt.get("triclinic_status_counts") != dict(
        tri_statuses
    ):
        errors.append("analysis receipt triclinic counts mismatch")
    result = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "size_table": str(size_path),
        "size_table_sha256": sha256(size_path),
        "triclinic_table": str(tri_path),
        "triclinic_table_sha256": sha256(tri_path),
        "analysis_receipt": str(receipt_path),
        "analysis_receipt_sha256": sha256(receipt_path),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
