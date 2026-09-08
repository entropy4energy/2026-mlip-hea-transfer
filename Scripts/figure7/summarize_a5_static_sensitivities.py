#!/usr/bin/env python3
"""Build complete-population 250-atom and triclinic sensitivity tables."""

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
CHEMICAL_SEEDS = (20260825, 20261834, 20262843, 20263852, 20264861)
SENSITIVITY_SEED = 20260825
SIZE_RELATIVE_LIMIT = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def atomic_csv(path: Path, values: list[dict[str, object]]) -> None:
    if not values:
        raise ValueError(f"refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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


def load_validation(
    path: Path, calculation: str, expected_rows: int
) -> tuple[dict[str, object], list[dict[str, str]], dict[str, dict[str, str]], Path, Path]:
    record = json.loads(path.read_text(encoding="utf-8"))
    plan_path = Path(str(record["plan"])).resolve()
    summary_path = Path(str(record["summary_csv"])).resolve()
    if record.get("plan_sha256") != sha256(plan_path):
        raise ValueError(f"{calculation}: validation plan hash mismatch")
    if record.get("summary_csv_sha256") != sha256(summary_path):
        raise ValueError(f"{calculation}: validation summary hash mismatch")
    if record.get("planned_tasks") != expected_rows or record.get("validated_tasks") != expected_rows:
        raise ValueError(f"{calculation}: validation does not retain {expected_rows} rows")
    plans = read_rows(plan_path, "\t")
    summaries = read_rows(summary_path)
    plan_by_id = {row["task_id"]: row for row in plans}
    summary_by_id = {row["task_id"]: row for row in summaries}
    if (
        len(plan_by_id) != expected_rows
        or len(summary_by_id) != expected_rows
        or set(plan_by_id) != set(summary_by_id)
        or {row["calculation"] for row in plans} != {calculation}
        or {row["calculation"] for row in summaries} != {calculation}
    ):
        raise ValueError(f"{calculation}: plan/summary population mismatch")
    return record, summaries, plan_by_id, plan_path, summary_path


def key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["model_id"], row["phase"], int(row["chemical_seed"])


def plan_key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["model_id"], row["simulated_phase"], int(row["chemical_seed"])


def verified_raw_summary(plan: dict[str, str]) -> dict[str, str]:
    root = Path(plan["output_dir"])
    raw_path = root / "relax_summary.csv"
    index_path = root / "run_outputs.tsv"
    receipt_path = root / "completion_receipt.json"
    marker_path = root / "run.complete"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed" or receipt.get("lammps_return_code") != 0:
        raise ValueError(f"{plan['task_id']}: completion receipt did not pass")
    if receipt.get("output_index_sha256") != sha256(index_path):
        raise ValueError(f"{plan['task_id']}: completion receipt does not bind output index")
    if marker.get("completion_receipt_sha256") != sha256(receipt_path):
        raise ValueError(f"{plan['task_id']}: run.complete does not bind completion receipt")
    indexed = {row["relative_path"]: row for row in read_rows(index_path, "\t")}
    if "relax_summary.csv" not in indexed or indexed["relax_summary.csv"]["sha256"] != sha256(raw_path):
        raise ValueError(f"{plan['task_id']}: relax_summary.csv is not output-index-bound")
    values = read_rows(raw_path)
    if len(values) != 1:
        raise ValueError(f"{plan['task_id']}: raw relaxation summary is not one row")
    return values[0]


def cell_metrics(raw: dict[str, str]) -> dict[str, float]:
    lx, ly, lz = (float(raw[field]) for field in ("lx_A", "ly_A", "lz_A"))
    xy, xz, yz = (float(raw[field]) for field in ("xy_A", "xz_A", "yz_A"))
    a = (lx, 0.0, 0.0)
    b = (xy, ly, 0.0)
    c = (xz, yz, lz)

    def norm(vector: tuple[float, float, float]) -> float:
        return math.sqrt(sum(value * value for value in vector))

    def angle(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
        cosine = sum(x * y for x, y in zip(first, second, strict=True)) / (norm(first) * norm(second))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    lengths = (norm(a), norm(b), norm(c))
    return {
        "cell_a_A": lengths[0],
        "cell_b_A": lengths[1],
        "cell_c_A": lengths[2],
        "cell_alpha_deg": angle(b, c),
        "cell_beta_deg": angle(a, c),
        "cell_gamma_deg": angle(a, b),
        "maximum_tilt_over_box_length": max(
            abs(xy) / lx, abs(xz) / lx, abs(yz) / ly
        ),
        "length_anisotropy_fraction": max(lengths) / min(lengths) - 1.0,
    }


def relative_difference(sensitivity: float, primary: float) -> float:
    if primary == 0.0:
        raise ValueError("relative difference has a zero primary reference")
    return (sensitivity - primary) / abs(primary)


def build_size_rows(
    primary: dict[tuple[str, str, int], dict[str, str]],
    size: dict[tuple[str, str, int], dict[str, str]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for model_id, phase in sorted(MODEL_PHASE.items()):
        primary_group = [primary[(model_id, phase, seed)] for seed in CHEMICAL_SEEDS]
        reference = primary[(model_id, phase, SENSITIVITY_SEED)]
        sensitivity = size[(model_id, phase, SENSITIVITY_SEED)]
        reference_complete = all(row["status"] == "passed" for row in primary_group)
        pair_passed = reference["status"] == "passed" and sensitivity["status"] == "passed"
        row: dict[str, object] = {
            "model_id": model_id,
            "architecture": model_id.split("__", 1)[0],
            "training_parent": phase,
            "simulated_phase": phase,
            "chemical_seed": SENSITIVITY_SEED,
            "primary_natoms": 2000,
            "sensitivity_natoms": 250,
            "primary_status": reference["status"],
            "primary_five_decoration_status": (
                "complete" if reference_complete else "blocked_incomplete"
            ),
            "sensitivity_status": sensitivity["status"],
            "energy_spread_definition": "max_minus_min_across_five_2000_atom_decorations",
            "volume_density_relative_limit": SIZE_RELATIVE_LIMIT,
            "finite_temperature_size_status": "not_evaluated_static_only",
            "static_size_claim_status": "blocked_incomplete_primary_or_sensitivity",
        }
        if reference_complete and pair_passed:
            energies = [float(item["energy_eV_per_atom"]) for item in primary_group]
            energy_spread = max(energies) - min(energies)
            primary_energy = float(reference["energy_eV_per_atom"])
            size_energy = float(sensitivity["energy_eV_per_atom"])
            primary_volume = float(reference["volume_A3_per_atom"])
            size_volume = float(sensitivity["volume_A3_per_atom"])
            primary_density = float(reference["density_g_cm3"])
            size_density = float(sensitivity["density_g_cm3"])
            energy_difference = size_energy - primary_energy
            volume_relative = relative_difference(size_volume, primary_volume)
            density_relative = relative_difference(size_density, primary_density)
            energy_passed = abs(energy_difference) <= energy_spread + 1.0e-15
            volume_passed = abs(volume_relative) <= SIZE_RELATIVE_LIMIT + 1.0e-15
            density_passed = abs(density_relative) <= SIZE_RELATIVE_LIMIT + 1.0e-15
            size_passed = energy_passed and volume_passed and density_passed
            row.update(
                {
                    "primary_energy_eV_per_atom": primary_energy,
                    "sensitivity_energy_eV_per_atom": size_energy,
                    "energy_250_minus_2000_eV_per_atom": energy_difference,
                    "five_decoration_energy_range_eV_per_atom": energy_spread,
                    "energy_size_gate_passed": str(energy_passed).lower(),
                    "primary_volume_A3_per_atom": primary_volume,
                    "sensitivity_volume_A3_per_atom": size_volume,
                    "volume_relative_difference": volume_relative,
                    "volume_size_gate_passed": str(volume_passed).lower(),
                    "primary_density_g_cm3": primary_density,
                    "sensitivity_density_g_cm3": size_density,
                    "density_relative_difference": density_relative,
                    "density_size_gate_passed": str(density_passed).lower(),
                    "static_size_claim_status": (
                        "eligible_size_insensitive" if size_passed else "blocked_detected_size_effect"
                    ),
                }
            )
        output.append(row)
    return output


def build_triclinic_rows(
    primary: dict[tuple[str, str, int], dict[str, str]],
    tri: dict[tuple[str, str, int], dict[str, str]],
    tri_plans: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    plan_by_key = {plan_key(plan): plan for plan in tri_plans.values()}
    output: list[dict[str, object]] = []
    expected = {
        (f"{architecture}__{parent}_parent", phase, SENSITIVITY_SEED)
        for architecture in ARCHITECTURES
        for parent in PHASES
        for phase in PHASES
    }
    if set(tri) != expected or set(plan_by_key) != expected:
        raise ValueError("triclinic population is not the exact 12 model/phase rows")
    for item in sorted(expected):
        model_id, phase, seed = item
        reference = primary[item]
        sensitivity = tri[item]
        relation = "phase_native" if MODEL_PHASE[model_id] == phase else "reciprocal_alarm"
        row: dict[str, object] = {
            "model_id": model_id,
            "architecture": model_id.split("__", 1)[0],
            "training_parent": MODEL_PHASE[model_id],
            "simulated_phase": phase,
            "training_relation": relation,
            "chemical_seed": seed,
            "natoms": 2000,
            "isotropic_status": reference["status"],
            "triclinic_status": sensitivity["status"],
            "sensitivity_role": "diagnostic_not_a_substitute_for_failed_isotropic_reference",
            "comparison_status": "blocked_primary_or_triclinic_failure",
        }
        if reference["status"] == "passed" and sensitivity["status"] == "passed":
            raw = verified_raw_summary(plan_by_key[item])
            metrics = cell_metrics(raw)
            primary_energy = float(reference["energy_eV_per_atom"])
            tri_energy = float(sensitivity["energy_eV_per_atom"])
            primary_volume = float(reference["volume_A3_per_atom"])
            tri_volume = float(sensitivity["volume_A3_per_atom"])
            row.update(
                {
                    "isotropic_energy_eV_per_atom": primary_energy,
                    "triclinic_energy_eV_per_atom": tri_energy,
                    "triclinic_minus_isotropic_energy_eV_per_atom": tri_energy - primary_energy,
                    "isotropic_volume_A3_per_atom": primary_volume,
                    "triclinic_volume_A3_per_atom": tri_volume,
                    "triclinic_minus_isotropic_volume_relative": relative_difference(
                        tri_volume, primary_volume
                    ),
                    "isotropic_expected_phase_fraction": reference[f"fraction_{phase}"],
                    "triclinic_expected_phase_fraction": sensitivity[f"fraction_{phase}"],
                    "triclinic_max_force_eV_per_A": sensitivity["max_force_eV_per_A"],
                    "triclinic_pressure_error_GPa": sensitivity["pressure_error_GPa"],
                    "triclinic_max_stress_error_GPa": sensitivity["max_stress_error_GPa"],
                    **metrics,
                    "comparison_status": "eligible_diagnostic",
                }
            )
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-validation", type=Path, required=True)
    parser.add_argument("--size-validation", type=Path, required=True)
    parser.add_argument("--triclinic-validation", type=Path, required=True)
    parser.add_argument("--size-output", type=Path, required=True)
    parser.add_argument("--triclinic-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    primary_path = args.primary_validation.resolve()
    size_path = args.size_validation.resolve()
    tri_path = args.triclinic_validation.resolve()
    _, primary_rows, _, primary_plan, primary_summary = load_validation(
        primary_path, "relax_iso_coupled", 60
    )
    _, size_rows, _, size_plan, size_summary = load_validation(
        size_path, "relax_size_coupled", 6
    )
    _, tri_rows, tri_plans, tri_plan, tri_summary = load_validation(
        tri_path, "relax_tri_coupled", 12
    )
    primary = {key(row): row for row in primary_rows}
    size = {key(row): row for row in size_rows}
    tri = {key(row): row for row in tri_rows}
    expected_primary = {
        (f"{architecture}__{parent}_parent", phase, seed)
        for architecture in ARCHITECTURES
        for parent in PHASES
        for phase in PHASES
        for seed in CHEMICAL_SEEDS
    }
    expected_size = {
        (model_id, phase, SENSITIVITY_SEED) for model_id, phase in MODEL_PHASE.items()
    }
    if set(primary) != expected_primary or set(size) != expected_size:
        raise ValueError("primary or size sensitivity population is invalid")
    size_output = args.size_output.resolve()
    tri_output = args.triclinic_output.resolve()
    receipt_path = args.receipt.resolve()
    for path in (size_output, tri_output, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    size_values = build_size_rows(primary, size)
    tri_values = build_triclinic_rows(primary, tri, tri_plans)
    atomic_csv(size_output, size_values)
    atomic_csv(tri_output, tri_values)
    size_counts = Counter(str(row["static_size_claim_status"]) for row in size_values)
    tri_counts = Counter(str(row["comparison_status"]) for row in tri_values)
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "scientific_status": "complete_population_with_explicit_blocks",
        "primary_validation": str(primary_path),
        "primary_validation_sha256": sha256(primary_path),
        "primary_plan": str(primary_plan),
        "primary_plan_sha256": sha256(primary_plan),
        "primary_summary": str(primary_summary),
        "primary_summary_sha256": sha256(primary_summary),
        "size_validation": str(size_path),
        "size_validation_sha256": sha256(size_path),
        "size_plan": str(size_plan),
        "size_plan_sha256": sha256(size_plan),
        "size_summary": str(size_summary),
        "size_summary_sha256": sha256(size_summary),
        "triclinic_validation": str(tri_path),
        "triclinic_validation_sha256": sha256(tri_path),
        "triclinic_plan": str(tri_plan),
        "triclinic_plan_sha256": sha256(tri_plan),
        "triclinic_summary": str(tri_summary),
        "triclinic_summary_sha256": sha256(tri_summary),
        "size_output": str(size_output),
        "size_output_sha256": sha256(size_output),
        "size_rows": len(size_values),
        "size_status_counts": dict(size_counts),
        "triclinic_output": str(tri_output),
        "triclinic_output_sha256": sha256(tri_output),
        "triclinic_rows": len(tri_values),
        "triclinic_status_counts": dict(tri_counts),
        "energy_spread_definition": "max-minus-min across the five fixed 2,000-atom chemical decorations",
        "finite_temperature_size_sensitivity": "not_evaluated_by_these_static_runs",
        "analysis_program": str(Path(__file__).resolve()),
        "analysis_program_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
