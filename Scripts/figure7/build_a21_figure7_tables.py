#!/usr/bin/env python3
"""Build A21 Figure 7 tables with DPA4/BCC 1200 K and no FCC curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import build_a19_figure7_tables as a19


CHEMICAL_SEEDS = a19.CHEMICAL_SEEDS
TEMPERATURES = a19.TEMPERATURES
METRICS = a19.METRICS
CURVE_BRANCHES = (
    ("DPA2__bcc_parent", "DPA2", "bcc", "bcc"),
    ("DPA3__bcc_parent", "DPA3", "bcc", "bcc"),
    ("DPA4__bcc_parent", "DPA4", "bcc", "bcc"),
)
EXCLUDED_CURVE_BRANCH = ("DPA4__fcc_parent", "DPA4", "fcc", "fcc")
PANEL_BRANCHES = a19.PANEL_BRANCHES
A21_COHORT = "dpa4_bcc_length_resolution_a21"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_a19_binding(validation_path: Path, endpoint_path: Path) -> dict[str, object]:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or validation.get("errors") not in ([], None):
        raise ValueError("A19 independent table validation is not passed")
    bindings = validation.get("validated_outputs")
    if not isinstance(bindings, dict):
        raise ValueError("A19 validation lacks output bindings")
    expected = bindings.get(str(endpoint_path.resolve()))
    if expected != sha256(endpoint_path):
        raise ValueError("A19 endpoint table is not hash-bound by its validator")
    return validation


def a21_endpoints(plan_path: Path) -> list[dict[str, object]]:
    receipt_path = plan_path.with_suffix(plan_path.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("plan_sha256") != sha256(plan_path):
        raise ValueError("A21 plan receipt/hash mismatch")
    plans = a19.read_csv(plan_path, delimiter="\t")
    if len(plans) != 3 or {int(row["chemical_seed"]) for row in plans} != set(CHEMICAL_SEEDS):
        raise ValueError("A21 plan is not the exact three-chemistry population")
    if any(
        row["cohort"] != A21_COHORT
        or row["model_id"] != "DPA4__bcc_parent"
        or row["phase"] != "bcc"
        or int(row["temperature_K"]) != 1200
        or int(row["velocity_seed"]) != 20260905
        or int(row["n_prod"]) != 100000
        for row in plans
    ):
        raise ValueError("A21 plan differs from the frozen DPA4/BCC contract")

    output: list[dict[str, object]] = []
    for plan in plans:
        root = Path(plan["output_root"]).resolve()
        production = root / "production"
        validation_path = production / "scientific_validation.json"
        if validation_path.is_file():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            status = str(validation.get("status", "failed"))
            errors = "; ".join(str(value) for value in validation.get("errors", []))
        elif (production / "run.failed.json").is_file():
            validation = {}
            status = "failed_before_scientific_validation"
            errors = "LAMMPS/finalizer failure"
        elif root.exists():
            validation = {}
            status = "incomplete"
            errors = "terminal scientific validation is absent"
        else:
            validation = {}
            status = "not_started"
            errors = "run directory is absent"
        trajectory = production / "trajectory_npt.lammpstrj"
        output.append(
            {
                "task_id": plan["task_id"],
                "site": plan["site"],
                "cohort": plan["cohort"],
                "model_id": plan["model_id"],
                "architecture": plan["architecture"],
                "training_parent": plan["training_parent"],
                "simulated_phase": plan["phase"],
                "chemical_seed": int(plan["chemical_seed"]),
                "velocity_seed": int(plan["velocity_seed"]),
                "target_temperature_K": int(plan["temperature_K"]),
                "planned_production_ps": int(plan["n_prod"]) * 0.001,
                "scientific_status": status,
                "failure_reasons": errors,
                "endpoint_window_ps": validation.get("endpoint_window_ps", ""),
                "temperature_K_mean": validation.get("temperature_K_mean", ""),
                "pressure_GPa_mean": validation.get("pressure_GPa_mean", ""),
                "volume_A3_per_atom_mean": validation.get("volume_A3_per_atom_mean", ""),
                "collapse_gate_temperature_K_mean": validation.get("collapse_gate_temperature_K_mean", ""),
                "collapse_gate_volume_A3_per_atom_mean": validation.get("collapse_gate_volume_A3_per_atom_mean", ""),
                "length_convergence_passed": (
                    ""
                    if validation.get("length_convergence") is None
                    else str(validation["length_convergence"].get("passed", False)).lower()
                ),
                "local_run_directory": str(root),
                "trajectory_path": str(trajectory) if trajectory.is_file() else "",
                "trajectory_size_bytes": trajectory.stat().st_size if trajectory.is_file() else "",
                "trajectory_sha256": sha256(trajectory) if trajectory.is_file() else "",
                "scientific_validation_path": str(validation_path) if validation_path.is_file() else "",
                "scientific_validation_sha256": sha256(validation_path) if validation_path.is_file() else "",
            }
        )
    return output


def promoted_chemistry(
    endpoints: list[dict[str, object]], model_id: str, phase: str, temperature: int
) -> list[dict[str, object]]:
    selected = [
        row
        for row in endpoints
        if row["model_id"] == model_id
        and row["simulated_phase"] == phase
        and int(row["target_temperature_K"]) == temperature
        and row["scientific_status"] == "passed"
    ]
    if len(selected) != 3 or {int(row["chemical_seed"]) for row in selected} != set(CHEMICAL_SEEDS):
        return []
    return [
        {
            "chemical_seed": int(row["chemical_seed"]),
            "temperature_K": float(row["temperature_K_mean"]),
            "pressure_GPa": float(row["pressure_GPa_mean"]),
            "volume_A3_per_atom": float(row["volume_A3_per_atom_mean"]),
        }
        for row in sorted(selected, key=lambda item: int(item["chemical_seed"]))
    ]


def group_row(
    model_id: str,
    architecture: str,
    training_parent: str,
    phase: str,
    temperature: int,
    chemistry: list[dict[str, object]],
    source: str,
) -> dict[str, object]:
    complete = len(chemistry) == 3
    row: dict[str, object] = {
        "model_id": model_id,
        "architecture": architecture,
        "training_parent": training_parent,
        "simulated_phase": phase,
        "temperature_K": temperature,
        "planned_chemical_replicas": 3,
        "eligible_chemical_replicas": len(chemistry),
        "status": "eligible" if complete else "blocked_incomplete_a21_group",
        "source": source,
        "ci_method": "two-sided Student-t, df=2",
    }
    for metric in METRICS:
        field = f"{metric}_mean"
        if complete:
            mean, sd, low, high = a19.mean_interval(
                [float(value[metric]) for value in chemistry]
            )
            row[field] = mean
            row[f"{field}_sd"] = sd
            row[f"{field}_ci95_low"] = low
            row[f"{field}_ci95_high"] = high
        else:
            row[field] = row[f"{field}_sd"] = ""
            row[f"{field}_ci95_low"] = row[f"{field}_ci95_high"] = ""
    return row


def response_row(
    model_id: str,
    architecture: str,
    training_parent: str,
    phase: str,
    chemistry_by_group: dict[tuple[str, str, int], list[dict[str, object]]],
    groups: dict[tuple[str, str, int], dict[str, object]],
) -> dict[str, object]:
    complete = all(
        len(chemistry_by_group[(model_id, phase, temperature)]) == 3
        for temperature in TEMPERATURES
    )
    row: dict[str, object] = {
        "model_id": model_id,
        "architecture": architecture,
        "training_parent": training_parent,
        "simulated_phase": phase,
        "temperature_range_K": "300;600;900;1200",
        "planned_chemical_replicas_per_temperature": 3,
        "status": "eligible" if complete else "blocked_incomplete_a21_temperature_matrix",
        "estimand": "equal-chemical mean; paired chemical-level Student-t intervals",
    }
    if not complete:
        return row
    by_temperature = {
        temperature: {
            int(item["chemical_seed"]): item
            for item in chemistry_by_group[(model_id, phase, temperature)]
        }
        for temperature in TEMPERATURES
    }
    adjacent_lows: list[float] = []
    for low_temperature, high_temperature in zip(TEMPERATURES[:-1], TEMPERATURES[1:], strict=True):
        deltas = [
            float(by_temperature[high_temperature][seed]["volume_A3_per_atom"])
            - float(by_temperature[low_temperature][seed]["volume_A3_per_atom"])
            for seed in CHEMICAL_SEEDS
        ]
        mean, low, high = a19.paired_interval(deltas)
        field = f"volume_delta_{high_temperature}_minus_{low_temperature}_A3_per_atom"
        row[field] = mean
        row[f"{field}_ci95_low"] = low
        row[f"{field}_ci95_high"] = high
        adjacent_lows.append(low)
    slopes = [
        a19.slope(
            TEMPERATURES,
            [
                math.log(float(by_temperature[temperature][seed]["volume_A3_per_atom"]))
                for temperature in TEMPERATURES
            ],
        )
        for seed in CHEMICAL_SEEDS
    ]
    alpha, alpha_low, alpha_high = a19.paired_interval(slopes)
    row.update(
        {
            "volumetric_alpha_per_K": alpha,
            "volumetric_alpha_per_K_ci95_low": alpha_low,
            "volumetric_alpha_per_K_ci95_high": alpha_high,
            "isotropic_linear_alpha_per_K": alpha / 3.0,
            "isotropic_linear_alpha_per_K_ci95_low": alpha_low / 3.0,
            "isotropic_linear_alpha_per_K_ci95_high": alpha_high / 3.0,
            "monotonic_status": (
                "resolved_increasing_all_adjacent_intervals_above_zero"
                if all(value > 0.0 for value in adjacent_lows)
                else "point_estimates_increasing_but_not_all_intervals_resolved"
            ),
        }
    )
    for temperature in TEMPERATURES:
        row[f"volume_A3_per_atom_{temperature}K"] = groups[
            (model_id, phase, temperature)
        ]["volume_A3_per_atom_mean"]
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a21-plan", type=Path, required=True)
    parser.add_argument("--a19-validation", type=Path, required=True)
    parser.add_argument("--a19-endpoints", type=Path, required=True)
    parser.add_argument("--a11-replicas", type=Path, required=True)
    parser.add_argument("--a12-replicas", type=Path, required=True)
    parser.add_argument("--run-endpoints", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--panel-groups", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--excluded-curve-audit", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    plan_path = args.a21_plan.resolve()
    a19_validation_path = args.a19_validation.resolve()
    a19_endpoint_path = args.a19_endpoints.resolve()
    a11_path = args.a11_replicas.resolve()
    a12_path = args.a12_replicas.resolve()
    require_a19_binding(a19_validation_path, a19_endpoint_path)
    a11_rows = a19.read_csv(a11_path)
    a12_rows = a19.read_csv(a12_path)
    old_endpoints = a19.read_csv(a19_endpoint_path)
    new_endpoints = a21_endpoints(plan_path)

    endpoint_fields = tuple(new_endpoints[0])
    a19.atomic_csv(args.run_endpoints.resolve(), endpoint_fields, new_endpoints)

    chemistry_by_group: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    curve_rows: list[dict[str, object]] = []
    for model_id, architecture, training_parent, phase in CURVE_BRANCHES:
        for temperature in TEMPERATURES:
            if model_id == "DPA3__bcc_parent" and temperature == 900:
                chemistry = promoted_chemistry(old_endpoints, model_id, phase, temperature)
                source = "A19/A20_primary_replacement"
            elif model_id == "DPA4__bcc_parent" and temperature == 1200:
                chemistry = promoted_chemistry(new_endpoints, model_id, phase, temperature)
                source = "A21_fresh_100ps_length_resolution"
            else:
                source_rows = a11_rows if temperature in (300, 1200) else a12_rows
                chemistry = a19.baseline_chemistry_values(source_rows, model_id, phase, temperature)
                source = "A11" if temperature in (300, 1200) else "A12"
            chemistry_by_group[(model_id, phase, temperature)] = chemistry
            curve_rows.append(
                group_row(model_id, architecture, training_parent, phase, temperature, chemistry, source)
            )

    group_fields = a19.group_row("x", "x", "x", "x", 0, [], "x").keys()
    group_fields = tuple(group_fields)
    a19.atomic_csv(args.groups.resolve(), group_fields, curve_rows)
    curve_map = {
        (str(row["model_id"]), str(row["simulated_phase"]), int(row["temperature_K"])): row
        for row in curve_rows
    }

    panel_rows: list[dict[str, object]] = []
    for model_id, architecture, training_parent, phase in PANEL_BRANCHES:
        for temperature in (300, 1200):
            key = (model_id, phase, temperature)
            if key in curve_map:
                panel_rows.append(curve_map[key])
            elif model_id == "DPA4__fcc_parent" and temperature == 300:
                chemistry = a19.baseline_chemistry_values(a11_rows, model_id, phase, temperature)
                panel_rows.append(group_row(model_id, architecture, training_parent, phase, temperature, chemistry, "A11"))
            else:
                panel_rows.append(group_row(model_id, architecture, training_parent, phase, temperature, [], "upstream_or_length_blocked"))
    a19.atomic_csv(args.panel_groups.resolve(), group_fields, panel_rows)

    excluded_rows: list[dict[str, object]] = []
    model_id, architecture, training_parent, phase = EXCLUDED_CURVE_BRANCH
    for temperature in TEMPERATURES:
        if temperature == 1200:
            chemistry = []
            source = "A11_length_convergence_block; excluded from Figure 7c by A21 scope"
        else:
            source_rows = a11_rows if temperature == 300 else a12_rows
            chemistry = a19.baseline_chemistry_values(source_rows, model_id, phase, temperature)
            source = ("A11" if temperature == 300 else "A12") + "; excluded from Figure 7c by A21 scope"
        excluded_rows.append(group_row(model_id, architecture, training_parent, phase, temperature, chemistry, source))
    a19.atomic_csv(args.excluded_curve_audit.resolve(), group_fields, excluded_rows)

    response_rows = [
        response_row(model_id, architecture, training_parent, phase, chemistry_by_group, curve_map)
        for model_id, architecture, training_parent, phase in CURVE_BRANCHES
    ]
    response_fields = a19.RESPONSE_FIELDS if hasattr(a19, "RESPONSE_FIELDS") else (
        "model_id", "architecture", "training_parent", "simulated_phase",
        "temperature_range_K", "planned_chemical_replicas_per_temperature", "status", "estimand",
        "volume_delta_600_minus_300_A3_per_atom", "volume_delta_600_minus_300_A3_per_atom_ci95_low", "volume_delta_600_minus_300_A3_per_atom_ci95_high",
        "volume_delta_900_minus_600_A3_per_atom", "volume_delta_900_minus_600_A3_per_atom_ci95_low", "volume_delta_900_minus_600_A3_per_atom_ci95_high",
        "volume_delta_1200_minus_900_A3_per_atom", "volume_delta_1200_minus_900_A3_per_atom_ci95_low", "volume_delta_1200_minus_900_A3_per_atom_ci95_high",
        "monotonic_status", "volume_A3_per_atom_300K", "volume_A3_per_atom_600K", "volume_A3_per_atom_900K", "volume_A3_per_atom_1200K",
        "volumetric_alpha_per_K", "volumetric_alpha_per_K_ci95_low", "volumetric_alpha_per_K_ci95_high",
        "isotropic_linear_alpha_per_K", "isotropic_linear_alpha_per_K_ci95_low", "isotropic_linear_alpha_per_K_ci95_high",
    )
    for row in response_rows:
        for field in response_fields:
            row.setdefault(field, "")
    a19.atomic_csv(args.response.resolve(), response_fields, response_rows)

    all_a21_passed = all(row["scientific_status"] == "passed" for row in new_endpoints)
    complete_curves = all(row["status"] == "eligible" for row in curve_rows)
    outputs = {
        "run_endpoints": args.run_endpoints.resolve(),
        "groups": args.groups.resolve(),
        "panel_groups": args.panel_groups.resolve(),
        "response": args.response.resolve(),
        "excluded_curve_audit": args.excluded_curve_audit.resolve(),
    }
    receipt = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all_a21_passed and complete_curves else "incomplete_or_blocked",
        "all_a21_rows_passed": all_a21_passed,
        "all_three_promoted_curves_complete": complete_curves,
        "a21_rows": len(new_endpoints),
        "a21_passed_rows": sum(row["scientific_status"] == "passed" for row in new_endpoints),
        "figure7c_branches": [f"{architecture}/{phase.upper()}" for _, architecture, _, phase in CURVE_BRANCHES],
        "figure7c_numeric_points": sum(row["status"] == "eligible" for row in curve_rows),
        "excluded_from_figure7c": "DPA4/FCC",
        "excluded_curve_audit_rows": len(excluded_rows),
        "a19_validation": str(a19_validation_path),
        "a19_validation_sha256": sha256(a19_validation_path),
        "input_tables": {str(path): sha256(path) for path in (a19_endpoint_path, a11_path, a12_path)},
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "outputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
    }
    a19.atomic_json(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
