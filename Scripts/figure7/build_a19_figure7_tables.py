#!/usr/bin/env python3
"""Build Figure 7 A19 tables from baseline replica tables and new raw runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path


CHEMICAL_SEEDS = (20260825, 20262843, 20264861)
VELOCITY_SEEDS = (20260901, 20260903)
TEMPERATURES = (300, 600, 900, 1200)
T_CRIT_DF2 = 4.302652729911275
CURVE_BRANCHES = (
    ("DPA2__bcc_parent", "DPA2", "bcc", "bcc"),
    ("DPA3__bcc_parent", "DPA3", "bcc", "bcc"),
    ("DPA4__bcc_parent", "DPA4", "bcc", "bcc"),
    ("DPA4__fcc_parent", "DPA4", "fcc", "fcc"),
)
PANEL_BRANCHES = (
    ("DPA2__bcc_parent", "DPA2", "bcc", "bcc"),
    ("DPA2__fcc_parent", "DPA2", "fcc", "fcc"),
    ("DPA3__bcc_parent", "DPA3", "bcc", "bcc"),
    ("DPA3__fcc_parent", "DPA3", "fcc", "fcc"),
    ("DPA4__bcc_parent", "DPA4", "bcc", "bcc"),
    ("DPA4__fcc_parent", "DPA4", "fcc", "fcc"),
)
METRICS = (
    "temperature_K",
    "pressure_GPa",
    "volume_A3_per_atom",
)
PROMOTED_COHORT = "primary_replacement"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"empty table: {path}")
    return rows


def atomic_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n"
            )
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
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
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


def mean_interval(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("A19 group interval requires three finite values")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = T_CRIT_DF2 * sd / math.sqrt(3.0)
    return mean, sd, mean - half, mean + half


def baseline_chemistry_values(
    rows: list[dict[str, str]],
    model_id: str,
    phase: str,
    temperature: int,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if row["model_id"] == model_id
        and row["simulated_phase"] == phase
        and int(row["target_temperature_K"]) == temperature
        and row["endpoint_status"] == "eligible"
    ]
    output: list[dict[str, object]] = []
    for chemical_seed in CHEMICAL_SEEDS:
        replicas = [
            row
            for row in selected
            if int(row["chemical_seed"]) == chemical_seed
            and int(row["velocity_seed"]) in VELOCITY_SEEDS
        ]
        if len(replicas) != 2 or {
            int(row["velocity_seed"]) for row in replicas
        } != set(VELOCITY_SEEDS):
            raise ValueError(
                f"{model_id}/{phase}/{temperature}/{chemical_seed}: "
                "baseline chemistry does not contain two eligible velocities"
            )
        output.append(
            {
                "chemical_seed": chemical_seed,
                "temperature_K": statistics.fmean(
                    float(row["temperature_K_mean"]) for row in replicas
                ),
                "pressure_GPa": statistics.fmean(
                    float(row["pressure_GPa_mean"]) for row in replicas
                ),
                "volume_A3_per_atom": statistics.fmean(
                    float(row["volume_A3_per_atom_mean"]) for row in replicas
                ),
            }
        )
    return output


def endpoint_rows(
    campaign_root: Path, plan_paths: list[Path]
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    endpoints: list[dict[str, object]] = []
    by_task: dict[str, dict[str, object]] = {}
    for plan_path in plan_paths:
        receipt_path = plan_path.with_suffix(plan_path.suffix + ".receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("plan_sha256") != sha256(plan_path):
            raise ValueError(f"plan receipt mismatch: {plan_path}")
        for plan in read_csv(plan_path, delimiter="\t"):
            if plan["cohort"] != PROMOTED_COHORT:
                continue
            task_id = plan["task_id"]
            site = plan["site"]
            root = campaign_root / "runs" / site / task_id
            production = root / "production"
            validation_path = production / "scientific_validation.json"
            if validation_path.is_file():
                validation = json.loads(
                    validation_path.read_text(encoding="utf-8")
                )
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
            row: dict[str, object] = {
                "task_id": task_id,
                "site": site,
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
                "volume_A3_per_atom_mean": validation.get(
                    "volume_A3_per_atom_mean", ""
                ),
                "collapse_gate_temperature_K_mean": validation.get(
                    "collapse_gate_temperature_K_mean", ""
                ),
                "collapse_gate_volume_A3_per_atom_mean": validation.get(
                    "collapse_gate_volume_A3_per_atom_mean", ""
                ),
                "length_convergence_passed": (
                    ""
                    if validation.get("length_convergence") is None
                    else str(
                        validation["length_convergence"].get("passed", False)
                    ).lower()
                ),
                "local_run_directory": str(root),
                "trajectory_path": str(trajectory) if trajectory.is_file() else "",
                "trajectory_size_bytes": (
                    trajectory.stat().st_size if trajectory.is_file() else ""
                ),
                "trajectory_sha256": sha256(trajectory) if trajectory.is_file() else "",
                "scientific_validation_path": (
                    str(validation_path) if validation_path.is_file() else ""
                ),
                "scientific_validation_sha256": (
                    sha256(validation_path) if validation_path.is_file() else ""
                ),
            }
            if task_id in by_task:
                raise ValueError(f"duplicate A19 task ID: {task_id}")
            endpoints.append(row)
            by_task[task_id] = row
    return endpoints, by_task


def recovery_chemistry_values(
    endpoints: list[dict[str, object]],
    model_id: str,
    phase: str,
    cohort: str,
    temperature: int,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in endpoints
        if row["model_id"] == model_id
        and row["simulated_phase"] == phase
        and row["cohort"] == cohort
        and row["target_temperature_K"] == temperature
        and row["scientific_status"] == "passed"
    ]
    if len(selected) != 3 or {
        int(row["chemical_seed"]) for row in selected
    } != set(CHEMICAL_SEEDS):
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
    row: dict[str, object] = {
        "model_id": model_id,
        "architecture": architecture,
        "training_parent": training_parent,
        "simulated_phase": phase,
        "temperature_K": temperature,
        "planned_chemical_replicas": 3,
        "eligible_chemical_replicas": len(chemistry),
        "status": "eligible" if len(chemistry) == 3 else "blocked_incomplete_a19_group",
        "source": source,
        "ci_method": "two-sided Student-t, df=2",
    }
    for metric in METRICS:
        prefix = f"{metric}_mean"
        if len(chemistry) == 3:
            mean, sd, low, high = mean_interval(
                [float(value[metric]) for value in chemistry]
            )
            row[prefix] = mean
            row[f"{prefix}_sd"] = sd
            row[f"{prefix}_ci95_low"] = low
            row[f"{prefix}_ci95_high"] = high
        else:
            row[prefix] = ""
            row[f"{prefix}_sd"] = ""
            row[f"{prefix}_ci95_low"] = ""
            row[f"{prefix}_ci95_high"] = ""
    return row


def paired_interval(values: list[float]) -> tuple[float, float, float]:
    mean, _, low, high = mean_interval(values)
    return mean, low, high


def slope(x: tuple[int, ...], y: list[float]) -> float:
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    return sum(
        (value_x - mean_x) * (value_y - mean_y)
        for value_x, value_y in zip(x, y, strict=True)
    ) / sum((value - mean_x) ** 2 for value in x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--a11-replicas", type=Path, required=True)
    parser.add_argument("--a12-replicas", type=Path, required=True)
    parser.add_argument("--run-endpoints", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--panel-groups", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    campaign_root = args.campaign_root.resolve()
    plans = [path.resolve() for path in args.plan]
    a11_path = args.a11_replicas.resolve()
    a12_path = args.a12_replicas.resolve()
    a11 = read_csv(a11_path)
    a12 = read_csv(a12_path)
    endpoints, _ = endpoint_rows(campaign_root, plans)

    endpoint_fields = (
        "task_id",
        "site",
        "cohort",
        "model_id",
        "architecture",
        "training_parent",
        "simulated_phase",
        "chemical_seed",
        "velocity_seed",
        "target_temperature_K",
        "planned_production_ps",
        "scientific_status",
        "failure_reasons",
        "endpoint_window_ps",
        "temperature_K_mean",
        "pressure_GPa_mean",
        "volume_A3_per_atom_mean",
        "collapse_gate_temperature_K_mean",
        "collapse_gate_volume_A3_per_atom_mean",
        "length_convergence_passed",
        "local_run_directory",
        "trajectory_path",
        "trajectory_size_bytes",
        "trajectory_sha256",
        "scientific_validation_path",
        "scientific_validation_sha256",
    )
    atomic_csv(args.run_endpoints.resolve(), endpoint_fields, endpoints)

    chemistry_by_group: dict[
        tuple[str, str, int], list[dict[str, object]]
    ] = {}
    group_rows: list[dict[str, object]] = []
    for model_id, architecture, training_parent, phase in CURVE_BRANCHES:
        for temperature in TEMPERATURES:
            if model_id == "DPA3__bcc_parent" and temperature == 900:
                chemistry = recovery_chemistry_values(
                    endpoints,
                    model_id,
                    phase,
                    "primary_replacement",
                    temperature,
                )
                source = "A19_primary_replacement"
            elif model_id.startswith("DPA4") and temperature == 1200:
                chemistry = []
                source = "A11_length_convergence_block_retained_by_A20"
            else:
                source_rows = a11 if temperature in (300, 1200) else a12
                chemistry = baseline_chemistry_values(
                    source_rows, model_id, phase, temperature
                )
                source = "A11" if temperature in (300, 1200) else "A12"
            chemistry_by_group[(model_id, phase, temperature)] = chemistry
            group_rows.append(
                group_row(
                    model_id,
                    architecture,
                    training_parent,
                    phase,
                    temperature,
                    chemistry,
                    source,
                )
            )

    group_fields = (
        "model_id",
        "architecture",
        "training_parent",
        "simulated_phase",
        "temperature_K",
        "planned_chemical_replicas",
        "eligible_chemical_replicas",
        "status",
        "source",
        "ci_method",
        "temperature_K_mean",
        "temperature_K_mean_sd",
        "temperature_K_mean_ci95_low",
        "temperature_K_mean_ci95_high",
        "pressure_GPa_mean",
        "pressure_GPa_mean_sd",
        "pressure_GPa_mean_ci95_low",
        "pressure_GPa_mean_ci95_high",
        "volume_A3_per_atom_mean",
        "volume_A3_per_atom_mean_sd",
        "volume_A3_per_atom_mean_ci95_low",
        "volume_A3_per_atom_mean_ci95_high",
    )
    atomic_csv(args.groups.resolve(), group_fields, group_rows)

    group_map = {
        (
            str(row["model_id"]),
            str(row["simulated_phase"]),
            int(row["temperature_K"]),
        ): row
        for row in group_rows
    }
    panel_rows: list[dict[str, object]] = []
    for model_id, architecture, training_parent, phase in PANEL_BRANCHES:
        for temperature in (300, 1200):
            if (model_id, phase, temperature) in group_map:
                panel_rows.append(group_map[(model_id, phase, temperature)])
            else:
                panel_rows.append(
                    group_row(
                        model_id,
                        architecture,
                        training_parent,
                        phase,
                        temperature,
                        [],
                        "upstream_blocked",
                    )
                )
    atomic_csv(args.panel_groups.resolve(), group_fields, panel_rows)

    response_rows: list[dict[str, object]] = []
    for model_id, architecture, training_parent, phase in CURVE_BRANCHES:
        chemistry_sets = [
            chemistry_by_group[(model_id, phase, temperature)]
            for temperature in TEMPERATURES
        ]
        complete = all(len(values) == 3 for values in chemistry_sets)
        row: dict[str, object] = {
            "model_id": model_id,
            "architecture": architecture,
            "training_parent": training_parent,
            "simulated_phase": phase,
            "temperature_range_K": "300;600;900;1200",
            "planned_chemical_replicas_per_temperature": 3,
            "status": "eligible" if complete else "blocked_incomplete_a19_temperature_matrix",
            "estimand": "equal-chemical mean; paired chemical-level Student-t intervals",
        }
        if complete:
            by_temperature = {
                temperature: {
                    int(item["chemical_seed"]): item
                    for item in chemistry_by_group[
                        (model_id, phase, temperature)
                    ]
                }
                for temperature in TEMPERATURES
            }
            adjacent_lows: list[float] = []
            for low_t, high_t in zip(
                TEMPERATURES[:-1], TEMPERATURES[1:], strict=True
            ):
                deltas = [
                    float(by_temperature[high_t][seed]["volume_A3_per_atom"])
                    - float(by_temperature[low_t][seed]["volume_A3_per_atom"])
                    for seed in CHEMICAL_SEEDS
                ]
                mean, low, high = paired_interval(deltas)
                name = f"volume_delta_{high_t}_minus_{low_t}_A3_per_atom"
                row[name] = mean
                row[f"{name}_ci95_low"] = low
                row[f"{name}_ci95_high"] = high
                adjacent_lows.append(low)
            slopes = [
                slope(
                    TEMPERATURES,
                    [
                        math.log(
                            float(
                                by_temperature[temperature][seed][
                                    "volume_A3_per_atom"
                                ]
                            )
                        )
                        for temperature in TEMPERATURES
                    ],
                )
                for seed in CHEMICAL_SEEDS
            ]
            alpha, alpha_low, alpha_high = paired_interval(slopes)
            row["volumetric_alpha_per_K"] = alpha
            row["volumetric_alpha_per_K_ci95_low"] = alpha_low
            row["volumetric_alpha_per_K_ci95_high"] = alpha_high
            row["isotropic_linear_alpha_per_K"] = alpha / 3.0
            row["isotropic_linear_alpha_per_K_ci95_low"] = alpha_low / 3.0
            row["isotropic_linear_alpha_per_K_ci95_high"] = alpha_high / 3.0
            row["monotonic_status"] = (
                "resolved_increasing_all_adjacent_intervals_above_zero"
                if all(value > 0.0 for value in adjacent_lows)
                else "point_estimates_increasing_but_not_all_intervals_resolved"
            )
            for temperature in TEMPERATURES:
                row[f"volume_A3_per_atom_{temperature}K"] = group_map[
                    (model_id, phase, temperature)
                ]["volume_A3_per_atom_mean"]
        response_rows.append(row)

    response_fields = (
        "model_id",
        "architecture",
        "training_parent",
        "simulated_phase",
        "temperature_range_K",
        "planned_chemical_replicas_per_temperature",
        "status",
        "estimand",
        "volume_delta_600_minus_300_A3_per_atom",
        "volume_delta_600_minus_300_A3_per_atom_ci95_low",
        "volume_delta_600_minus_300_A3_per_atom_ci95_high",
        "volume_delta_900_minus_600_A3_per_atom",
        "volume_delta_900_minus_600_A3_per_atom_ci95_low",
        "volume_delta_900_minus_600_A3_per_atom_ci95_high",
        "volume_delta_1200_minus_900_A3_per_atom",
        "volume_delta_1200_minus_900_A3_per_atom_ci95_low",
        "volume_delta_1200_minus_900_A3_per_atom_ci95_high",
        "monotonic_status",
        "volume_A3_per_atom_300K",
        "volume_A3_per_atom_600K",
        "volume_A3_per_atom_900K",
        "volume_A3_per_atom_1200K",
        "volumetric_alpha_per_K",
        "volumetric_alpha_per_K_ci95_low",
        "volumetric_alpha_per_K_ci95_high",
        "isotropic_linear_alpha_per_K",
        "isotropic_linear_alpha_per_K_ci95_low",
        "isotropic_linear_alpha_per_K_ci95_high",
    )
    for row in response_rows:
        for field in response_fields:
            row.setdefault(field, "")
    atomic_csv(args.response.resolve(), response_fields, response_rows)

    terminal = {
        "passed",
        "failed",
        "failed_before_scientific_validation",
    }
    all_terminal = all(
        str(row["scientific_status"]) in terminal for row in endpoints
    )
    replacement_complete = (
        group_map[("DPA3__bcc_parent", "bcc", 900)]["status"] == "eligible"
    )
    all_four_temperature_groups_complete = all(
        row["status"] == "eligible" for row in group_rows
    )
    outputs = {
        "run_endpoints": args.run_endpoints.resolve(),
        "groups": args.groups.resolve(),
        "panel_groups": args.panel_groups.resolve(),
        "response": args.response.resolve(),
    }
    receipt = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "passed"
            if all_terminal and replacement_complete
            else "incomplete_or_blocked"
        ),
        "all_planned_a19_rows_terminal": all_terminal,
        "all_required_a20_replacement_groups_complete": replacement_complete,
        "all_four_temperature_groups_complete": all_four_temperature_groups_complete,
        "retained_dpa4_1200_length_blocks": 2,
        "a19_rows": len(endpoints),
        "a19_passed_rows": sum(
            row["scientific_status"] == "passed" for row in endpoints
        ),
        "a19_failed_rows": sum(
            str(row["scientific_status"]).startswith("failed")
            for row in endpoints
        ),
        "input_tables": {
            str(a11_path): sha256(a11_path),
            str(a12_path): sha256(a12_path),
        },
        "plans": {str(path): sha256(path) for path in plans},
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
    }
    atomic_json(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if receipt["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
