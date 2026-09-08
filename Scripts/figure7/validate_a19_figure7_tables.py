#!/usr/bin/env python3
"""Independently reconstruct every A19-promoted Figure 7 endpoint."""

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
from datetime import datetime, timezone
from pathlib import Path


CHEMICAL_SEEDS = (20260825, 20262843, 20264861)
BASELINE_VELOCITIES = (20260901, 20260903)
TEMPERATURES = (300, 600, 900, 1200)
T_DF2 = 4.302652729911275
CURVES = (
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
METRICS = ("temperature_K", "pressure_GPa", "volume_A3_per_atom")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        values = list(csv.DictReader(handle, delimiter=delimiter))
    if not values:
        raise ValueError(f"empty table: {path}")
    return values


def record(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def true(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise ValueError(message)

    def close(self, observed: object, expected: float, message: str) -> None:
        self.count += 1
        try:
            value = float(observed)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{message}: non-numeric {observed!r}") from error
        if not math.isclose(value, expected, rel_tol=2.0e-11, abs_tol=2.0e-12):
            raise ValueError(f"{message}: {value:.16g} != {expected:.16g}")


def chemistry_interval(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("three finite chemical values are required")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = T_DF2 * standard_deviation / math.sqrt(3.0)
    return mean, standard_deviation, mean - half_width, mean + half_width


def baseline_chemistry(
    source: list[dict[str, str]], model_id: str, phase: str, temperature: int
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for chemical_seed in CHEMICAL_SEEDS:
        selected = [
            row
            for row in source
            if row["model_id"] == model_id
            and row["simulated_phase"] == phase
            and int(row["target_temperature_K"]) == temperature
            and int(row["chemical_seed"]) == chemical_seed
            and int(row["velocity_seed"]) in BASELINE_VELOCITIES
            and row["endpoint_status"] == "eligible"
        ]
        if len(selected) != 2 or {
            int(row["velocity_seed"]) for row in selected
        } != set(BASELINE_VELOCITIES):
            raise ValueError(
                f"baseline population mismatch: {model_id}/{phase}/{temperature}/{chemical_seed}"
            )
        result[chemical_seed] = {
            metric: statistics.fmean(
                float(row[f"{metric}_mean"]) for row in selected
            )
            for metric in METRICS
        }
    return result


def plan_rows(plan_paths: list[Path], checks: Checks) -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    for path in plan_paths:
        receipt_path = path.with_suffix(path.suffix + ".receipt.json")
        receipt = record(receipt_path)
        checks.true(
            receipt.get("plan_sha256") == sha256(path),
            f"plan receipt hash mismatch: {path}",
        )
        all_rows.extend(rows(path, delimiter="\t"))
    task_ids = [row["task_id"] for row in all_rows]
    checks.true(len(task_ids) == 7, "Skipjack A19 plan does not contain exactly 7 tasks")
    checks.true(len(set(task_ids)) == 7, "A19 task IDs are not unique")
    checks.true(
        Counter(row["cohort"] for row in all_rows)
        == Counter(
            {
                "diagnostic_exact_replay": 1,
                "primary_replacement": 3,
                "velocity_sensitivity": 3,
            }
        ),
        "A19 cohort population mismatch",
    )
    diagnostic = [
        row for row in all_rows if row["cohort"] == "diagnostic_exact_replay"
    ]
    checks.true(
        diagnostic[0]["chemical_seed"] == "20260825"
        and diagnostic[0]["velocity_seed"] == "20260903",
        "original collapsed identity is not isolated as the stopped diagnostic",
    )
    result = [row for row in all_rows if row["cohort"] == "primary_replacement"]
    checks.true(len(result) == 3, "A20 primary replacement population is not three")
    checks.true(
        {int(row["chemical_seed"]) for row in result} == set(CHEMICAL_SEEDS)
        and {int(row["velocity_seed"]) for row in result} == {20260905},
        "A20 primary replacement seeds do not match the frozen cohort",
    )
    return result


def validate_runs(
    campaign_root: Path,
    plans: list[dict[str, str]],
    endpoint_path: Path,
    checks: Checks,
) -> tuple[
    dict[str, dict[str, str]],
    dict[tuple[str, str, int], dict[int, dict[str, float]]],
]:
    endpoint_values = rows(endpoint_path)
    endpoints = {row["task_id"]: row for row in endpoint_values}
    checks.true(len(endpoint_values) == 3 and len(endpoints) == 3, "endpoint population mismatch")
    promoted: dict[tuple[str, str, int], dict[int, dict[str, float]]] = {}
    for plan in plans:
        task_id = plan["task_id"]
        checks.true(task_id in endpoints, f"missing endpoint row: {task_id}")
        endpoint = endpoints[task_id]
        root = campaign_root / "runs" / plan["site"] / task_id
        validation_path = root / "production" / "scientific_validation.json"
        checks.true(validation_path.is_file(), f"missing scientific validation: {task_id}")
        validation = record(validation_path)
        status = str(validation.get("status", ""))
        checks.true(status in {"passed", "failed"}, f"nonterminal validation: {task_id}")
        checks.true(endpoint["scientific_status"] == status, f"endpoint status mismatch: {task_id}")
        checks.true(
            endpoint["scientific_validation_sha256"] == sha256(validation_path),
            f"validation hash mismatch: {task_id}",
        )
        checks.true(
            Path(endpoint["scientific_validation_path"]).resolve() == validation_path.resolve(),
            f"validation path mismatch: {task_id}",
        )
        trajectory = root / "production" / "trajectory_npt.lammpstrj"
        if trajectory.is_file():
            checks.true(endpoint["trajectory_sha256"] == sha256(trajectory), f"trajectory hash mismatch: {task_id}")
            checks.true(int(endpoint["trajectory_size_bytes"]) == trajectory.stat().st_size, f"trajectory size mismatch: {task_id}")
            checks.true(trajectory.stat().st_size < 10_000_000_000, f"trajectory exceeds 10 GB: {task_id}")
        checks.true(status == "passed", f"promoted A20 task failed: {task_id}")
        checks.true(validation.get("natoms") == [2000], f"atom population mismatch: {task_id}")
        for field in (
            "temperature_K_mean",
            "pressure_GPa_mean",
            "volume_A3_per_atom_mean",
        ):
            checks.close(endpoint[field], float(validation[field]), f"endpoint {field}: {task_id}")
        key = (plan["model_id"], plan["phase"], int(plan["temperature_K"]))
        promoted.setdefault(key, {})[int(plan["chemical_seed"])] = {
            metric: float(validation[f"{metric}_mean"]) for metric in METRICS
        }
    checks.true(
        {
            key: set(value) for key, value in promoted.items()
        }
        == {
            ("DPA3__bcc_parent", "bcc", 900): set(CHEMICAL_SEEDS),
        },
        "promoted A19 chemistry population mismatch",
    )
    checks.true(
        all(
            not (
                row["chemical_seed"] == "20260825"
                and row["velocity_seed"] == "20260903"
            )
            for row in plans
        ),
        "original collapsed identity leaked into the promoted population",
    )
    return endpoints, promoted


def expected_chemistry(
    a11: list[dict[str, str]],
    a12: list[dict[str, str]],
    promoted: dict[tuple[str, str, int], dict[int, dict[str, float]]],
) -> dict[tuple[str, str, int], dict[int, dict[str, float]]]:
    result: dict[tuple[str, str, int], dict[int, dict[str, float]]] = {}
    for model_id, _, _, phase in CURVES:
        for temperature in TEMPERATURES:
            key = (model_id, phase, temperature)
            if key in promoted:
                result[key] = promoted[key]
            elif model_id.startswith("DPA4") and temperature == 1200:
                result[key] = {}
            else:
                source = a11 if temperature in (300, 1200) else a12
                result[key] = baseline_chemistry(source, model_id, phase, temperature)
    return result


def validate_groups(
    group_path: Path,
    chemistry: dict[tuple[str, str, int], dict[int, dict[str, float]]],
    checks: Checks,
) -> dict[tuple[str, str, int], dict[str, str]]:
    values = rows(group_path)
    groups = {
        (row["model_id"], row["simulated_phase"], int(row["temperature_K"])): row
        for row in values
    }
    checks.true(len(values) == 16 and set(groups) == set(chemistry), "four-temperature group matrix mismatch")
    for key, chemical_values in chemistry.items():
        row = groups[key]
        complete = len(chemical_values) == 3
        checks.true(
            row["status"]
            == ("eligible" if complete else "blocked_incomplete_a19_group"),
            f"group status mismatch: {key}",
        )
        checks.true(row["ci_method"] == "two-sided Student-t, df=2", f"interval method mismatch: {key}")
        checks.true(
            int(row["eligible_chemical_replicas"]) == (3 if complete else 0),
            f"replica count mismatch: {key}",
        )
        if not complete:
            for metric in METRICS:
                checks.true(not row[f"{metric}_mean"], f"blocked group leaks {metric}: {key}")
            continue
        for metric in METRICS:
            mean, standard_deviation, low, high = chemistry_interval(
                [chemical_values[seed][metric] for seed in CHEMICAL_SEEDS]
            )
            for suffix, expected in (
                ("", mean),
                ("_sd", standard_deviation),
                ("_ci95_low", low),
                ("_ci95_high", high),
            ):
                checks.close(row[f"{metric}_mean{suffix}"], expected, f"{key}/{metric}{suffix}")
    return groups


def validate_panel(
    panel_path: Path,
    groups: dict[tuple[str, str, int], dict[str, str]],
    checks: Checks,
) -> None:
    values = rows(panel_path)
    panel = {
        (row["model_id"], row["simulated_phase"], int(row["temperature_K"])): row
        for row in values
    }
    expected = {
        (model_id, phase, temperature)
        for model_id, _, _, phase in PANEL_BRANCHES
        for temperature in (300, 1200)
    }
    checks.true(len(values) == 12 and set(panel) == expected, "panel-b/d group matrix mismatch")
    for key, row in panel.items():
        if key in groups:
            checks.true(row == groups[key], f"panel group is not copied exactly: {key}")
        else:
            checks.true(row["status"] == "blocked_incomplete_a19_group", f"upstream block missing: {key}")
            checks.true(not row["volume_A3_per_atom_mean"], f"blocked panel group has a value: {key}")


def fit_slope(volumes: list[float]) -> float:
    x_mean = statistics.fmean(TEMPERATURES)
    y = [math.log(value) for value in volumes]
    y_mean = statistics.fmean(y)
    return sum(
        (temperature - x_mean) * (value - y_mean)
        for temperature, value in zip(TEMPERATURES, y, strict=True)
    ) / sum((temperature - x_mean) ** 2 for temperature in TEMPERATURES)


def validate_responses(
    response_path: Path,
    chemistry: dict[tuple[str, str, int], dict[int, dict[str, float]]],
    checks: Checks,
) -> None:
    values = rows(response_path)
    responses = {(row["model_id"], row["simulated_phase"]): row for row in values}
    expected = {(model_id, phase) for model_id, _, _, phase in CURVES}
    checks.true(len(values) == 4 and set(responses) == expected, "response population mismatch")
    for model_id, phase in expected:
        row = responses[(model_id, phase)]
        complete = all(
            len(chemistry[(model_id, phase, temperature)]) == 3
            for temperature in TEMPERATURES
        )
        checks.true(
            row["status"]
            == ("eligible" if complete else "blocked_incomplete_a19_temperature_matrix"),
            f"response status mismatch: {model_id}/{phase}",
        )
        if not complete:
            checks.true(
                not row["volumetric_alpha_per_K"],
                f"blocked response leaks a slope: {model_id}/{phase}",
            )
            continue
        for low_temperature, high_temperature in zip(TEMPERATURES[:-1], TEMPERATURES[1:], strict=True):
            deltas = [
                chemistry[(model_id, phase, high_temperature)][seed]["volume_A3_per_atom"]
                - chemistry[(model_id, phase, low_temperature)][seed]["volume_A3_per_atom"]
                for seed in CHEMICAL_SEEDS
            ]
            mean, _, low, high = chemistry_interval(deltas)
            field = f"volume_delta_{high_temperature}_minus_{low_temperature}_A3_per_atom"
            checks.close(row[field], mean, field)
            checks.close(row[f"{field}_ci95_low"], low, f"{field}/low")
            checks.close(row[f"{field}_ci95_high"], high, f"{field}/high")
        slopes = [
            fit_slope(
                [
                    chemistry[(model_id, phase, temperature)][seed]["volume_A3_per_atom"]
                    for temperature in TEMPERATURES
                ]
            )
            for seed in CHEMICAL_SEEDS
        ]
        alpha, _, alpha_low, alpha_high = chemistry_interval(slopes)
        checks.close(row["volumetric_alpha_per_K"], alpha, f"{model_id}/alpha")
        checks.close(row["volumetric_alpha_per_K_ci95_low"], alpha_low, f"{model_id}/alpha low")
        checks.close(row["volumetric_alpha_per_K_ci95_high"], alpha_high, f"{model_id}/alpha high")
        checks.close(row["isotropic_linear_alpha_per_K"], alpha / 3.0, f"{model_id}/linear alpha")


def validate(args: argparse.Namespace) -> dict[str, object]:
    checks = Checks()
    campaign_root = args.campaign_root.resolve()
    plan_paths = [path.resolve() for path in args.plan]
    plans = plan_rows(plan_paths, checks)
    analysis_receipt = record(args.analysis_receipt.resolve())
    checks.true(analysis_receipt.get("status") == "passed", "analysis receipt is not passed")
    output_paths = {
        "run_endpoints": args.run_endpoints.resolve(),
        "groups": args.groups.resolve(),
        "panel_groups": args.panel_groups.resolve(),
        "response": args.response.resolve(),
    }
    bound_outputs = analysis_receipt.get("outputs")
    checks.true(isinstance(bound_outputs, dict), "analysis receipt has no output bindings")
    for name, path in output_paths.items():
        bound = bound_outputs.get(name) if isinstance(bound_outputs, dict) else None
        checks.true(isinstance(bound, dict), f"missing analysis binding: {name}")
        checks.true(Path(str(bound["path"])).resolve() == path, f"analysis path mismatch: {name}")
        checks.true(bound["sha256"] == sha256(path), f"analysis hash mismatch: {name}")
    _, promoted = validate_runs(campaign_root, plans, output_paths["run_endpoints"], checks)
    a11_path = args.a11_replicas.resolve()
    a12_path = args.a12_replicas.resolve()
    chemistry = expected_chemistry(rows(a11_path), rows(a12_path), promoted)
    groups = validate_groups(output_paths["groups"], chemistry, checks)
    validate_panel(output_paths["panel_groups"], groups, checks)
    validate_responses(output_paths["response"], chemistry, checks)
    return {
        "schema_version": 1,
        "status": "passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks.count,
        "errors": [],
        "analysis_receipt": str(args.analysis_receipt.resolve()),
        "analysis_receipt_sha256": sha256(args.analysis_receipt.resolve()),
        "plans": {str(path): sha256(path) for path in plan_paths},
        "baseline_replica_tables": {
            str(a11_path): sha256(a11_path),
            str(a12_path): sha256(a12_path),
        },
        "validated_outputs": {str(path): sha256(path) for path in output_paths.values()},
        "collapsed_identity_excluded": {
            "model_id": "DPA3__bcc_parent",
            "chemical_seed": 20260825,
            "velocity_seed": 20260903,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--a11-replicas", type=Path, required=True)
    parser.add_argument("--a12-replicas", type=Path, required=True)
    parser.add_argument("--run-endpoints", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--panel-groups", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--analysis-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args)
    except (KeyError, OSError, TypeError, ValueError) as error:
        result = {
            "schema_version": 1,
            "status": "failed",
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": 0,
            "errors": [str(error)],
        }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
