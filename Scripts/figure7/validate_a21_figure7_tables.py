#!/usr/bin/env python3
"""Independently reconstruct A21 Figure 7 tables from raw thermodynamics."""

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
BASELINE_VELOCITIES = (20260901, 20260903)
TEMPERATURES = (300, 600, 900, 1200)
T_DF2 = 4.302652729911275
CURVES = (
    ("DPA2__bcc_parent", "DPA2", "bcc", "bcc"),
    ("DPA3__bcc_parent", "DPA3", "bcc", "bcc"),
    ("DPA4__bcc_parent", "DPA4", "bcc", "bcc"),
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
THERMO_COLUMNS = (
    "step", "time_ps", "natoms", "temperature_K", "pressure_bar",
    "volume_A3", "density_g_cm3", "enthalpy_eV", "enthalpy2_eV2",
    "potential_energy_eV",
)
PROJECT = Path(__file__).resolve().parents[2]
A23_PROTOCOL = PROJECT / "protocols/lammps_validation_protocol_amendment_A23_20260901.md"


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


def interval(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("three finite chemistry values are required")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half = T_DF2 * standard_deviation / math.sqrt(3.0)
    return mean, standard_deviation, mean - half, mean + half


def baseline_chemistry(
    source: list[dict[str, str]], model_id: str, phase: str, temperature: int
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for seed in CHEMICAL_SEEDS:
        selected = [
            row for row in source
            if row["model_id"] == model_id
            and row["simulated_phase"] == phase
            and int(row["target_temperature_K"]) == temperature
            and int(row["chemical_seed"]) == seed
            and int(row["velocity_seed"]) in BASELINE_VELOCITIES
            and row["endpoint_status"] == "eligible"
        ]
        if len(selected) != 2 or {int(row["velocity_seed"]) for row in selected} != set(BASELINE_VELOCITIES):
            raise ValueError(f"baseline population mismatch: {model_id}/{phase}/{temperature}/{seed}")
        result[seed] = {
            metric: statistics.fmean(float(row[f"{metric}_mean"]) for row in selected)
            for metric in METRICS
        }
    return result


def read_thermo(path: Path) -> list[dict[str, float]]:
    values: list[dict[str, float]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = [float(item) for item in line.split()]
        if len(raw) != len(THERMO_COLUMNS) or not all(math.isfinite(item) for item in raw):
            raise ValueError(f"invalid thermo row: {path}:{number}")
        row = dict(zip(THERMO_COLUMNS, raw, strict=True))
        row["pressure_GPa"] = row["pressure_bar"] * 1.0e-4
        row["volume_A3_per_atom"] = row["volume_A3"] / row["natoms"]
        values.append(row)
    if not values:
        raise ValueError(f"no thermodynamic values: {path}")
    return values


def five_ps_blocks(values: list[dict[str, float]], field: str, start: float, stop: float) -> list[float]:
    result: list[float] = []
    lower = start
    while lower < stop - 1.0e-12:
        upper = min(lower + 5.0, stop)
        selected = [row[field] for row in values if lower < row["time_ps"] <= upper + 1.0e-10]
        if not selected:
            raise ValueError(f"empty 5 ps block: {field}/{lower}/{upper}")
        result.append(statistics.fmean(selected))
        lower = upper
    return result


def historical_five_ps_blocks(
    values: list[dict[str, float]], field: str, start: float, stop: float
) -> list[float]:
    """Independently reconstruct the pre-outcome run validator's boundary bins."""
    count = math.ceil((stop - start) / 5.0)
    grouped: dict[int, list[float]] = {}
    for row in values:
        time = row["time_ps"]
        if start < time <= stop + 1.0e-10:
            index = min(int((time - start) / 5.0), count - 1)
            grouped.setdefault(index, []).append(row[field])
    if set(grouped) != set(range(count)):
        raise ValueError(f"historical 5 ps block population changed: {field}/{start}/{stop}")
    return [statistics.fmean(grouped[index]) for index in range(count)]


def length_gate(first: list[float], second: list[float]) -> dict[str, float | bool]:
    if len(first) != 10 or len(second) != 10:
        raise ValueError("length gate requires ten 5 ps blocks per half")
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    first_se = statistics.stdev(first) / math.sqrt(len(first))
    second_se = statistics.stdev(second) / math.sqrt(len(second))
    difference = abs(second_mean - first_mean)
    tolerance = max(
        0.005 * statistics.fmean((first_mean, second_mean)),
        2.0 * math.hypot(first_se, second_se),
    )
    return {
        "first_50ps_mean_A3_per_atom": first_mean,
        "second_50ps_mean_A3_per_atom": second_mean,
        "absolute_difference_A3_per_atom": difference,
        "tolerance_A3_per_atom": tolerance,
        "passed": difference <= tolerance,
    }


def trajectory_headers(path: Path) -> tuple[int, set[int], float, float]:
    frames = 0
    populations: set[int] = set()
    first_step: float | None = None
    final_step: float | None = None
    with path.open(encoding="utf-8", errors="strict") as handle:
        iterator = iter(handle)
        for line in iterator:
            if line.rstrip() != "ITEM: TIMESTEP":
                continue
            step = float(next(iterator).strip())
            frames += 1
            first_step = step if first_step is None else first_step
            final_step = step
            marker = next(iterator).rstrip()
            if marker != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"trajectory header order changed: {path}")
            populations.add(int(next(iterator).strip()))
    if first_step is None or final_step is None:
        raise ValueError(f"trajectory contains no frames: {path}")
    return frames, populations, first_step, final_step


def plan_rows(plan_path: Path, checks: Checks) -> list[dict[str, str]]:
    receipt_path = plan_path.with_suffix(plan_path.suffix + ".receipt.json")
    receipt = record(receipt_path)
    checks.true(receipt.get("plan_sha256") == sha256(plan_path), "A21 plan hash mismatch")
    plans = rows(plan_path, delimiter="\t")
    checks.true(len(plans) == 3 and len({row["task_id"] for row in plans}) == 3, "A21 plan population mismatch")
    checks.true({int(row["chemical_seed"]) for row in plans} == set(CHEMICAL_SEEDS), "A21 chemistry mismatch")
    checks.true(
        all(
            row["cohort"] == "dpa4_bcc_length_resolution_a21"
            and row["model_id"] == "DPA4__bcc_parent"
            and row["phase"] == "bcc"
            and int(row["temperature_K"]) == 1200
            and int(row["velocity_seed"]) == 20260905
            and int(row["n_prod"]) == 100000
            for row in plans
        ),
        "A21 plan differs from frozen scope",
    )
    return plans


def validate_raw_a21(
    plans: list[dict[str, str]], endpoint_path: Path, checks: Checks
) -> tuple[dict[int, dict[str, float]], list[dict[str, object]]]:
    endpoints_values = rows(endpoint_path)
    endpoints = {row["task_id"]: row for row in endpoints_values}
    checks.true(len(endpoints_values) == 3 and len(endpoints) == 3, "A21 endpoint population mismatch")
    promoted: dict[int, dict[str, float]] = {}
    reconciliation: list[dict[str, object]] = []
    for plan in plans:
        task_id = plan["task_id"]
        endpoint = endpoints.get(task_id)
        checks.true(endpoint is not None, f"missing A21 endpoint: {task_id}")
        root = Path(plan["output_root"]).resolve()
        production = root / "production"
        validation_path = production / "scientific_validation.json"
        samples_path = production / "thermo_samples.dat"
        trajectory_path = production / "trajectory_npt.lammpstrj"
        checks.true((production / "run.complete").is_file(), f"missing completion marker: {task_id}")
        checks.true(validation_path.is_file(), f"missing scientific validation: {task_id}")
        validation = record(validation_path)
        checks.true(validation.get("status") == "passed" and validation.get("errors") == [], f"A21 run is not passed: {task_id}")
        checks.true(endpoint["scientific_status"] == "passed", f"endpoint status mismatch: {task_id}")
        checks.true(endpoint["scientific_validation_sha256"] == sha256(validation_path), f"validation hash mismatch: {task_id}")
        checks.true(endpoint["trajectory_sha256"] == sha256(trajectory_path), f"trajectory hash mismatch: {task_id}")
        checks.true(int(endpoint["trajectory_size_bytes"]) == trajectory_path.stat().st_size, f"trajectory size mismatch: {task_id}")

        thermo = read_thermo(samples_path)
        checks.true(len(thermo) == 1001, f"sample population mismatch: {task_id}")
        checks.true({round(row["natoms"]) for row in thermo} == {2000}, f"atom population drift: {task_id}")
        checks.true(math.isclose(thermo[0]["time_ps"], 0.0) and math.isclose(thermo[-1]["time_ps"], 100.0), f"time range mismatch: {task_id}")
        checks.true(
            all(
                right["time_ps"] > left["time_ps"]
                for left, right in zip(thermo, thermo[1:])
            ),
            f"non-increasing sample time: {task_id}",
        )
        endpoint_values = [row for row in thermo if 50.0 < row["time_ps"] <= 100.0 + 1.0e-10]
        expected = {
            "temperature_K": statistics.fmean(row["temperature_K"] for row in endpoint_values),
            "pressure_GPa": statistics.fmean(row["pressure_GPa"] for row in endpoint_values),
            "volume_A3_per_atom": statistics.fmean(row["volume_A3_per_atom"] for row in endpoint_values),
        }
        for metric, value in expected.items():
            checks.close(endpoint[f"{metric}_mean"], value, f"raw endpoint {metric}: {task_id}")
            checks.close(validation[f"{metric}_mean"], value, f"scientific endpoint {metric}: {task_id}")

        equal_interval = length_gate(
            five_ps_blocks(thermo, "volume_A3_per_atom", 0.0, 50.0),
            five_ps_blocks(thermo, "volume_A3_per_atom", 50.0, 100.0),
        )
        historical = length_gate(
            historical_five_ps_blocks(thermo, "volume_A3_per_atom", 0.0, 50.0),
            historical_five_ps_blocks(thermo, "volume_A3_per_atom", 50.0, 100.0),
        )
        checks.true(equal_interval["passed"] is True, f"equal-interval length convergence failed: {task_id}")
        checks.true(historical["passed"] is True, f"historical-bin length convergence failed: {task_id}")
        checks.true(
            equal_interval["passed"] == historical["passed"],
            f"A23 boundary conventions disagree on the length decision: {task_id}",
        )
        length = validation.get("length_convergence")
        checks.true(isinstance(length, dict) and length.get("passed") is True, f"recorded length convergence failed: {task_id}")
        for field in (
            "first_50ps_mean_A3_per_atom",
            "second_50ps_mean_A3_per_atom",
            "absolute_difference_A3_per_atom",
            "tolerance_A3_per_atom",
        ):
            checks.close(length[field], float(historical[field]), f"recorded historical-bin {field}: {task_id}")
        reconciliation.append(
            {
                "task_id": task_id,
                "chemical_seed": int(plan["chemical_seed"]),
                "historical_integer_bin": historical,
                "equal_interval_lower_open_upper_closed": equal_interval,
                "classification_concordant": equal_interval["passed"] == historical["passed"],
            }
        )

        frames, populations, first_step, final_step = trajectory_headers(trajectory_path)
        checks.true(frames == 1001 and populations == {2000}, f"trajectory frame/population mismatch: {task_id}")
        checks.true(first_step == 0 and final_step == 100000, f"trajectory timestep mismatch: {task_id}")
        promoted[int(plan["chemical_seed"])] = expected
    checks.true(set(promoted) == set(CHEMICAL_SEEDS), "A21 promoted chemistry population mismatch")
    return promoted, reconciliation


def a19_chemistry(validation_path: Path, endpoint_path: Path, checks: Checks) -> dict[int, dict[str, float]]:
    validation = record(validation_path)
    checks.true(validation.get("status") == "passed" and validation.get("errors") == [], "A19 validation is not passed")
    bindings = validation.get("validated_outputs")
    checks.true(isinstance(bindings, dict) and bindings.get(str(endpoint_path.resolve())) == sha256(endpoint_path), "A19 endpoint binding mismatch")
    selected = [
        row for row in rows(endpoint_path)
        if row["model_id"] == "DPA3__bcc_parent"
        and row["simulated_phase"] == "bcc"
        and int(row["target_temperature_K"]) == 900
        and row["scientific_status"] == "passed"
    ]
    checks.true(len(selected) == 3 and {int(row["chemical_seed"]) for row in selected} == set(CHEMICAL_SEEDS), "A19 promoted chemistry mismatch")
    return {
        int(row["chemical_seed"]): {
            metric: float(row[f"{metric}_mean"]) for metric in METRICS
        }
        for row in selected
    }


def expected_chemistry(
    a11: list[dict[str, str]],
    a12: list[dict[str, str]],
    old: dict[int, dict[str, float]],
    new: dict[int, dict[str, float]],
) -> dict[tuple[str, str, int], dict[int, dict[str, float]]]:
    result: dict[tuple[str, str, int], dict[int, dict[str, float]]] = {}
    for model_id, _, _, phase in CURVES:
        for temperature in TEMPERATURES:
            key = (model_id, phase, temperature)
            if model_id == "DPA3__bcc_parent" and temperature == 900:
                result[key] = old
            elif model_id == "DPA4__bcc_parent" and temperature == 1200:
                result[key] = new
            else:
                result[key] = baseline_chemistry(a11 if temperature in (300, 1200) else a12, model_id, phase, temperature)
    return result


def validate_group_values(
    path: Path,
    chemistry: dict[tuple[str, str, int], dict[int, dict[str, float]]],
    checks: Checks,
) -> dict[tuple[str, str, int], dict[str, str]]:
    values = rows(path)
    groups = {(row["model_id"], row["simulated_phase"], int(row["temperature_K"])): row for row in values}
    checks.true(len(values) == 12 and set(groups) == set(chemistry), "A21 curve-group population mismatch")
    for key, chemical_values in chemistry.items():
        row = groups[key]
        checks.true(row["status"] == "eligible" and int(row["eligible_chemical_replicas"]) == 3, f"A21 group is not complete: {key}")
        checks.true(row["ci_method"] == "two-sided Student-t, df=2", f"interval method mismatch: {key}")
        for metric in METRICS:
            expected = interval([chemical_values[seed][metric] for seed in CHEMICAL_SEEDS])
            for suffix, value in zip(("", "_sd", "_ci95_low", "_ci95_high"), expected, strict=True):
                checks.close(row[f"{metric}_mean{suffix}"], value, f"{key}/{metric}{suffix}")
    return groups


def validate_panel(
    path: Path,
    groups: dict[tuple[str, str, int], dict[str, str]],
    a11: list[dict[str, str]],
    checks: Checks,
) -> None:
    values = rows(path)
    panel = {(row["model_id"], row["simulated_phase"], int(row["temperature_K"])): row for row in values}
    expected = {(model_id, phase, temperature) for model_id, _, _, phase in PANEL_BRANCHES for temperature in (300, 1200)}
    checks.true(len(values) == 12 and set(panel) == expected, "A21 panel-b/d population mismatch")
    for key, row in panel.items():
        if key in groups:
            checks.true(row == groups[key], f"panel group differs from curve group: {key}")
        elif key == ("DPA4__fcc_parent", "fcc", 300):
            chemistry = baseline_chemistry(a11, key[0], key[1], key[2])
            checks.true(row["status"] == "eligible", f"DPA4/FCC 300 K audit point missing: {key}")
            expected_mean = statistics.fmean(chemistry[seed]["volume_A3_per_atom"] for seed in CHEMICAL_SEEDS)
            checks.close(row["volume_A3_per_atom_mean"], expected_mean, "DPA4/FCC 300 K panel mean")
        else:
            checks.true(row["status"] == "blocked_incomplete_a21_group", f"panel block missing: {key}")
            checks.true(not row["volume_A3_per_atom_mean"], f"blocked panel leaks value: {key}")


def fit_slope(volumes: list[float]) -> float:
    x_mean = statistics.fmean(TEMPERATURES)
    logs = [math.log(value) for value in volumes]
    y_mean = statistics.fmean(logs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(TEMPERATURES, logs, strict=True)) / sum((x - x_mean) ** 2 for x in TEMPERATURES)


def validate_responses(
    path: Path,
    chemistry: dict[tuple[str, str, int], dict[int, dict[str, float]]],
    checks: Checks,
) -> None:
    values = rows(path)
    responses = {(row["model_id"], row["simulated_phase"]): row for row in values}
    expected = {(model_id, phase) for model_id, _, _, phase in CURVES}
    checks.true(len(values) == 3 and set(responses) == expected, "A21 response population mismatch")
    for model_id, phase in expected:
        row = responses[(model_id, phase)]
        checks.true(row["status"] == "eligible", f"A21 response is not eligible: {model_id}")
        for low_temperature, high_temperature in zip(TEMPERATURES[:-1], TEMPERATURES[1:], strict=True):
            deltas = [chemistry[(model_id, phase, high_temperature)][seed]["volume_A3_per_atom"] - chemistry[(model_id, phase, low_temperature)][seed]["volume_A3_per_atom"] for seed in CHEMICAL_SEEDS]
            mean, _, low, high = interval(deltas)
            field = f"volume_delta_{high_temperature}_minus_{low_temperature}_A3_per_atom"
            checks.close(row[field], mean, field)
            checks.close(row[f"{field}_ci95_low"], low, field + "/low")
            checks.close(row[f"{field}_ci95_high"], high, field + "/high")
        slopes = [fit_slope([chemistry[(model_id, phase, temperature)][seed]["volume_A3_per_atom"] for temperature in TEMPERATURES]) for seed in CHEMICAL_SEEDS]
        alpha, _, alpha_low, alpha_high = interval(slopes)
        checks.close(row["volumetric_alpha_per_K"], alpha, f"{model_id}/alpha")
        checks.close(row["volumetric_alpha_per_K_ci95_low"], alpha_low, f"{model_id}/alpha low")
        checks.close(row["volumetric_alpha_per_K_ci95_high"], alpha_high, f"{model_id}/alpha high")
        checks.close(row["isotropic_linear_alpha_per_K"], alpha / 3.0, f"{model_id}/linear alpha")


def validate_excluded(path: Path, a11: list[dict[str, str]], a12: list[dict[str, str]], checks: Checks) -> None:
    values = rows(path)
    keyed = {int(row["temperature_K"]): row for row in values}
    checks.true(len(values) == 4 and set(keyed) == set(TEMPERATURES), "DPA4/FCC excluded-audit population mismatch")
    for temperature, row in keyed.items():
        checks.true(row["model_id"] == "DPA4__fcc_parent" and row["simulated_phase"] == "fcc", "excluded audit identity mismatch")
        checks.true("excluded from Figure 7c" in row["source"], "excluded audit lacks scope label")
        if temperature == 1200:
            checks.true(row["status"] == "blocked_incomplete_a21_group" and not row["volume_A3_per_atom_mean"], "excluded 1200 K block mismatch")
        else:
            source = a11 if temperature == 300 else a12
            chemistry = baseline_chemistry(source, "DPA4__fcc_parent", "fcc", temperature)
            checks.true(row["status"] == "eligible", f"excluded audit value missing: {temperature}")
            checks.close(row["volume_A3_per_atom_mean"], statistics.fmean(chemistry[seed]["volume_A3_per_atom"] for seed in CHEMICAL_SEEDS), f"excluded audit mean: {temperature}")


def validate(args: argparse.Namespace) -> dict[str, object]:
    checks = Checks()
    plan_path = args.a21_plan.resolve()
    plans = plan_rows(plan_path, checks)
    analysis_path = args.analysis_receipt.resolve()
    analysis = record(analysis_path)
    checks.true(analysis.get("status") == "passed" and analysis.get("all_a21_rows_passed") is True, "A21 analysis receipt is not passed")
    output_paths = {
        "run_endpoints": args.run_endpoints.resolve(),
        "groups": args.groups.resolve(),
        "panel_groups": args.panel_groups.resolve(),
        "response": args.response.resolve(),
        "excluded_curve_audit": args.excluded_curve_audit.resolve(),
    }
    bindings = analysis.get("outputs")
    checks.true(isinstance(bindings, dict), "A21 analysis has no output bindings")
    for name, path in output_paths.items():
        bound = bindings.get(name) if isinstance(bindings, dict) else None
        checks.true(isinstance(bound, dict) and Path(str(bound.get("path", ""))).resolve() == path, f"A21 output path mismatch: {name}")
        checks.true(isinstance(bound, dict) and bound.get("sha256") == sha256(path), f"A21 output hash mismatch: {name}")

    checks.true(A23_PROTOCOL.is_file(), "A23 block-boundary reconciliation protocol is missing")
    new, boundary_reconciliation = validate_raw_a21(plans, output_paths["run_endpoints"], checks)
    old = a19_chemistry(args.a19_validation.resolve(), args.a19_endpoints.resolve(), checks)
    a11_path = args.a11_replicas.resolve()
    a12_path = args.a12_replicas.resolve()
    a11 = rows(a11_path)
    a12 = rows(a12_path)
    chemistry = expected_chemistry(a11, a12, old, new)
    groups = validate_group_values(output_paths["groups"], chemistry, checks)
    validate_panel(output_paths["panel_groups"], groups, a11, checks)
    validate_responses(output_paths["response"], chemistry, checks)
    validate_excluded(output_paths["excluded_curve_audit"], a11, a12, checks)
    return {
        "schema_version": 1,
        "status": "passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks.count,
        "errors": [],
        "analysis_receipt": str(analysis_path),
        "analysis_receipt_sha256": sha256(analysis_path),
        "a21_plan": str(plan_path),
        "a21_plan_sha256": sha256(plan_path),
        "a19_validation": str(args.a19_validation.resolve()),
        "a19_validation_sha256": sha256(args.a19_validation.resolve()),
        "a23_protocol": str(A23_PROTOCOL),
        "a23_protocol_sha256": sha256(A23_PROTOCOL),
        "baseline_replica_tables": {str(path): sha256(path) for path in (a11_path, a12_path)},
        "validated_outputs": {str(path): sha256(path) for path in output_paths.values()},
        "raw_trajectory_headers_replayed": True,
        "block_boundary_reconciliation": {
            "status": "passed",
            "classification_concordant_rows": len(boundary_reconciliation),
            "rows": boundary_reconciliation,
        },
        "dpa4_fcc_excluded_from_figure7c": True,
    }


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
