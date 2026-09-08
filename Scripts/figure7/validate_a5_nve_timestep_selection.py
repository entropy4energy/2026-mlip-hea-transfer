#!/usr/bin/env python3
"""Independently replay the A5 NVE timestep decision for all six branches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


MODEL_PHASE = {
    f"{architecture}__{phase}_parent": phase
    for architecture in ("DPA2", "DPA3", "DPA4")
    for phase in ("bcc", "fcc")
}
TEMPERATURES = (300.0, 1200.0)
TIMESTEPS_PS = (0.0005, 0.001)
ABSOLUTE_LIMIT = 0.1
RELATIVE_MULTIPLIER = 2.0
RELATIVE_OFFSET = 0.01


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


def close(left: object, right: float, *, absolute: float = 1.0e-14) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=1.0e-12, abs_tol=absolute)
    except (TypeError, ValueError):
        return False


def expect_equal(
    errors: list[str], checks: list[int], label: str, observed: object, expected: object
) -> None:
    checks[0] += 1
    if observed != expected:
        errors.append(f"{label}: observed {observed!r}, expected {expected!r}")


def expect_close(
    errors: list[str], checks: list[int], label: str, observed: object, expected: float
) -> None:
    checks[0] += 1
    if not close(observed, expected):
        errors.append(f"{label}: observed {observed!r}, expected {expected!r}")


def unique_by(
    values: list[dict[str, str]], key, label: str, errors: list[str]
) -> dict[object, dict[str, str]]:
    result: dict[object, dict[str, str]] = {}
    for row in values:
        item_key = key(row)
        if item_key in result:
            errors.append(f"duplicate {label} key: {item_key}")
        result[item_key] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--dependency-ledger", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selector-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation_path = args.validation.resolve()
    ledger_path = args.dependency_ledger.resolve()
    comparisons_path = args.comparisons.resolve()
    selection_path = args.selection.resolve()
    receipt_path = args.selector_receipt.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    errors: list[str] = []
    checks = [0]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    bound_inputs = (
        ("validation", "validation_sha256", validation_path),
        ("dependency_ledger", "dependency_ledger_sha256", ledger_path),
        ("comparisons", "comparisons_sha256", comparisons_path),
        ("selection", "selection_sha256", selection_path),
    )
    for path_field, hash_field, path in bound_inputs:
        expect_equal(errors, checks, f"receipt {path_field}", Path(str(receipt.get(path_field, ""))).resolve(), path)
        expect_equal(errors, checks, f"receipt {hash_field}", receipt.get(hash_field), sha256(path))

    plan_path = Path(str(validation.get("plan", ""))).resolve()
    summary_path = Path(str(validation.get("summary_csv", ""))).resolve()
    expect_equal(errors, checks, "validation plan hash", validation.get("plan_sha256"), sha256(plan_path))
    expect_equal(errors, checks, "validation summary hash", validation.get("summary_csv_sha256"), sha256(summary_path))
    expect_equal(errors, checks, "receipt plan path", Path(str(receipt.get("plan", ""))).resolve(), plan_path)
    expect_equal(errors, checks, "receipt plan hash", receipt.get("plan_sha256"), sha256(plan_path))
    expect_equal(errors, checks, "receipt summary path", Path(str(receipt.get("validated_summary", ""))).resolve(), summary_path)
    expect_equal(errors, checks, "receipt summary hash", receipt.get("validated_summary_sha256"), sha256(summary_path))
    expect_equal(errors, checks, "planned/validated task count", validation.get("planned_tasks"), validation.get("validated_tasks"))

    plan_rows = rows(plan_path, "\t")
    summary_rows = rows(summary_path)
    ledger_rows = rows(ledger_path, "\t")
    comparison_rows = rows(comparisons_path)
    selection_rows = rows(selection_path)
    plan = unique_by(plan_rows, lambda row: row["task_id"], "plan", errors)
    summary = unique_by(summary_rows, lambda row: row["task_id"], "summary", errors)
    ledger = unique_by(
        ledger_rows,
        lambda row: (
            row["model_id"],
            row["simulated_phase"],
            float(row["temperature_K"]),
            float(row["timestep_ps"]),
        ),
        "ledger",
        errors,
    )
    comparisons = unique_by(
        comparison_rows,
        lambda row: (row["model_id"], row["simulated_phase"], float(row["temperature_K"])),
        "comparison",
        errors,
    )
    selections = unique_by(
        selection_rows,
        lambda row: (row["model_id"], row["simulated_phase"]),
        "selection",
        errors,
    )

    expected_pairs = set(MODEL_PHASE.items())
    expected_temperatures = {
        (model_id, phase, temperature)
        for model_id, phase in expected_pairs
        for temperature in TEMPERATURES
    }
    expected_ledger = {
        (model_id, phase, temperature, timestep)
        for model_id, phase in expected_pairs
        for temperature in TEMPERATURES
        for timestep in TIMESTEPS_PS
    }
    expect_equal(errors, checks, "ledger row count", len(ledger_rows), 24)
    expect_equal(errors, checks, "ledger population", set(ledger), expected_ledger)
    expect_equal(errors, checks, "comparison row count", len(comparison_rows), 12)
    expect_equal(errors, checks, "comparison population", set(comparisons), expected_temperatures)
    expect_equal(errors, checks, "selection row count", len(selection_rows), 6)
    expect_equal(errors, checks, "selection population", set(selections), expected_pairs)
    expect_equal(errors, checks, "plan/summary population", set(plan), set(summary))

    runnable_ids = {
        row["runnable_task_id"] for row in ledger_rows if row.get("runnable_task_id", "")
    }
    expect_equal(errors, checks, "ledger/plan runnable population", runnable_ids, set(plan))
    expected_branch: dict[tuple[str, str], tuple[str, float | None]] = {}
    decisions_by_branch: dict[tuple[str, str], list[tuple[str, bool]]] = defaultdict(list)

    for key in sorted(expected_temperatures):
        model_id, phase, temperature = key
        output_row = comparisons.get(key)
        if output_row is None:
            continue
        ledger_pair = [ledger[(model_id, phase, temperature, timestep)] for timestep in TIMESTEPS_PS if (model_id, phase, temperature, timestep) in ledger]
        if len(ledger_pair) != 2:
            continue
        statuses = {row["dependency_status"] for row in ledger_pair}
        expect_equal(errors, checks, f"{key} matched dependency status", len(statuses), 1)
        dependency = next(iter(statuses)) if len(statuses) == 1 else "invalid"
        expect_equal(errors, checks, f"{key} comparison dependency status", output_row["dependency_status"], dependency)
        for row in ledger_pair:
            expect_equal(errors, checks, f"{key} ledger calculation", row["calculation"], "nve")
            expect_equal(errors, checks, f"{key} ledger atom count", row["natoms"], "2000")
            expect_equal(errors, checks, f"{key} ledger chemical seed", row["chemical_seed"], "20260825")
            expect_equal(errors, checks, f"{key} ledger velocity seed", row["velocity_seed"], "20260901")

        if dependency == "blocked_upstream_failure":
            for field, expected in (
                ("reference_0p5fs_status", "not_run_blocked"),
                ("trial_1fs_status", "not_run_blocked"),
                ("trial_absolute_gate_passed", "false"),
                ("trial_relative_gate_passed", "false"),
                ("trial_1fs_passed", "false"),
                ("temperature_decision_status", "blocked_upstream_failure"),
            ):
                expect_equal(errors, checks, f"{key} {field}", output_row[field], expected)
            decisions_by_branch[(model_id, phase)].append(("blocked_upstream_failure", False))
            continue
        if dependency != "runnable_pending_execution":
            errors.append(f"{key}: unsupported dependency status {dependency!r}")
            continue

        task_rows: dict[float, tuple[dict[str, str], dict[str, str]]] = {}
        for ledger_row in ledger_pair:
            timestep = float(ledger_row["timestep_ps"])
            task_id = ledger_row["runnable_task_id"]
            if task_id not in plan or task_id not in summary:
                errors.append(f"{key}: missing runnable task {task_id}")
                continue
            task_rows[timestep] = (plan[task_id], summary[task_id])
        if set(task_rows) != set(TIMESTEPS_PS):
            continue
        reference_plan, reference = task_rows[0.0005]
        trial_plan, trial = task_rows[0.001]
        for field in ("data_path", "chemical_seed", "velocity_seed"):
            expect_equal(errors, checks, f"{key} matched {field}", reference_plan[field], trial_plan[field])
        expect_close(errors, checks, f"{key} reference duration", int(reference_plan["n_prod"]) * 0.0005, 50.0)
        expect_close(errors, checks, f"{key} trial duration", int(trial_plan["n_prod"]) * 0.001, 50.0)

        reference_passed = reference["status"] == "passed"
        trial_completed = trial["status"] == "passed"
        reference_drift = float(reference["absolute_drift_meV_per_atom_ps"]) if reference_passed else math.nan
        trial_drift = float(trial["absolute_drift_meV_per_atom_ps"]) if trial_completed else math.nan
        absolute_pass = trial_completed and trial_drift <= ABSOLUTE_LIMIT + 1.0e-15
        relative_limit = RELATIVE_MULTIPLIER * reference_drift + RELATIVE_OFFSET if reference_passed else math.nan
        relative_pass = reference_passed and trial_completed and trial_drift <= relative_limit + 1.0e-15
        trial_pass = reference_passed and absolute_pass and relative_pass
        expected_temperature_status = (
            "passed_1fs" if trial_pass else "use_0p5fs" if reference_passed else "blocked_reference_failure"
        )
        for field, expected in (
            ("reference_0p5fs_status", reference["status"]),
            ("trial_1fs_status", trial["status"]),
            ("trial_absolute_gate_passed", str(absolute_pass).lower()),
            ("trial_relative_gate_passed", str(relative_pass).lower()),
            ("trial_1fs_passed", str(trial_pass).lower()),
            ("temperature_decision_status", expected_temperature_status),
        ):
            expect_equal(errors, checks, f"{key} {field}", output_row[field], expected)
        if reference_passed:
            expect_close(errors, checks, f"{key} reference drift", output_row["reference_0p5fs_abs_drift_meV_per_atom_ps"], reference_drift)
            expect_close(errors, checks, f"{key} relative limit", output_row["trial_relative_limit_meV_per_atom_ps"], relative_limit)
        if trial_completed:
            expect_close(errors, checks, f"{key} trial drift", output_row["trial_1fs_abs_drift_meV_per_atom_ps"], trial_drift)
        decisions_by_branch[(model_id, phase)].append((expected_temperature_status, trial_pass))

    for pair in sorted(expected_pairs):
        decisions = decisions_by_branch.get(pair, [])
        if len(decisions) != 2:
            errors.append(f"{pair}: expected two temperature decisions, found {len(decisions)}")
            continue
        statuses = [status for status, _ in decisions]
        if "blocked_upstream_failure" in statuses:
            expected_branch[pair] = ("blocked_upstream_failure", None)
        elif "blocked_reference_failure" in statuses:
            expected_branch[pair] = ("blocked_reference_failure", None)
        elif all(passed for _, passed in decisions):
            expected_branch[pair] = ("selected", 0.001)
        else:
            expected_branch[pair] = ("selected", 0.0005)

    for pair, (expected_status, expected_timestep) in expected_branch.items():
        row = selections.get(pair)
        if row is None:
            continue
        expect_equal(errors, checks, f"{pair} selection status", row["status"], expected_status)
        if expected_timestep is None:
            expect_equal(errors, checks, f"{pair} blocked timestep", row["accepted_timestep_ps"], "")
        else:
            expect_close(errors, checks, f"{pair} accepted timestep", row["accepted_timestep_ps"], expected_timestep)

    status_counts = Counter(row["status"] for row in selection_rows)
    scientific_status = "selected" if status_counts == {"selected": 6} else "partial_selection_with_blocked_branches"
    expect_equal(errors, checks, "receipt status", receipt.get("status"), "passed")
    expect_equal(errors, checks, "receipt scientific status", receipt.get("scientific_status"), scientific_status)
    expect_equal(errors, checks, "receipt status counts", receipt.get("selection_status_counts"), dict(status_counts))
    expect_equal(errors, checks, "receipt comparison count", receipt.get("comparison_rows"), 12)
    expect_equal(errors, checks, "receipt selection count", receipt.get("selection_rows"), 6)
    expect_close(errors, checks, "receipt absolute drift limit", receipt.get("absolute_drift_limit_meV_per_atom_ps"), ABSOLUTE_LIMIT)

    result = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "checks": checks[0],
        "errors": errors,
        "validation": str(validation_path),
        "validation_sha256": sha256(validation_path),
        "dependency_ledger": str(ledger_path),
        "dependency_ledger_sha256": sha256(ledger_path),
        "selector_receipt": str(receipt_path),
        "selector_receipt_sha256": sha256(receipt_path),
        "comparison_rows": len(comparison_rows),
        "selection_rows": len(selection_rows),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(output_path, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
