#!/usr/bin/env python3
"""Select A5 timesteps while retaining all six intended model/phase branches."""

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
CHEMICAL_SEED = 20260825
VELOCITY_SEED = 20260901
PHYSICAL_DURATION_PS = 50.0
ABSOLUTE_DRIFT_LIMIT = 0.1
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


def atomic_csv(path: Path, values: list[dict[str, object]]) -> None:
    if not values:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
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


def load_inputs(
    validation_path: Path, ledger_path: Path
) -> tuple[
    dict[str, object],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    list[dict[str, str]],
    Path,
    Path,
]:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("planned_tasks") != validation.get("validated_tasks"):
        raise ValueError("NVE validation lacks one row per executable-plan task")
    plan_path = Path(str(validation["plan"])).resolve()
    summary_path = Path(str(validation["summary_csv"])).resolve()
    if validation.get("plan_sha256") != sha256(plan_path):
        raise ValueError("NVE validation plan hash mismatch")
    if validation.get("summary_csv_sha256") != sha256(summary_path):
        raise ValueError("NVE validation summary hash mismatch")
    plan_rows = rows(plan_path, "\t")
    summary_rows = rows(summary_path)
    plans = {row["task_id"]: row for row in plan_rows}
    summaries = {row["task_id"]: row for row in summary_rows}
    if (
        len(plans) != len(plan_rows)
        or len(summaries) != len(summary_rows)
        or set(plans) != set(summaries)
    ):
        raise ValueError("NVE plan/validated-summary task population mismatch")
    if plan_rows and {row["calculation"] for row in plan_rows} != {"nve"}:
        raise ValueError("NVE selection received another executable calculation")

    ledger_rows = rows(ledger_path, "\t")
    if len(ledger_rows) != 24 or {row["calculation"] for row in ledger_rows} != {"nve"}:
        raise ValueError("A5 NVE dependency ledger is not the complete 24-row population")
    intended_ids = [row["intended_task_id"] for row in ledger_rows]
    if len(set(intended_ids)) != 24:
        raise ValueError("A5 NVE dependency ledger has duplicate intended IDs")
    runnable_ids = {
        row["runnable_task_id"] for row in ledger_rows if row["runnable_task_id"]
    }
    if runnable_ids != set(plans):
        raise ValueError("A5 NVE dependency ledger and executable plan differ")
    receipt_path = plan_path.with_suffix(plan_path.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("plan_sha256") != sha256(plan_path):
        raise ValueError("A5 NVE plan receipt hash mismatch")
    if receipt.get("dependency_ledger_sha256") != sha256(ledger_path):
        raise ValueError("A5 NVE plan receipt does not bind the dependency ledger")
    if receipt.get("profile") != "A5_two_temperature_dynamic_core":
        raise ValueError("NVE plan is not an A5 core plan")
    return validation, plans, summaries, ledger_rows, plan_path, summary_path


def ledger_key(row: dict[str, str]) -> tuple[str, str, float, float]:
    return (
        row["model_id"],
        row["simulated_phase"],
        float(row["temperature_K"]),
        float(row["timestep_ps"]),
    )


def validate_ledger_population(ledger_rows: list[dict[str, str]]) -> None:
    expected = {
        (model_id, phase, temperature, timestep)
        for model_id, phase in MODEL_PHASE.items()
        for temperature in TEMPERATURES
        for timestep in TIMESTEPS_PS
    }
    observed = {ledger_key(row) for row in ledger_rows}
    if observed != expected:
        raise ValueError("A5 NVE dependency ledger key population is incomplete")
    for row in ledger_rows:
        if (
            MODEL_PHASE.get(row["model_id"]) != row["simulated_phase"]
            or int(row["chemical_seed"]) != CHEMICAL_SEED
            or int(row["velocity_seed"]) != VELOCITY_SEED
            or int(row["natoms"]) != 2000
            or row["dependency_status"]
            not in {"runnable_pending_execution", "blocked_upstream_failure"}
        ):
            raise ValueError(f"invalid A5 NVE ledger row: {row['intended_task_id']}")


def comparison_for_temperature(
    model_id: str,
    phase: str,
    temperature: float,
    ledger_by_key: dict[tuple[str, str, float, float], dict[str, str]],
    plans: dict[str, dict[str, str]],
    summaries: dict[str, dict[str, str]],
) -> dict[str, object]:
    ledger_pair = {
        timestep: ledger_by_key[(model_id, phase, temperature, timestep)]
        for timestep in TIMESTEPS_PS
    }
    dependency_statuses = {
        row["dependency_status"] for row in ledger_pair.values()
    }
    if len(dependency_statuses) != 1:
        raise ValueError(f"{model_id}/{phase}/{temperature:g} K has split timestep dependencies")
    base: dict[str, object] = {
        "model_id": model_id,
        "architecture": model_id.split("__", 1)[0],
        "training_parent": phase,
        "simulated_phase": phase,
        "temperature_K": int(temperature),
        "chemical_seed": CHEMICAL_SEED,
        "velocity_seed": VELOCITY_SEED,
        "dependency_status": next(iter(dependency_statuses)),
        "reference_0p5fs_status": "not_run_blocked",
        "reference_0p5fs_abs_drift_meV_per_atom_ps": "",
        "trial_1fs_status": "not_run_blocked",
        "trial_1fs_abs_drift_meV_per_atom_ps": "",
        "trial_absolute_limit_meV_per_atom_ps": ABSOLUTE_DRIFT_LIMIT,
        "trial_relative_limit_meV_per_atom_ps": "",
        "trial_absolute_gate_passed": "false",
        "trial_relative_gate_passed": "false",
        "trial_1fs_passed": "false",
        "temperature_decision_status": "blocked_upstream_failure",
        "blocking_reason": "",
    }
    if dependency_statuses == {"blocked_upstream_failure"}:
        reasons = sorted(
            {
                row.get("upstream_failure_reason", "").strip()
                or row.get("upstream_status", "blocked upstream")
                for row in ledger_pair.values()
            }
        )
        base["blocking_reason"] = "; ".join(reasons)
        return base

    task_rows: dict[float, tuple[dict[str, str], dict[str, str]]] = {}
    for timestep, ledger in ledger_pair.items():
        task_id = ledger["runnable_task_id"]
        if task_id not in plans or task_id not in summaries:
            raise ValueError(f"missing runnable NVE task {task_id}")
        task_rows[timestep] = plans[task_id], summaries[task_id]
    reference_plan, reference = task_rows[0.0005]
    trial_plan, trial = task_rows[0.001]
    for field in ("data_path", "chemical_seed", "velocity_seed"):
        if reference_plan[field] != trial_plan[field]:
            raise ValueError(f"{model_id}/{phase}/{temperature:g} K timestep pair differs in {field}")
    for timestep, plan in ((0.0005, reference_plan), (0.001, trial_plan)):
        duration = int(plan["n_prod"]) * timestep
        if not math.isclose(duration, PHYSICAL_DURATION_PS, abs_tol=1.0e-12):
            raise ValueError(f"{plan['task_id']}: NVE duration is not 50 ps")

    reference_passed = reference["status"] == "passed"
    trial_completed = trial["status"] == "passed"
    reference_drift = (
        float(reference["absolute_drift_meV_per_atom_ps"])
        if reference_passed
        else math.nan
    )
    trial_drift = (
        float(trial["absolute_drift_meV_per_atom_ps"])
        if trial_completed
        else math.nan
    )
    absolute_gate = trial_completed and trial_drift <= ABSOLUTE_DRIFT_LIMIT + 1.0e-15
    relative_limit = (
        RELATIVE_MULTIPLIER * reference_drift + RELATIVE_OFFSET
        if reference_passed
        else math.nan
    )
    relative_gate = (
        reference_passed
        and trial_completed
        and trial_drift <= relative_limit + 1.0e-15
    )
    trial_passed = reference_passed and absolute_gate and relative_gate
    base.update(
        {
            "reference_0p5fs_status": reference["status"],
            "reference_0p5fs_abs_drift_meV_per_atom_ps": (
                reference_drift if reference_passed else ""
            ),
            "trial_1fs_status": trial["status"],
            "trial_1fs_abs_drift_meV_per_atom_ps": trial_drift if trial_completed else "",
            "trial_relative_limit_meV_per_atom_ps": (
                relative_limit if reference_passed else ""
            ),
            "trial_absolute_gate_passed": str(absolute_gate).lower(),
            "trial_relative_gate_passed": str(relative_gate).lower(),
            "trial_1fs_passed": str(trial_passed).lower(),
            "temperature_decision_status": (
                "passed_1fs"
                if trial_passed
                else "use_0p5fs"
                if reference_passed
                else "blocked_reference_failure"
            ),
            "blocking_reason": (
                ""
                if reference_passed
                else reference.get("failure_reason", "") or "0.5 fs reference did not pass"
            ),
        }
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--dependency-ledger", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    validation_path = args.validation.resolve()
    ledger_path = args.dependency_ledger.resolve()
    validation, plans, summaries, ledger_rows, plan_path, summary_path = load_inputs(
        validation_path, ledger_path
    )
    validate_ledger_population(ledger_rows)
    ledger_by_key = {ledger_key(row): row for row in ledger_rows}

    comparisons: list[dict[str, object]] = []
    by_pair: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for model_id, phase in sorted(MODEL_PHASE.items()):
        for temperature in TEMPERATURES:
            result = comparison_for_temperature(
                model_id,
                phase,
                temperature,
                ledger_by_key,
                plans,
                summaries,
            )
            comparisons.append(result)
            by_pair[(model_id, phase)].append(result)

    selections: list[dict[str, object]] = []
    for (model_id, phase), decisions in sorted(by_pair.items()):
        if len(decisions) != 2 or {row["temperature_K"] for row in decisions} != {300, 1200}:
            raise ValueError(f"{model_id}/{phase}: incomplete A5 temperature-decision pair")
        upstream_blocked = any(
            row["dependency_status"] == "blocked_upstream_failure" for row in decisions
        )
        reference_passed = not upstream_blocked and all(
            row["reference_0p5fs_status"] == "passed" for row in decisions
        )
        trial_passed = reference_passed and all(
            row["trial_1fs_passed"] == "true" for row in decisions
        )
        if upstream_blocked:
            status = "blocked_upstream_failure"
            accepted: float | str = ""
            reason = "at least one A5 NVT/NVE dependency row was blocked upstream"
        elif not reference_passed:
            status = "blocked_reference_failure"
            accepted = ""
            reason = "at least one matched 0.5 fs reference did not pass"
        elif trial_passed:
            status = "selected"
            accepted = 0.001
            reason = "1 fs passed absolute and matched-0.5-fs relative drift gates at 300 and 1200 K"
        else:
            status = "selected"
            accepted = 0.0005
            reason = "1 fs failed at least one fixed gate; use the matched 0.5 fs reference"
        selections.append(
            {
                "model_id": model_id,
                "architecture": model_id.split("__", 1)[0],
                "training_parent": phase,
                "simulated_phase": phase,
                "status": status,
                "accepted_timestep_ps": accepted,
                "selection_reason": reason,
                "temperatures_K": "300;1200",
                "chemical_seed": CHEMICAL_SEED,
                "velocity_seed": VELOCITY_SEED,
                "source_validation": str(validation_path),
                "source_validation_sha256": sha256(validation_path),
                "dependency_ledger": str(ledger_path),
                "dependency_ledger_sha256": sha256(ledger_path),
            }
        )

    comparisons_path = args.comparisons.resolve()
    selection_path = args.selection.resolve()
    receipt_path = args.receipt.resolve()
    for path in (comparisons_path, selection_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    atomic_csv(comparisons_path, comparisons)
    atomic_csv(selection_path, selections)
    counts = Counter(str(row["status"]) for row in selections)
    receipt = {
        "schema_version": 2,
        "status": "passed",
        "scientific_status": (
            "selected" if counts == {"selected": 6} else "partial_selection_with_blocked_branches"
        ),
        "selection_status_counts": dict(counts),
        "validation": str(validation_path),
        "validation_sha256": sha256(validation_path),
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "validated_summary": str(summary_path),
        "validated_summary_sha256": sha256(summary_path),
        "dependency_ledger": str(ledger_path),
        "dependency_ledger_sha256": sha256(ledger_path),
        "comparisons": str(comparisons_path),
        "comparisons_sha256": sha256(comparisons_path),
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "selector": str(Path(__file__).resolve()),
        "selector_sha256": sha256(Path(__file__).resolve()),
        "comparison_rows": len(comparisons),
        "selection_rows": len(selections),
        "intended_model_phase_pairs": 6,
        "absolute_drift_limit_meV_per_atom_ps": ABSOLUTE_DRIFT_LIMIT,
        "relative_drift_rule": "trial <= 2*reference + 0.01 meV/atom/ps",
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
