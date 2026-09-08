#!/usr/bin/env python3
"""Build the A5 two-temperature dynamic core without hiding blocked rows.

The executable TSV contains only tasks whose exact prerequisites passed.  A
separate TSV contains the complete intended population, including permanent
``blocked_upstream_failure`` rows.  The plan receipt binds both files and all
evidence used for the eligibility decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import build_dynamic_stage_plan as base


CAMPAIGN_ID = base.CAMPAIGN_ID
A5_PROTOCOL_SHA256 = "9da53546855fb4fe406edb8562c707aabbe7cd5a3b5bea3ad139402c802be9d9"
A9_PROTOCOL_SHA256 = "37a4b6cf1554fe436bdf7f227035457f93b06bd674415ae929fca5947e6c036e"
SUPPORTED_PROTOCOL_SHA256 = {A5_PROTOCOL_SHA256, A9_PROTOCOL_SHA256}
BENCHMARK_SUMMARY_SHA256 = "c5da5cbcaf76d7775ff880ca8f5620bed392ea64cf8e18eae1b5d12de13a635a"
BENCHMARK_VALIDATION_SHA256 = "4118c81a802c12066bf2a8f416a81d2ae69abecae26204921adc122e04415eec"

CORE_CHEMICAL_SEEDS = (20260825, 20262843, 20264861)
CORE_VELOCITY_SEEDS = (20260901, 20260903)
CORE_TEMPERATURES = (300, 1200)
TIMESTEP_GATE_CHEMICAL_SEED = 20260825
TIMESTEP_GATE_VELOCITY_SEED = 20260901
TIMESTEP_CANDIDATES_PS = (0.0005, 0.001)
EQUILIBRATION_PS = 40.0
NVE_PRODUCTION_PS = 50.0
NPT_PRODUCTION_PS = 50.0
NPT_LENGTH_SENSITIVITY_PS = 100.0

LEDGER_FIELDS = (
    "intended_task_id",
    "calculation",
    "model_id",
    "architecture",
    "training_parent",
    "simulated_phase",
    "chemical_seed",
    "velocity_seed",
    "temperature_K",
    "timestep_ps",
    "natoms",
    "length_sensitivity_role",
    "dependency_status",
    "upstream_task_id",
    "upstream_status",
    "upstream_failure_reason",
    "upstream_validation_path",
    "upstream_validation_sha256",
    "timestep_selection_status",
    "timestep_selection_reason",
    "timestep_selection_path",
    "timestep_selection_sha256",
    "runnable_task_id",
    "trajectory_expectation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def atomic_tsv(path: Path, fieldnames: tuple[str, ...], values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter="\t",
                lineterminator="\n",
            )
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


@dataclass(frozen=True)
class ValidationBundle:
    path: Path
    plan_path: Path
    summary_path: Path
    plan_rows: tuple[dict[str, str], ...]
    summary_by_task: dict[str, dict[str, str]]

    @property
    def sha256(self) -> str:
        return sha256(self.path)


def load_validation(path: Path) -> ValidationBundle:
    path = path.resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("planned_tasks") != value.get("validated_tasks"):
        raise ValueError(f"{path}: validation lacks one row per planned task")
    plan_path = Path(str(value["plan"])).resolve()
    summary_path = Path(str(value["summary_csv"])).resolve()
    if value.get("plan_sha256") != sha256(plan_path):
        raise ValueError(f"{path}: upstream plan hash mismatch")
    if value.get("summary_csv_sha256") != sha256(summary_path):
        raise ValueError(f"{path}: upstream summary hash mismatch")
    plan_rows = rows(plan_path, "\t")
    summary_rows = rows(summary_path)
    summary_by_task = {row["task_id"]: row for row in summary_rows}
    if len(summary_by_task) != len(summary_rows):
        raise ValueError(f"{path}: duplicate upstream summary task IDs")
    if {row["task_id"] for row in plan_rows} != set(summary_by_task):
        raise ValueError(f"{path}: upstream plan/summary task population mismatch")
    return ValidationBundle(
        path=path,
        plan_path=plan_path,
        summary_path=summary_path,
        plan_rows=tuple(plan_rows),
        summary_by_task=summary_by_task,
    )


def match_validation_row(
    bundle: ValidationBundle,
    *,
    calculation: str,
    model_id: str,
    phase: str,
    chemical_seed: int,
    temperature: int | None = None,
    velocity_seed: int | None = None,
) -> tuple[dict[str, str], dict[str, str]] | None:
    matches: list[dict[str, str]] = []
    for row in bundle.plan_rows:
        if (
            row["calculation"] != calculation
            or row["model_id"] != model_id
            or row["simulated_phase"] != phase
            or int(row["chemical_seed"]) != chemical_seed
        ):
            continue
        if temperature is not None and int(float(row["temperature_K"])) != temperature:
            continue
        if velocity_seed is not None and int(row["velocity_seed"]) != velocity_seed:
            continue
        matches.append(row)
    if len(matches) > 1:
        raise ValueError(
            "duplicate upstream dependency for "
            f"{calculation}/{model_id}/{phase}/C{chemical_seed}/T{temperature}/V{velocity_seed}"
        )
    if not matches:
        return None
    plan = matches[0]
    return plan, bundle.summary_by_task[plan["task_id"]]


def load_prior_ledger(path: Path | None) -> tuple[Path | None, dict[tuple[str, str, int, int, int], dict[str, str]]]:
    if path is None:
        return None, {}
    path = path.resolve()
    index: dict[tuple[str, str, int, int, int], dict[str, str]] = {}
    for row in rows(path, "\t"):
        key = (
            row["model_id"],
            row["simulated_phase"],
            int(row["chemical_seed"]),
            int(float(row["temperature_K"])),
            int(row["velocity_seed"]),
        )
        # NVE ledgers have two rows per key; they are not parents of another A5 stage.
        if key in index:
            raise ValueError(f"{path}: duplicate prior-ledger dependency key {key}")
        index[key] = row
    return path, index


def blocked_from_prior(
    prior: dict[tuple[str, str, int, int, int], dict[str, str]],
    key: tuple[str, str, int, int, int],
) -> tuple[str, str, str]:
    row = prior.get(key)
    if row is None:
        raise ValueError(f"missing both validated parent and prior-ledger row for {key}")
    if row["dependency_status"] != "blocked_upstream_failure":
        raise ValueError(f"non-blocked prior-ledger row lacks a validated parent for {key}")
    return (
        row["upstream_task_id"],
        row["upstream_status"],
        row["upstream_failure_reason"],
    )


def verified_parent(
    match: tuple[dict[str, str], dict[str, str]] | None,
    *,
    output_name: str,
    prior: dict[tuple[str, str, int, int, int], dict[str, str]],
    prior_key: tuple[str, str, int, int, int] | None,
) -> tuple[dict[str, str] | None, str, str, str]:
    if match is None:
        if prior_key is None:
            raise ValueError("missing required upstream validation row")
        task_id, status, reason = blocked_from_prior(prior, prior_key)
        return None, task_id, status, reason
    plan, summary = match
    status = summary.get("status", "missing_status")
    reason = summary.get("failure_reason", "").strip() or status
    if status != "passed":
        return None, plan["task_id"], status, reason
    parent = base.verify_parent_data(plan, output_name)
    return parent, plan["task_id"], status, ""


def load_partial_timestep_selection(
    selection_path: Path,
    receipt_path: Path,
    validation_path: Path,
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, str]]:
    selection_path = selection_path.resolve()
    receipt_path = receipt_path.resolve()
    validation_path = validation_path.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed":
        raise ValueError("timestep-selection receipt did not complete")
    if receipt.get("selection_sha256") != sha256(selection_path):
        raise ValueError("timestep-selection receipt hash mismatch")
    if validation.get("status") != "passed":
        raise ValueError("independent timestep-selection validation failed")
    if validation.get("selector_receipt_sha256") != sha256(receipt_path):
        raise ValueError("timestep-selection validation does not bind its receipt")
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows(selection_path):
        key = (row["model_id"], row["simulated_phase"])
        if key in selected:
            raise ValueError(f"duplicate timestep-selection row for {key}")
        if row["status"] == "selected":
            value = float(row["accepted_timestep_ps"])
            if value not in TIMESTEP_CANDIDATES_PS:
                raise ValueError(f"unsupported timestep {value} for {key}")
        selected[key] = row
    artifacts = {
        "timestep_selection_path": str(selection_path),
        "timestep_selection_sha256": sha256(selection_path),
        "timestep_selection_receipt_path": str(receipt_path),
        "timestep_selection_receipt_sha256": sha256(receipt_path),
        "timestep_selection_validation_path": str(validation_path),
        "timestep_selection_validation_sha256": sha256(validation_path),
    }
    return selected, artifacts


def validate_capacity_evidence(summary_path: Path, validation_path: Path) -> None:
    if sha256(summary_path) != BENCHMARK_SUMMARY_SHA256:
        raise ValueError("benchmark summary is not the A5-frozen capacity evidence")
    if sha256(validation_path) != BENCHMARK_VALIDATION_SHA256:
        raise ValueError("benchmark validation is not the A5-frozen independent replay")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed" or summary.get("benchmark_atoms") != 2000:
        raise ValueError("capacity benchmark did not pass at 2,000 atoms")
    if validation.get("status") != "passed" or validation.get("summary_sha256") != sha256(summary_path):
        raise ValueError("capacity benchmark independent validation did not pass")


def task_id(
    calculation: str,
    model_id: str,
    structure_id: str,
    temperature: int,
    velocity_seed: int,
    timestep: float,
) -> str:
    return "__".join(
        (
            calculation,
            model_id,
            structure_id,
            f"T{temperature}",
            f"V{velocity_seed}",
            f"dt{timestep}",
        )
    )


def intended_count(stage: str) -> int:
    return {
        "equilibrate_nvt": 12,
        "nve": 24,
        "equilibrate_npt": 72,
        "npt": 72,
    }[stage]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("equilibrate_nvt", "nve", "equilibrate_npt", "npt"),
    )
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--benchmark-validation", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--upstream-validation", type=Path, required=True)
    parser.add_argument("--upstream-ledger", type=Path)
    parser.add_argument("--timestep-selection", type=Path)
    parser.add_argument("--timestep-selection-receipt", type=Path)
    parser.add_argument("--timestep-selection-validation", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    builder = Path(__file__).resolve()
    suite_root = builder.parents[1]
    base_builder = (suite_root / "scripts" / "build_dynamic_stage_plan.py").resolve()
    model_manifest = args.model_manifest.resolve()
    structure_manifest = args.structure_manifest.resolve()
    protocol = args.protocol.resolve()
    benchmark_summary = args.benchmark_summary.resolve()
    benchmark_validation = args.benchmark_validation.resolve()
    campaign_root = args.campaign_root.resolve()
    output = args.output.resolve()
    ledger_path = args.ledger.resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    for path in (output, ledger_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if campaign_root.name != CAMPAIGN_ID:
        raise ValueError(f"campaign root must end in {CAMPAIGN_ID}")
    if sha256(protocol) not in SUPPORTED_PROTOCOL_SHA256:
        raise ValueError("protocol is neither immutable Amendment A5 nor Amendment A9")
    validate_capacity_evidence(benchmark_summary, benchmark_validation)

    models = rows(model_manifest, "\t")
    expected_models = {
        f"{architecture}__{phase}_parent"
        for architecture in ("DPA2", "DPA3", "DPA4")
        for phase in ("bcc", "fcc")
    }
    if len(models) != 6 or {row["model_id"] for row in models} != expected_models:
        raise ValueError("model manifest is not the complete six-model phase-native set")
    for model in models:
        model_path = Path(model["frozen_model"])
        if not model_path.is_file() or sha256(model_path) != model["frozen_sha256"]:
            raise ValueError(f"frozen model hash mismatch: {model['model_id']}")

    structures = rows(structure_manifest)
    primary = {
        (row["phase"], int(row["chemical_seed"])): row
        for row in structures
        if row["role"] == "primary" and int(row["natoms"]) == 2000
    }
    expected_structures = {
        (phase, seed) for phase in ("bcc", "fcc") for seed in CORE_CHEMICAL_SEEDS
    }
    if not expected_structures <= set(primary):
        raise ValueError("structure manifest lacks an A5 2,000-atom primary cell")
    for key in expected_structures:
        structure = primary[key]
        path = Path(structure["data_path"])
        if not path.is_file() or sha256(path) != structure["data_sha256"]:
            raise ValueError(f"structure hash mismatch: {structure['structure_id']}")

    upstream = load_validation(args.upstream_validation)
    prior_ledger_path, prior_ledger = load_prior_ledger(args.upstream_ledger)
    if args.stage in {"nve", "npt"} and prior_ledger_path is None:
        raise ValueError(f"{args.stage} requires --upstream-ledger to preserve blocked rows")

    selection: dict[tuple[str, str], dict[str, str]] = {}
    selection_artifacts = {field: "NA" for field in base.CHAIN_FIELDS[5:11]}
    if args.stage in {"equilibrate_npt", "npt"}:
        if not all(
            (
                args.timestep_selection,
                args.timestep_selection_receipt,
                args.timestep_selection_validation,
            )
        ):
            raise ValueError("NPT stages require all timestep-selection artifacts")
        selection, selection_artifacts = load_partial_timestep_selection(
            args.timestep_selection.resolve(),
            args.timestep_selection_receipt.resolve(),
            args.timestep_selection_validation.resolve(),
        )

    plan_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    for model in sorted(models, key=lambda row: row["model_id"]):
        phase = model["training_parent"]
        chemical_seeds = (
            (TIMESTEP_GATE_CHEMICAL_SEED,)
            if args.stage in {"equilibrate_nvt", "nve"}
            else CORE_CHEMICAL_SEEDS
        )
        velocity_seeds = (
            (TIMESTEP_GATE_VELOCITY_SEED,)
            if args.stage in {"equilibrate_nvt", "nve"}
            else CORE_VELOCITY_SEEDS
        )
        timestep_values: tuple[float, ...]
        if args.stage == "nve":
            timestep_values = TIMESTEP_CANDIDATES_PS
        elif args.stage == "equilibrate_nvt":
            timestep_values = (0.0005,)
        else:
            selected_row = selection.get((model["model_id"], phase))
            timestep_values = (
                (float(selected_row["accepted_timestep_ps"]),)
                if selected_row is not None and selected_row["status"] == "selected"
                else (math.nan,)
            )

        for chemical_seed in chemical_seeds:
            structure = primary[(phase, chemical_seed)]
            for temperature in CORE_TEMPERATURES:
                for velocity_seed in velocity_seeds:
                    for timestep in timestep_values:
                        selection_row = selection.get((model["model_id"], phase))
                        selection_status = "not_required"
                        selection_reason = ""
                        if args.stage in {"equilibrate_npt", "npt"}:
                            if selection_row is None:
                                selection_status = "missing_or_blocked"
                                selection_reason = "no timestep-selection row for model/phase branch"
                            else:
                                selection_status = selection_row["status"]
                                selection_reason = selection_row.get("selection_reason", "")

                        if args.stage == "equilibrate_nvt":
                            match = match_validation_row(
                                upstream,
                                calculation="relax_iso_coupled",
                                model_id=model["model_id"],
                                phase=phase,
                                chemical_seed=chemical_seed,
                            )
                            parent, upstream_task, upstream_status, failure_reason = verified_parent(
                                match,
                                output_name="relaxed.data",
                                prior={},
                                prior_key=None,
                            )
                            calculation = "equilibrate_nvt"
                            input_path = (suite_root / "inputs" / "11_nvt_equilibrate.in").resolve()
                            output_dir = campaign_root / "stage_a_equilibrate_nvt_a5" / model["model_id"] / structure["structure_id"] / f"T{temperature}"
                            n_eq, n_prod = base.integral_steps(EQUILIBRATION_PS, timestep), None
                            duration = EQUILIBRATION_PS
                            length_role = "timestep_gate_equilibration"
                        elif args.stage == "nve":
                            match = match_validation_row(
                                upstream,
                                calculation="equilibrate_nvt",
                                model_id=model["model_id"],
                                phase=phase,
                                chemical_seed=chemical_seed,
                                temperature=temperature,
                                velocity_seed=velocity_seed,
                            )
                            prior_key = (model["model_id"], phase, chemical_seed, temperature, velocity_seed)
                            parent, upstream_task, upstream_status, failure_reason = verified_parent(
                                match,
                                output_name="equilibrated.data",
                                prior=prior_ledger,
                                prior_key=prior_key,
                            )
                            calculation = "nve"
                            input_path = (suite_root / "inputs" / "10_nve_drift.in").resolve()
                            output_dir = campaign_root / "stage_a_nve_a5" / model["model_id"] / structure["structure_id"] / f"T{temperature}" / f"dt_{str(timestep).replace('.', 'p')}"
                            n_eq, n_prod = None, base.integral_steps(NVE_PRODUCTION_PS, timestep)
                            duration = NVE_PRODUCTION_PS
                            length_role = "matched_timestep_gate"
                        elif args.stage == "equilibrate_npt":
                            match = match_validation_row(
                                upstream,
                                calculation="relax_iso_coupled",
                                model_id=model["model_id"],
                                phase=phase,
                                chemical_seed=chemical_seed,
                            )
                            parent, upstream_task, upstream_status, failure_reason = verified_parent(
                                match,
                                output_name="relaxed.data",
                                prior={},
                                prior_key=None,
                            )
                            calculation = "equilibrate_npt"
                            input_path = (suite_root / "inputs" / "12_npt_equilibrate.in").resolve()
                            output_dir = campaign_root / "stage_c_equilibrate_npt_a5" / model["model_id"] / structure["structure_id"] / f"T{temperature}" / f"V{velocity_seed}"
                            n_eq = (
                                base.integral_steps(EQUILIBRATION_PS, timestep)
                                if math.isfinite(timestep)
                                else None
                            )
                            n_prod, duration = None, EQUILIBRATION_PS
                            length_role = "primary_equilibration"
                        else:
                            match = match_validation_row(
                                upstream,
                                calculation="equilibrate_npt",
                                model_id=model["model_id"],
                                phase=phase,
                                chemical_seed=chemical_seed,
                                temperature=temperature,
                                velocity_seed=velocity_seed,
                            )
                            prior_key = (model["model_id"], phase, chemical_seed, temperature, velocity_seed)
                            parent, upstream_task, upstream_status, failure_reason = verified_parent(
                                match,
                                output_name="equilibrated.data",
                                prior=prior_ledger,
                                prior_key=prior_key,
                            )
                            calculation = "npt"
                            input_path = (suite_root / "inputs" / "05_npt_thermo.in").resolve()
                            output_dir = campaign_root / "stage_c_npt_a5" / model["model_id"] / structure["structure_id"] / f"T{temperature}" / f"V{velocity_seed}"
                            n_eq = None
                            extended = (
                                chemical_seed in {CORE_CHEMICAL_SEEDS[0], CORE_CHEMICAL_SEEDS[-1]}
                                and velocity_seed == CORE_VELOCITY_SEEDS[0]
                                and temperature in CORE_TEMPERATURES
                            )
                            duration = NPT_LENGTH_SENSITIVITY_PS if extended else NPT_PRODUCTION_PS
                            n_prod = (
                                base.integral_steps(duration, timestep)
                                if math.isfinite(timestep)
                                else None
                            )
                            length_role = "extended_100ps" if extended else "primary_50ps"

                        identifier_timestep = timestep if math.isfinite(timestep) else 0.0
                        intended_id = task_id(
                            calculation,
                            model["model_id"],
                            structure["structure_id"],
                            temperature,
                            velocity_seed,
                            identifier_timestep,
                        )
                        runnable = parent is not None and selection_status in {"not_required", "selected"}
                        if parent is not None and not runnable and not failure_reason:
                            failure_reason = selection_reason or "timestep gate did not select a timestep"
                        dependency_status = (
                            "runnable_pending_execution" if runnable else "blocked_upstream_failure"
                        )
                        if runnable:
                            plan_rows.append(
                                base.task_row(
                                    calculation=calculation,
                                    model=model,
                                    structure=structure,
                                    data_path=parent["data_path"],
                                    parent=parent,
                                    parent_validation=upstream.path,
                                    input_path=input_path,
                                    output_dir=output_dir,
                                    protocol=protocol,
                                    model_manifest=model_manifest,
                                    velocity_seed=velocity_seed,
                                    temperature=temperature,
                                    timestep=timestep,
                                    n_eq=n_eq,
                                    n_prod=n_prod,
                                    duration_ps=duration,
                                    length_role=length_role,
                                    selection_artifacts=selection_artifacts,
                                )
                            )
                        ledger_rows.append(
                            {
                                "intended_task_id": intended_id,
                                "calculation": calculation,
                                "model_id": model["model_id"],
                                "architecture": model["architecture"],
                                "training_parent": model["training_parent"],
                                "simulated_phase": phase,
                                "chemical_seed": chemical_seed,
                                "velocity_seed": velocity_seed,
                                "temperature_K": temperature,
                                "timestep_ps": timestep if math.isfinite(timestep) else "NA",
                                "natoms": structure["natoms"],
                                "length_sensitivity_role": length_role,
                                "dependency_status": dependency_status,
                                "upstream_task_id": upstream_task,
                                "upstream_status": upstream_status,
                                "upstream_failure_reason": failure_reason,
                                "upstream_validation_path": str(upstream.path),
                                "upstream_validation_sha256": upstream.sha256,
                                "timestep_selection_status": selection_status,
                                "timestep_selection_reason": selection_reason,
                                "timestep_selection_path": selection_artifacts["timestep_selection_path"],
                                "timestep_selection_sha256": selection_artifacts["timestep_selection_sha256"],
                                "runnable_task_id": intended_id if runnable else "",
                                "trajectory_expectation": (
                                    "required_after_execution"
                                    if runnable
                                    else "absent_by_design_blocked"
                                ),
                            }
                        )

    expected = intended_count(args.stage)
    if len(ledger_rows) != expected or len({row["intended_task_id"] for row in ledger_rows}) != expected:
        raise ValueError(f"A5 {args.stage} intended ledger is not the fixed {expected}-row population")
    if len({row["task_id"] for row in plan_rows}) != len(plan_rows):
        raise ValueError("A5 executable plan contains duplicate task IDs")
    runnable_ids = {str(row["task_id"]) for row in plan_rows}
    ledger_runnable_ids = {
        str(row["runnable_task_id"]) for row in ledger_rows if row["runnable_task_id"]
    }
    if runnable_ids != ledger_runnable_ids:
        raise ValueError("A5 executable plan and intended dependency ledger differ")
    if args.stage == "npt":
        extended = sum(row["length_sensitivity_role"] == "extended_100ps" for row in ledger_rows)
        if extended != 24:
            raise ValueError(f"A5 intended NPT ledger must contain 24 extended rows, found {extended}")

    atomic_tsv(ledger_path, LEDGER_FIELDS, ledger_rows)
    base.atomic_rows(output, plan_rows)
    receipt: dict[str, object] = {
        "schema_version": 3,
        "status": "frozen",
        "campaign_id": CAMPAIGN_ID,
        "profile": "A5_two_temperature_dynamic_core",
        "stage": args.stage,
        "builder": str(builder),
        "builder_sha256": sha256(builder),
        "base_builder": str(base_builder),
        "base_builder_sha256": sha256(base_builder),
        "intended_tasks": len(ledger_rows),
        "runnable_tasks": len(plan_rows),
        "blocked_upstream_failure_tasks": len(ledger_rows) - len(plan_rows),
        "dependency_status_counts": dict(Counter(str(row["dependency_status"]) for row in ledger_rows)),
        "calculations": dict(Counter(str(row["calculation"]) for row in ledger_rows)),
        "model_ids": sorted({str(row["model_id"]) for row in ledger_rows}),
        "simulated_phases": sorted({str(row["simulated_phase"]) for row in ledger_rows}),
        "chemical_seeds": sorted({int(row["chemical_seed"]) for row in ledger_rows}),
        "velocity_seeds": sorted({int(row["velocity_seed"]) for row in ledger_rows}),
        "temperatures_K": sorted({int(row["temperature_K"]) for row in ledger_rows}),
        "model_manifest": str(model_manifest),
        "model_manifest_sha256": sha256(model_manifest),
        "structure_manifest": str(structure_manifest),
        "structure_manifest_sha256": sha256(structure_manifest),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "benchmark_summary": str(benchmark_summary),
        "benchmark_summary_sha256": sha256(benchmark_summary),
        "benchmark_validation": str(benchmark_validation),
        "benchmark_validation_sha256": sha256(benchmark_validation),
        "upstream_validation": str(upstream.path),
        "upstream_validation_sha256": upstream.sha256,
        "upstream_ledger": str(prior_ledger_path) if prior_ledger_path else "NA",
        "upstream_ledger_sha256": sha256(prior_ledger_path) if prior_ledger_path else "NA",
        "dependency_ledger": str(ledger_path),
        "dependency_ledger_sha256": sha256(ledger_path),
        "plan": str(output),
        "plan_sha256": sha256(output),
        "tasks": len(plan_rows),
    }
    if args.stage in {"equilibrate_npt", "npt"}:
        receipt.update(selection_artifacts)
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
