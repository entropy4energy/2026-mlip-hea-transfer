#!/usr/bin/env python3
"""Build immutable, dependency-gated plans for the H100 campaign stages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import (
    ROOT,
    atomic_json,
    atomic_tsv,
    load_json,
    read_tsv,
    require_complete_receipt,
    sha256,
)


STAGES = ("relax", "equilibrate", "nve", "tension")


def rate_tag(rate: float) -> str:
    return f"{rate:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def run_dir(stage: str, run_id: str) -> Path:
    return (ROOT / "runs" / stage / run_id).resolve()


def receipt_for(stage: str, run_id: str) -> Path:
    return run_dir(stage, run_id) / "run.complete.json"


def gate_for(stage: str, run_id: str, name: str) -> Path:
    return run_dir(stage, run_id) / name


def require_gate(path: Path) -> dict[str, object]:
    gate = load_json(path)
    if gate.get("status") != "passed":
        raise ValueError(f"scientific gate is not passed: {path}")
    return gate


def base_row(
    *,
    stage: str,
    run_id: str,
    profile: str,
    orientation: str,
    natoms: int,
    chemical_seed: int,
    velocity_seed: int | str,
    temperature: int | str,
    rate: float | str,
    n_steps: int | str,
    data: Path,
    parent_receipt: Path | str,
) -> dict[str, object]:
    campaign = load_json(ROOT / "config" / "campaign.json")
    dump_interval = float(campaign["trajectory_interval_ps"])
    dt = float(campaign["timestep_ps"])
    if stage == "tension":
        frames = math.ceil(int(n_steps) * dt / dump_interval) + 1
        bytes_per_atom_frame = 300
    elif stage == "equilibrate":
        frames = math.ceil(float(campaign["equilibration_ps"]) / 1.0) + 1
        bytes_per_atom_frame = 100
    elif stage == "nve":
        frames = math.ceil(float(campaign["nve_qualification_ps"]) / 0.5) + 1
        bytes_per_atom_frame = 210
    else:
        frames = 0
        bytes_per_atom_frame = 0
    estimate = natoms * frames * bytes_per_atom_frame
    if estimate > int(campaign["trajectory_max_bytes"]):
        raise ValueError(f"{run_id}: pre-run trajectory estimate exceeds 10 GB")
    return {
        "task_id": "",
        "stage": stage,
        "run_id": run_id,
        "profile": profile,
        "orientation": orientation,
        "natoms": natoms,
        "chemical_seed": chemical_seed,
        "velocity_seed": velocity_seed,
        "temperature_K": temperature,
        "strain_rate_ps_inverse": rate,
        "n_steps": n_steps,
        "data_path": str(data.resolve()),
        "data_sha256": sha256(data),
        "parent_receipt": str(parent_receipt) if parent_receipt else "",
        "output_dir": str(run_dir(stage, run_id)),
        "estimated_trajectory_bytes": estimate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage
    selection_path = args.selection.resolve()
    output = args.output.resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {output}")
    selection = load_json(selection_path)
    if selection.get("status") != "selected":
        raise ValueError("capacity/profile selection is not passed")
    profile = str(selection["profile"])
    campaign = load_json(ROOT / "config" / "campaign.json")
    if profile == "compact":
        structure_manifest = ROOT / "structures_a3_compact" / "structure_manifest.tsv"
    elif profile in {"medium", "large", "xlarge"}:
        structure_manifest = ROOT / "structures" / "structure_manifest.tsv"
    else:
        raise ValueError(f"unsupported frozen profile: {profile}")
    structures = [row for row in read_tsv(structure_manifest) if row["profile"] == profile]
    structure_by_key = {
        (row["orientation"], int(row["chemical_seed"])): row for row in structures
    }
    pairs = [
        (int(item["chemical_seed"]), int(item["velocity_seed"]))
        for item in campaign["chemical_velocity_pairs"]
    ]
    temperatures = [int(value) for value in campaign["temperatures_K"]]
    rows: list[dict[str, object]] = []
    if stage == "relax":
        for orientation in campaign["orientations"]:
            for chemical_seed, _ in pairs:
                structure = structure_by_key[(orientation, chemical_seed)]
                run_id = f"{profile}_{orientation}_chem{chemical_seed}"
                data = ROOT / Path(structure["data_path"])
                rows.append(
                    base_row(
                        stage=stage,
                        run_id=run_id,
                        profile=profile,
                        orientation=orientation,
                        natoms=int(structure["natoms"]),
                        chemical_seed=chemical_seed,
                        velocity_seed="",
                        temperature="",
                        rate="",
                        n_steps="",
                        data=data,
                        parent_receipt="",
                    )
                )
    elif stage == "equilibrate":
        for orientation in campaign["orientations"]:
            for chemical_seed, velocity_seed in pairs:
                structure = structure_by_key[(orientation, chemical_seed)]
                relax_id = f"{profile}_{orientation}_chem{chemical_seed}"
                parent_data = run_dir("relax", relax_id) / "relaxed.data"
                parent_receipt = receipt_for("relax", relax_id)
                require_complete_receipt(parent_receipt, parent_data)
                for temperature in temperatures:
                    run_id = f"{profile}_{orientation}_chem{chemical_seed}_T{temperature}"
                    rows.append(
                        base_row(
                            stage=stage,
                            run_id=run_id,
                            profile=profile,
                            orientation=orientation,
                            natoms=int(structure["natoms"]),
                            chemical_seed=chemical_seed,
                            velocity_seed=velocity_seed,
                            temperature=temperature,
                            rate="",
                            n_steps=round(float(campaign["equilibration_ps"]) / float(campaign["timestep_ps"])),
                            data=parent_data,
                            parent_receipt=parent_receipt,
                        )
                    )
    elif stage == "nve":
        chemical_seed, velocity_seed = pairs[0]
        for orientation in campaign["orientations"]:
            structure = structure_by_key[(orientation, chemical_seed)]
            for temperature in temperatures:
                eq_id = f"{profile}_{orientation}_chem{chemical_seed}_T{temperature}"
                parent_data = run_dir("equilibrate", eq_id) / "equilibrated.data"
                parent_receipt = receipt_for("equilibrate", eq_id)
                require_complete_receipt(parent_receipt, parent_data)
                require_gate(gate_for("equilibrate", eq_id, "stationarity_gate.json"))
                run_id = f"{profile}_{orientation}_chem{chemical_seed}_T{temperature}"
                rows.append(
                    base_row(
                        stage=stage,
                        run_id=run_id,
                        profile=profile,
                        orientation=orientation,
                        natoms=int(structure["natoms"]),
                        chemical_seed=chemical_seed,
                        velocity_seed=velocity_seed,
                        temperature=temperature,
                        rate="",
                        n_steps=round(float(campaign["nve_qualification_ps"]) / float(campaign["timestep_ps"])),
                        data=parent_data,
                        parent_receipt=parent_receipt,
                    )
                )
    else:
        dt = float(campaign["timestep_ps"])
        target = float(campaign["maximum_engineering_strain"])
        base_rate = float(campaign["core_rate_ps_inverse"])
        rates_for_pair = {pairs[0][0]: [base_rate], pairs[1][0]: [base_rate], pairs[2][0]: [base_rate]}
        if selection["tier"] == "core_plus_rate":
            rates_for_pair[pairs[0][0]].extend(float(value) for value in campaign["optional_rates_ps_inverse"])
        for orientation in campaign["orientations"]:
            for temperature in temperatures:
                nve_id = f"{profile}_{orientation}_chem{pairs[0][0]}_T{temperature}"
                require_complete_receipt(receipt_for("nve", nve_id), run_dir("nve", nve_id) / "nve_final.data")
                require_gate(gate_for("nve", nve_id, "nve_gate.json"))
                for chemical_seed, velocity_seed in pairs:
                    structure = structure_by_key[(orientation, chemical_seed)]
                    eq_id = f"{profile}_{orientation}_chem{chemical_seed}_T{temperature}"
                    parent_data = run_dir("equilibrate", eq_id) / "equilibrated.data"
                    parent_receipt = receipt_for("equilibrate", eq_id)
                    require_complete_receipt(parent_receipt, parent_data)
                    require_gate(gate_for("equilibrate", eq_id, "stationarity_gate.json"))
                    for rate in rates_for_pair[chemical_seed]:
                        n_steps = math.ceil(target / (rate * dt))
                        run_id = f"{profile}_{orientation}_chem{chemical_seed}_T{temperature}_rate{rate_tag(rate)}"
                        rows.append(
                            base_row(
                                stage=stage,
                                run_id=run_id,
                                profile=profile,
                                orientation=orientation,
                                natoms=int(structure["natoms"]),
                                chemical_seed=chemical_seed,
                                velocity_seed=velocity_seed,
                                temperature=temperature,
                                rate=rate,
                                n_steps=n_steps,
                                data=parent_data,
                                parent_receipt=parent_receipt,
                            )
                        )
    for task_id, row in enumerate(rows, 1):
        row["task_id"] = task_id
    expected = {"relax": 9, "equilibrate": 18, "nve": 6, "tension": 30 if selection["tier"] == "core_plus_rate" else 18}[stage]
    if len(rows) != expected:
        raise ValueError(f"{stage}: planned {len(rows)} rows, expected {expected}")
    atomic_tsv(output, rows)
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "campaign_id": campaign["campaign_id"],
        "stage": stage,
        "rows": len(rows),
        "plan": str(output),
        "plan_sha256": sha256(output),
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
