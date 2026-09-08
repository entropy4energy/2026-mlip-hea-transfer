#!/usr/bin/env python3
"""Execute one immutable H100 stage-plan row and write a hash-bound receipt."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from common import ROOT, atomic_json, load_json, read_tsv, require_hash, sha256


INPUTS = {
    "relax": "00_relax_oriented.in",
    "equilibrate": "01_equilibrate_npt.in",
    "nve": "02_nve_drift.in",
    "tension": "03_uniaxial_tension_a5.in",
}
EXPECTED = {
    "relax": ("relaxed.data", "relaxed.restart", "relax_summary.csv", "relax_convergence.csv"),
    "equilibrate": (
        "equilibrated.data",
        "equilibrated.restart",
        "equilibration_samples.dat",
        "equilibration_structure_samples.dat",
        "trajectory_equilibration.lammpstrj",
    ),
    "nve": ("nve_final.data", "nve_final.restart", "nve_energy_samples.dat", "trajectory_nve.lammpstrj"),
    "tension": ("tension_final.data", "tension_final.restart", "stress_strain.dat", "trajectory_tension.lammpstrj"),
}


def arguments(row: dict[str, str], model: Path, plugin: str) -> list[str]:
    common = [
        "-var", "MODEL", str(model),
        "-var", "PLUGIN", plugin,
        "-var", "DATA", row["data_path"],
        "-var", "MODEL_ID", "DPA2__bcc_parent",
        "-var", "ORIENTATION", row["orientation"],
    ]
    stage = row["stage"]
    if stage == "equilibrate":
        common += [
            "-var", "TEMP", row["temperature_K"],
            "-var", "SEED", row["velocity_seed"],
            "-var", "N_EQ", row["n_steps"],
        ]
    elif stage == "nve":
        common += ["-var", "TEMP", row["temperature_K"], "-var", "N_STEPS", row["n_steps"]]
    elif stage == "tension":
        common += [
            "-var", "TEMP", row["temperature_K"],
            "-var", "RATE", row["strain_rate_ps_inverse"],
            "-var", "N_STEPS", row["n_steps"],
            "-var", "TARGET_STRAIN", "0.20",
        ]
    return common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--row-number", type=int, required=True, help="one-based data-row number")
    args = parser.parse_args()
    plan = args.plan.resolve()
    rows = read_tsv(plan)
    if not 1 <= args.row_number <= len(rows):
        raise IndexError(f"row {args.row_number} outside 1..{len(rows)}")
    row = rows[args.row_number - 1]
    if int(row["task_id"]) != args.row_number:
        raise ValueError("plan task_id does not match physical row number")
    plan_receipt_path = plan.with_suffix(plan.suffix + ".receipt.json")
    plan_receipt = load_json(plan_receipt_path)
    if plan_receipt.get("status") != "passed" or plan_receipt.get("plan_sha256") != sha256(plan):
        raise ValueError("plan receipt/hash gate failed")
    campaign = load_json(ROOT / "config" / "campaign.json")
    model = ROOT / "assets" / "DPA2__bcc_parent.pth"
    require_hash(model, str(campaign["model_sha256"]), "frozen model")
    data = Path(row["data_path"]).resolve()
    require_hash(data, row["data_sha256"], "plan-bound input data")
    lmp = os.environ.get("H100_LAMMPS_EXE", "lmp")
    plugin = os.environ.get("H100_DEEPMD_PLUGIN", "")
    if not plugin or not Path(plugin).is_file():
        raise ValueError("H100_DEEPMD_PLUGIN is missing or invalid")
    stage = row["stage"]
    input_path = ROOT / "inputs" / INPUTS[stage]
    output = Path(row["output_dir"]).resolve()
    completion = output / "run.complete.json"
    if completion.exists():
        prior = load_json(completion)
        if prior.get("status") == "passed" and prior.get("plan_sha256") == sha256(plan):
            print(f"already complete: {output}")
            return
        raise ValueError(f"conflicting completion receipt: {completion}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"partial output must be quarantined before unchanged retry: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = [lmp, "-in", str(input_path), *arguments(row, model, plugin)]
    manifest = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "stage": stage,
        "task_id": int(row["task_id"]),
        "run_id": row["run_id"],
        "plan": str(plan),
        "plan_sha256": sha256(plan),
        "plan_receipt": str(plan_receipt_path),
        "plan_receipt_sha256": sha256(plan_receipt_path),
        "row": row,
        "model": str(model),
        "model_sha256": sha256(model),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "data": str(data),
        "data_sha256": sha256(data),
        "plugin": str(Path(plugin).resolve()),
        "plugin_sha256": sha256(Path(plugin).resolve()),
        "lammps_executable": lmp,
        "command": command,
        "hostname": socket.getfqdn(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    manifest_path = output / "run_manifest.json"
    atomic_json(manifest_path, manifest)
    stdout_path = output / "lammps.stdout"
    stderr_path = output / "lammps.stderr"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(command, cwd=output, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.time() - started
    marker = f"H100_CAMPAIGN_STAGE_COMPLETE {stage}"
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    missing = [name for name in EXPECTED[stage] if not (output / name).is_file()]
    oversized = [
        str(path)
        for path in output.glob("*.lammpstrj")
        if path.stat().st_size > int(campaign["trajectory_max_bytes"])
    ]
    if result.returncode or marker not in text or missing or oversized:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "return_code": result.returncode,
            "completion_marker_present": marker in text,
            "missing_outputs": missing,
            "oversized_trajectories": oversized,
            "elapsed_seconds": elapsed,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        }
        atomic_json(output / "run.failed.json", failure)
        print(json.dumps(failure, indent=2))
        raise SystemExit(result.returncode or 2)
    outputs = [
        {"path": str((output / name).resolve()), "bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
        for name in EXPECTED[stage]
    ]
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "stage": stage,
        "task_id": int(row["task_id"]),
        "run_id": row["run_id"],
        "return_code": result.returncode,
        "completion_marker": marker,
        "elapsed_seconds": elapsed,
        "plan": str(plan),
        "plan_sha256": sha256(plan),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "stdout": str(stdout_path),
        "stdout_sha256": sha256(stdout_path),
        "stderr": str(stderr_path),
        "stderr_sha256": sha256(stderr_path),
        "outputs": outputs,
    }
    atomic_json(completion, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
