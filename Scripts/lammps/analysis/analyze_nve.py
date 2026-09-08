#!/usr/bin/env python3
"""Quantify NVE total-energy drift without treating timesteps as replicas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discard-ps", type=float, default=5.0)
    parser.add_argument("--block-ps", type=float, default=5.0)
    args = parser.parse_args()
    source = args.input / "nve_energy_samples.dat" if args.input.is_dir() else args.input
    data = np.loadtxt(source, comments="#", ndmin=2)
    if data.shape[1] != 8:
        raise ValueError(f"expected 8 columns in {source}, found {data.shape[1]}")
    step, time_ps, atoms, temperature, pressure_bar, potential, kinetic, total = data.T
    if not np.all(np.isfinite(data)):
        raise ValueError("NVE time series contains non-finite values")
    atom_counts = np.rint(atoms).astype(int)
    if np.unique(atom_counts).size != 1:
        raise ValueError("atom count changed during NVE run")
    if np.any(np.diff(time_ps) <= 0.0):
        raise ValueError("NVE sample times are not strictly increasing")
    mask = time_ps >= args.discard_ps
    if np.count_nonzero(mask) < 20:
        raise ValueError("NVE drift fit contains fewer than 20 samples")
    fit_time = time_ps[mask]
    fit_energy = total[mask]
    slope, intercept = np.polyfit(fit_time, fit_energy, 1)
    fitted = slope * fit_time + intercept
    residual = fit_energy - fitted

    block_ids = np.floor((fit_time - fit_time.min()) / args.block_ps).astype(int)
    block_rows: list[dict[str, float | int]] = []
    for block_id in np.unique(block_ids):
        block_mask = block_ids == block_id
        if np.count_nonzero(block_mask) < 2:
            continue
        block_time = fit_time[block_mask]
        block_energy = fit_energy[block_mask]
        block_slope = np.polyfit(block_time, block_energy, 1)[0]
        block_rows.append(
            {
                "block": int(block_id),
                "start_ps": float(block_time.min()),
                "end_ps": float(block_time.max()),
                "samples": int(block_energy.size),
                "mean_total_eV_per_atom": float(block_energy.mean()),
                "peak_to_peak_meV_per_atom": float(np.ptp(block_energy) * 1000.0),
                "drift_meV_per_atom_ps": float(block_slope * 1000.0),
            }
        )
    if len(block_rows) < 2:
        raise ValueError("NVE series does not contain at least two complete analysis blocks")

    result = {
        "source": str(source.resolve()),
        "natoms": int(atom_counts[0]),
        "samples": int(data.shape[0]),
        "duration_ps": float(time_ps.max() - time_ps.min()),
        "discard_ps": args.discard_ps,
        "fit_time_range_ps": [float(fit_time.min()), float(fit_time.max())],
        "mean_temperature_K": float(temperature[mask].mean()),
        "temperature_standard_deviation_K": float(temperature[mask].std(ddof=1)),
        "mean_pressure_GPa": float(pressure_bar[mask].mean() * 1.0e-4),
        "initial_fit_energy_eV_per_atom": float(fit_energy[0]),
        "final_fit_energy_eV_per_atom": float(fit_energy[-1]),
        "net_change_meV_per_atom": float((fit_energy[-1] - fit_energy[0]) * 1000.0),
        "drift_meV_per_atom_ps": float(slope * 1000.0),
        "detrended_rmse_meV_per_atom": float(np.sqrt(np.mean(residual**2)) * 1000.0),
        "peak_to_peak_meV_per_atom": float(np.ptp(fit_energy) * 1000.0),
        "block_length_ps": args.block_ps,
        "blocks": len(block_rows),
        "maximum_block_peak_to_peak_meV_per_atom": max(
            float(row["peak_to_peak_meV_per_atom"]) for row in block_rows
        ),
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "nve_drift.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "nve_blocks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(block_rows[0]))
        writer.writeheader()
        writer.writerows(block_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
