#!/usr/bin/env python3
"""Fit Einstein diffusion coefficients to total and species MSD curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

LABELS = ("all", "Hf", "Mo", "Ta", "Ti", "Zr")


def fit_line(time: np.ndarray, values: np.ndarray) -> dict[str, float | bool]:
    slope, intercept = np.polyfit(time, values, 1)
    fitted = slope * time + intercept
    residual = float(np.sum((values - fitted) ** 2))
    total = float(np.sum((values - values.mean()) ** 2))
    diffusion_a2_ps = float(slope / 6.0)
    return {
        "slope_A2_per_ps": float(slope),
        "intercept_A2": float(intercept),
        "r_squared": 1.0 - residual / total if total > 0.0 else 1.0,
        "diffusion_A2_per_ps": diffusion_a2_ps,
        "diffusion_m2_per_s": diffusion_a2_ps * 1.0e-8,
        "positive_slope": bool(slope > 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="msd.dat or directory containing it")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.1 <= args.fit_fraction <= 1.0:
        raise ValueError("--fit-fraction must be between 0.1 and 1")

    files = [args.input] if args.input.is_file() else sorted(args.input.rglob("msd.dat"))
    if len(files) != 1:
        raise ValueError(f"expected exactly one msd.dat, found {len(files)}")
    data = np.loadtxt(files[0], comments="#", ndmin=2)
    if data.shape[1] != 11:
        raise ValueError(f"expected 11 columns in {files[0]}, found {data.shape[1]}")
    start = int(np.floor(data.shape[0] * (1.0 - args.fit_fraction)))
    if data.shape[0] - start < 20:
        raise ValueError("MSD fit window contains fewer than 20 samples")
    time = data[start:, 1]
    series = {
        "all": data[start:, 5],
        "Hf": data[start:, 6],
        "Mo": data[start:, 7],
        "Ta": data[start:, 8],
        "Ti": data[start:, 9],
        "Zr": data[start:, 10],
    }
    fits = {label: fit_line(time, series[label]) for label in LABELS}
    result = {
        "source": str(files[0]),
        "fit_fraction": args.fit_fraction,
        "fit_time_range_ps": [float(time.min()), float(time.max())],
        "fits": fits,
        "interpretation_warning": "A negative slope or poor linear fit means this trajectory does not resolve diffusion; do not clip it to zero.",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diffusion.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.output / "diffusion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["species", "slope_A2_per_ps", "diffusion_A2_per_ps", "diffusion_m2_per_s", "r_squared"])
        for label in LABELS:
            row = fits[label]
            writer.writerow(
                [label, row["slope_A2_per_ps"], row["diffusion_A2_per_ps"], row["diffusion_m2_per_s"], row["r_squared"]]
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

