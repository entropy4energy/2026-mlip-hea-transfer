#!/usr/bin/env python3
"""Extract small-strain modulus, peak strength, and 0.2% offset yield point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def interpolate_crossing(x1, y1, x2, y2):
    fraction = -y1 / (y2 - y1)
    return x1 + fraction * (x2 - x1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="stress_strain.dat or directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elastic-limit", type=float, default=0.01)
    parser.add_argument("--offset", type=float, default=0.002)
    args = parser.parse_args()

    files = [args.input] if args.input.is_file() else sorted(args.input.rglob("stress_strain.dat"))
    if len(files) != 1:
        raise ValueError(f"expected exactly one stress_strain.dat, found {len(files)}")
    data = np.loadtxt(files[0], comments="#", ndmin=2)
    if data.shape[1] != 12:
        raise ValueError(f"expected 12 columns in {files[0]}, found {data.shape[1]}")
    strain = data[:, 2]
    stress = data[:, 3]
    if strain.size < 6 or not np.all(np.isfinite(strain)) or not np.all(np.isfinite(stress)):
        raise ValueError("stress-strain data need at least six finite samples")
    loading_sign = 1.0 if np.nanmedian(strain[-max(5, strain.size // 10) :]) >= 0.0 else -1.0
    loading_strain = loading_sign * strain
    loading_stress = loading_sign * stress
    if np.any(np.diff(loading_strain) < -1.0e-12):
        raise ValueError("loading strain must be monotonic in the loading direction")
    elastic_mask = (loading_strain > 0.0) & (loading_strain <= args.elastic_limit)
    if np.count_nonzero(elastic_mask) < 5:
        raise ValueError("fewer than five samples in the requested elastic fit window")
    modulus, intercept = np.polyfit(strain[elastic_mask], stress[elastic_mask], 1)
    fit = modulus * strain[elastic_mask] + intercept
    residual = float(np.sum((stress[elastic_mask] - fit) ** 2))
    total = float(np.sum((stress[elastic_mask] - stress[elastic_mask].mean()) ** 2))

    peak_index = int(np.argmax(loading_stress))
    result: dict[str, object] = {
        "source": str(files[0]),
        "loading": "tension" if loading_sign > 0 else "compression",
        "small_strain_modulus_GPa": float(modulus),
        "elastic_fit_intercept_GPa": float(intercept),
        "elastic_fit_limit": args.elastic_limit,
        "elastic_fit_r_squared": 1.0 - residual / total if total > 0.0 else 1.0,
        "peak_axial_stress_GPa": float(stress[peak_index]),
        "strain_at_peak_stress": float(strain[peak_index]),
    }

    offset_difference = loading_stress - modulus * (loading_strain - args.offset)
    candidates = np.where(
        (loading_strain[1:] > args.offset)
        & (offset_difference[:-1] >= 0.0)
        & (offset_difference[1:] < 0.0)
    )[0]
    if candidates.size:
        index = int(candidates[0])
        yield_loading_strain = interpolate_crossing(
            loading_strain[index],
            offset_difference[index],
            loading_strain[index + 1],
            offset_difference[index + 1],
        )
        yield_loading_stress = float(
            np.interp(yield_loading_strain, loading_strain, loading_stress)
        )
        result["offset_yield_strain"] = float(loading_sign * yield_loading_strain)
        result["offset_yield_stress_GPa"] = float(loading_sign * yield_loading_stress)
    else:
        result["offset_yield_strain"] = None
        result["offset_yield_stress_GPa"] = None
    result["offset_definition"] = args.offset

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "uniaxial_properties.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
