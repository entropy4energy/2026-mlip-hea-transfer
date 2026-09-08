#!/usr/bin/env python3
"""Reduce NPT time series to thermodynamic averages and temperature trends."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

KB_EV_PER_K = 8.617333262145e-5
EV_A3_TO_GPA = 160.21766208


def block_standard_error(values: np.ndarray, maximum_blocks: int = 10) -> float:
    if values.size < 20:
        return float("nan")
    nblocks = min(maximum_blocks, values.size // 10)
    block_size = values.size // nblocks
    trimmed = values[: nblocks * block_size].reshape(nblocks, block_size)
    means = trimmed.mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(nblocks))


def block_heat_capacities(enthalpy: np.ndarray, natoms: int, temperature: float) -> np.ndarray:
    nblocks = min(10, enthalpy.size // 20)
    if nblocks < 2:
        return np.asarray([], dtype=float)
    block_size = enthalpy.size // nblocks
    blocks = enthalpy[: nblocks * block_size].reshape(nblocks, block_size)
    return np.var(blocks, axis=1, ddof=1) / (natoms * (KB_EV_PER_K * temperature) ** 2)


def read_one(path: Path) -> dict[str, float | int | str]:
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != 10:
        raise ValueError(f"expected 10 columns in {path}, found {data.shape[1]}")
    _, time_ps, atoms, temperature, pressure_bar, volume, density, enthalpy, enthalpy2, pe = data.T
    natoms_values = np.rint(atoms).astype(int)
    if np.unique(natoms_values).size != 1:
        raise ValueError(f"atom count changed in {path}")
    natoms = int(natoms_values[0])
    mean_temperature = float(temperature.mean())
    variance_h_direct = float(np.var(enthalpy, ddof=1))
    variance_h_moments = float(enthalpy2.mean() - enthalpy.mean() ** 2)
    cp_kb_per_atom = variance_h_direct / (natoms * (KB_EV_PER_K * mean_temperature) ** 2)
    block_cp = block_heat_capacities(enthalpy, natoms, mean_temperature)
    kappa_a3_per_ev = float(np.var(volume, ddof=1) / (KB_EV_PER_K * mean_temperature * volume.mean()))
    return {
        "source": str(path),
        "n_samples": int(data.shape[0]),
        "duration_ps": float(time_ps.max() - time_ps.min()),
        "natoms": natoms,
        "temperature_K": mean_temperature,
        "temperature_block_se_K": block_standard_error(temperature),
        "pressure_GPa": float(pressure_bar.mean() * 1.0e-4),
        "pressure_block_se_GPa": block_standard_error(pressure_bar) * 1.0e-4,
        "volume_A3": float(volume.mean()),
        "volume_A3_per_atom": float(volume.mean() / natoms),
        "volume_block_se_A3": block_standard_error(volume),
        "density_g_cm3": float(density.mean()),
        "density_block_se_g_cm3": block_standard_error(density),
        "enthalpy_eV_per_atom": float(enthalpy.mean() / natoms),
        "potential_energy_eV_per_atom": float(pe.mean() / natoms),
        "Cp_kB_per_atom": float(cp_kb_per_atom),
        "Cp_block_se_kB_per_atom": float(block_cp.std(ddof=1) / np.sqrt(block_cp.size)) if block_cp.size > 1 else float("nan"),
        "enthalpy_variance_eV2": variance_h_direct,
        "enthalpy_variance_moment_check_eV2": variance_h_moments,
        "isothermal_compressibility_GPa_inverse": kappa_a3_per_ev / EV_A3_TO_GPA,
    }


def clean_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input.rglob("thermo_samples.dat"))
    if not files:
        raise ValueError(f"no thermo_samples.dat files found below {args.input}")
    summaries = sorted((read_one(path) for path in files), key=lambda row: float(row["temperature_K"]))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    fieldnames = list(summaries[0])
    with (output / "thermo_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    result: dict[str, object] = {"runs": summaries}
    if len(summaries) >= 3:
        temperatures = np.asarray([float(row["temperature_K"]) for row in summaries])
        volumes_per_atom = np.asarray([float(row["volume_A3_per_atom"]) for row in summaries])
        alpha_v, intercept = np.polyfit(temperatures, np.log(volumes_per_atom), 1)
        predicted = alpha_v * temperatures + intercept
        residual = float(np.sum((np.log(volumes_per_atom) - predicted) ** 2))
        total = float(np.sum((np.log(volumes_per_atom) - np.log(volumes_per_atom).mean()) ** 2))
        result["thermal_expansion_fit"] = {
            "temperature_range_K": [float(temperatures.min()), float(temperatures.max())],
            "volumetric_alpha_per_K": float(alpha_v),
            "isotropic_linear_alpha_per_K": float(alpha_v / 3.0),
            "ln_volume_fit_r_squared": 1.0 - residual / total if total > 0.0 else 1.0,
            "note": "single linear fit over the supplied temperature range",
        }
    result = clean_json(result)
    (output / "thermo_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

