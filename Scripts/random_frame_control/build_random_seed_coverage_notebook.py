#!/usr/bin/env python3
"""Create the reader-facing notebook for the random-frame coverage control."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import nbformat as nbf


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results"
NOTEBOOKS = PACKAGE / "notebooks"
OUTPUT = NOTEBOOKS / "random_seed_coverage_control.ipynb"


def rows_by_key() -> dict[tuple[str, str], dict[str, str]]:
    with (RESULTS / "control_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["fold"], row["control"]): row for row in rows}


def value(row: dict[str, str], field: str) -> float:
    return float(row[field])


def main() -> int:
    summary = rows_by_key()
    fcc_pooled = summary[("fcc_to_bcc", "pooled_random_frames")]
    bcc_pooled = summary[("bcc_to_fcc", "pooled_random_frames")]
    fcc_stratified = summary[("fcc_to_bcc", "one_per_trajectory")]
    bcc_stratified = summary[("bcc_to_fcc", "one_per_trajectory")]

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (SciFy1)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.14"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Matched-label random-frame coverage control\n\n"
            "## tl;dr\n\n"
            f"The frozen systematic first-frame treatment reduced the median paired system-level "
            f"coverage distance by **{value(fcc_pooled, 'systematic_seed_reduction_percent'):.2f}%** "
            f"in FCC-to-BCC and **{value(bcc_pooled, 'systematic_seed_reduction_percent'):.2f}%** "
            f"in BCC-to-FCC. Across 10,000 matched pooled random-frame draws, the corresponding "
            f"medians were **{value(fcc_pooled, 'random_median_reduction_percent'):.2f}%** "
            f"[{value(fcc_pooled, 'random_q025_reduction_percent'):.2f}, "
            f"{value(fcc_pooled, 'random_q975_reduction_percent'):.2f}] and "
            f"**{value(bcc_pooled, 'random_median_reduction_percent'):.2f}%** "
            f"[{value(bcc_pooled, 'random_q025_reduction_percent'):.2f}, "
            f"{value(bcc_pooled, 'random_q975_reduction_percent'):.2f}]. Every pooled random draw "
            "exceeded the systematic value in both folds. Randomizing the time point while retaining "
            "exactly one frame per trajectory also exceeded the systematic first-frame result. The "
            "coverage gain therefore cannot be attributed to a special advantage of the first-frame "
            "selection; these controls remain within POCC-derived trajectories and do not compare POCC "
            "against random-alloy or SQS structure generation."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "The PI requested a matched-label reference for the existing structural-coverage claim. "
            "The analysis preserves the frozen 150-dimensional, label-blind, central-element-stratified "
            "radial descriptor and the original every-100th-frame temporal-thinning variants. For each "
            "draw, 29 FCC-parent or 30 BCC-parent frames replace the systematic first frames, and coverage "
            "is recomputed on the complete opposite-parent held-out systems. The fold statistic is the "
            "median of paired system-level relative reductions in median nearest-training distance.\n\n"
            "### Key Assumptions\n\n"
            "- The temporally thinned candidate pool is the governing frame population because it is the "
            "population fixed for the original coverage analysis; thinning reduces serial dependence but "
            "does not establish independence.\n"
            "- The primary pooled control samples distinct frames uniformly, so long trajectories receive "
            "more candidate weight and a draw can omit systems.\n"
            "- The one-per-trajectory sensitivity holds chemical-system breadth fixed and randomizes only "
            "the retained time point.\n"
            "- No energy, force, virial, model output, new DFT label, or model training enters the control.\n"
            "- Empirical draw intervals describe selection variability in these frozen trajectories; they "
            "are not population confidence intervals.\n\n"
            "The full locked protocol is in `../analysis_protocol.md`."
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from IPython.display import Image, display\n\n"
            "cwd = Path.cwd().resolve()\n"
            "PACKAGE = cwd if (cwd / 'results').is_dir() else cwd.parent\n"
            "RESULTS = PACKAGE / 'results'\n"
            "summary = pd.read_csv(RESULTS / 'control_summary.csv')\n"
            "draws = pd.read_csv(RESULTS / 'random_draw_distribution.csv.gz')\n"
            "candidates = pd.read_csv(RESULTS / 'candidate_frames.csv')\n"
            "trajectories = pd.read_csv(RESULTS / 'trajectory_manifest.csv')\n"
            "frames = pd.read_csv(RESULTS / 'frame_manifest.csv')\n"
            "seed_check = pd.read_csv(RESULTS / 'systematic_seed_reconstruction.csv')\n"
            "print(f'{len(draws):,} random-draw rows; {len(candidates)} candidate frames; '"
            "+ f'{len(trajectories)} trajectories; {len(frames)} CIF frames')\n"
            "display(seed_check)"
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            "derived = (draws.groupby(['fold', 'control'])['coverage_reduction_percent']\n"
            "           .agg(random_median='median', random_q025=lambda x: x.quantile(0.025),\n"
            "                random_q975=lambda x: x.quantile(0.975)).reset_index())\n"
            "checked = summary.merge(derived, on=['fold', 'control'], validate='one_to_one')\n"
            "assert np.allclose(checked.random_median, checked.random_median_reduction_percent)\n"
            "assert np.allclose(checked.random_q025, checked.random_q025_reduction_percent)\n"
            "assert np.allclose(checked.random_q975, checked.random_q975_reduction_percent)\n"
            "columns = ['fold_label', 'control', 'systematic_seed_reduction_percent',\n"
            "           'random_median_reduction_percent', 'random_q025_reduction_percent',\n"
            "           'random_q975_reduction_percent',\n"
            "           'systematic_minus_random_median_percentage_points',\n"
            "           'systematic_empirical_percentile',\n"
            "           'one_sided_p_random_ge_systematic']\n"
            "display(summary[columns].round(4))"
        ),
        nbf.v4.new_code_cell(
            "for fold in ['fcc_to_bcc', 'bcc_to_fcc']:\n"
            "    observed = summary.loc[(summary.fold == fold) & "
            "(summary.control == 'pooled_random_frames'), 'systematic_seed_reduction_fraction'].iloc[0]\n"
            "    values = draws.loc[(draws.fold == fold) & "
            "(draws.control == 'pooled_random_frames'), 'coverage_reduction_fraction'].to_numpy()\n"
            "    print(f'{fold}: {np.count_nonzero(values > observed):,}/{len(values):,} pooled draws '"
            "+ 'exceed the systematic first-frame value')\n"
            "display(Image(filename=str(PACKAGE / 'figures' / 'random_seed_coverage_control.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            f"- **Matched random frames cover more descriptor space than the prescribed first frames.** "
            f"Pooled random medians exceed the systematic values by "
            f"{abs(value(fcc_pooled, 'systematic_minus_random_median_percentage_points')):.2f} and "
            f"{abs(value(bcc_pooled, 'systematic_minus_random_median_percentage_points')):.2f} percentage "
            "points for FCC-to-BCC and BCC-to-FCC, respectively.\n"
            f"- **The result is not caused by pooled draws omitting trajectories.** Exactly one random frame "
            f"per trajectory gives {value(fcc_stratified, 'random_median_reduction_percent'):.2f}% "
            f"[{value(fcc_stratified, 'random_q025_reduction_percent'):.2f}, "
            f"{value(fcc_stratified, 'random_q975_reduction_percent'):.2f}] and "
            f"{value(bcc_stratified, 'random_median_reduction_percent'):.2f}% "
            f"[{value(bcc_stratified, 'random_q025_reduction_percent'):.2f}, "
            f"{value(bcc_stratified, 'random_q975_reduction_percent'):.2f}].\n"
            "- **Claim calibration:** approximately 30 POCC-derived configurations provide substantial "
            "coverage at low label count, but the evidence does not support attributing that gain to the "
            "first-frame seed rule. Later thermal frames are a stronger matched-label coverage reference.\n"
            "- **Boundary:** because every control frame comes from an existing POCC-derived trajectory, "
            "this analysis does not test whether POCC enumeration is superior to random-alloy, SQS, active-"
            "learning, or other structure-selection strategies, and it says nothing about model performance "
            "without training matched control models."
        ),
    ]

    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp")
    nbf.write(notebook, temporary)
    temporary.replace(OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
