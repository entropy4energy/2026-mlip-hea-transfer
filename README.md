# 2026 MLIP HEA Transfer

This repository provides data and inputs for the paper:
**"From Training Data to Molecular-Dynamics Deployment: Transfer of
Machine-Learning Interatomic Potentials in a Refractory High-Entropy Alloy"**.

The repository is the companion to the article and holds everything needed to
audit the reported figures and tables and to rerun the simulations: the
calculation inputs, the starting structures, the analysis and figure scripts,
and the numerical data behind every figure and table.

The first-principles records, the canonical training labels, the frozen
deployment models, and the raw trajectories are too large to distribute here
and are available from the corresponding author.

- **Input:** 15 architecture-native DeePMD training configurations, the
  held-out evaluation-system lists, the LAMMPS input decks for
  finite-temperature dynamics and uniaxial tension, and the 2,000-atom
  random-alloy cells.
- **Static evaluation:** equal-system force, energy, virial, and stress errors
  on unseen parent-lattice families, with paired system-bootstrap intervals,
  frame-thinning and checkpoint sensitivities, and structural-coverage
  descriptors.
- **Deployment:** zero-temperature relaxation, NVE drift, NPT thermal
  response with replica-level records, and finite-rate tensile endpoints.
- **Output:** CSV tables used directly by the manuscript figures and the
  supplementary material.

## Repository Structure

```text
Data/
  model/                           Static-evaluation estimates, paired contrasts,
                                   coverage descriptors, force-bin and checkpoint
                                   sensitivities, dataset counts, model recipes
  random_frame_control/            10,000-draw random-frame coverage control
  lammps/                          Relaxation, NPT groups, thermal response,
                                   DPA4/BCC 1200 K run history and follow-up
  tensile/                         Replica endpoints, group summaries, paired
                                   contrasts, binned stress-strain curves
  plot_ready/                      Per-figure tables (F01-F08, FS1)

Inputs/
  deepmd/training/<run_id>/        input.json for each of the 15 training runs
  deepmd/training_runs.csv         Run identifiers, parents, data regime, steps, seed
  deepmd/evaluation_sets/          Held-out system lists per fold (primary and
                                   every-100th-frame independence variants)
  lammps/finite_temperature/       Build, relax, NVE drift, NPT equilibrate, NPT
                                   production decks
  lammps/tensile/                  Relax, NPT equilibrate, NVE drift, uniaxial
                                   tension decks
  structures/finite_temperature/   Five 2,000-atom BCC and five FCC decorations
  structures/tensile/              Nine oriented BCC cells ([100], [110], [111])

Scripts/
  README.md                        Analysis and figure scripts (see note)
```

## Requirements

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) for dependency management

Scripts declare their Python dependencies in their `uv` headers. No separate
package installation is required when using `uv run`.

## Usage

Run commands from the repository root.

### Training

Each `Inputs/deepmd/training/<run_id>/input.json` is a complete DeePMD-kit
input. Replace `${DATA_ROOT}` with the directory holding the training labels
before training; those labels are not included here because of their size.
`Inputs/deepmd/training_runs.csv` maps each run identifier to
its architecture, training parent, held-out parent, data regime, optimizer
steps, and random seed.

Software versions used in the article: DeePMD-kit `3.2.0b1.dev162+gc8a2d85fe`
(commit `c8a2d85fe`), PyTorch `2.12.1+cu129`, CUDA 12.9, and LAMMPS 22 July
2025 Update 2 with the DeePMD plugin. LAMMPS used `metal` units and atom types
`1=Hf, 2=Mo, 3=Ta, 4=Ti, 5=Zr`.

DPA2 and DPA3 deployment models are portable frozen `.pth` files. The DPA4
`.pt2` files were frozen for NVIDIA L40S (`sm_89`); refreeze the supplied EMA
checkpoint on a different GPU architecture rather than moving the `.pt2` file
between unlike architectures.

VASP PAW potential files are not redistributed. The calculations used
`Hf_pv`, `Mo_pv`, `Ta_pv`, `Ti_sv`, and `Zr_sv`, in that order.

### Deployment

The LAMMPS decks in `Inputs/lammps/` run in the numbered order within each
directory against the structures in `Inputs/structures/` and a frozen model
from the archive.

### Analysis and figures

The scripts that reduce raw trajectories to the tables in `Data/` and draw the
figures from `Data/plot_ready/` are listed in `Scripts/README.md`.

## Output Dataset

`Data/` contains one CSV per reported quantity. Representative files:

| File | Content |
|---|---|
| `model/primary_equal_system_estimates.csv` | Equal-system force, energy, virial, and stress errors per architecture, fold, and regime |
| `model/primary_paired_contrasts.csv` | Paired regime contrasts with 95% system-bootstrap intervals |
| `model/independence_*.csv` | The same estimates and contrasts on the every-100th-frame corpus |
| `model/environment_coverage.csv.gz` | Nearest-training-environment distance for every sampled unseen environment |
| `model/coverage_error_associations.csv` | Spearman correlations between coverage change and force-error change |
| `model/dataset_regime_counts.csv` | System and frame counts for every role and regime |
| `random_frame_control/control_summary.csv` | Systematic first-frame versus 10,000 random-draw coverage reductions, per fold |
| `lammps/zero_temperature_relaxation.csv` | Relaxed volumes and acceptance per branch and decoration |
| `lammps/finite_temperature_response.csv` | Apparent volumetric and isotropic linear responses with intervals |
| `lammps/dpa4_bcc_1200K_run_history.csv` | Original and follow-up 1200 K runs with eligibility decisions |
| `tensile/replica_endpoints.csv` | Tangent slope, offset yield, maximum axial stress, strain at maximum, and work density per realization |
| `tensile/paired_contrasts.csv` | Temperature and orientation contrasts with Student-t intervals |

This repository is intended as a minimal working reference and starting point
for cross-lattice transfer and molecular-dynamics deployment tests of
machine-learning interatomic potentials in multicomponent alloys.

## License and citation

Copyright © 2026 Entropy for Energy Lab, Johns Hopkins University.

Released under the MIT License; see `LICENSE`. The VASP PAW potential files
are not redistributed here and remain subject to their own license.

If you use this repository, please cite the article and this repository; see
`CITATION.cff`.
