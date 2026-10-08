# GeoID-PINN Public Reproducibility Release

This repository accompanies **GeoID-PINN: Identifiability-Aware Regional
Epidemic Inference with Geographic Coupling**. It contains the model code,
public geographic inputs, aggregate experiment results, paper figures, and
output-free notebooks used to inspect the reported results.

The selected trajectory panel is included only as a paper figure, without its
underlying county-week table. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) and
[PUBLIC_RELEASE_AUDIT.md](PUBLIC_RELEASE_AUDIT.md) for the exact boundary.

Louisiana counties are county-equivalent units. Public-facing figures and
documentation use **county** for consistency; compatibility code may retain
`parish_fips` internally.

## Main reported results

All values below are test results averaged across four rolling forecast
origins; PINN entries also average three random seeds within each origin.

| Model | Test NLL | MAE | MSE |
|---|---:|---:|---:|
| Neighbor Network | 5.335 | 57.20 | 11,612 |
| No County Network (identity fixed) | 5.363 | 59.74 | 12,749 |
| NB-AR | **5.158** | 70.60 | 32,957 |

The Data-Inferred Network has MSE 11,468, MAE 57.73, and NLL 5.346. Thus the
PINN variants improve point accuracy while NB-AR retains the best likelihood.

## Repository map

- `data/public_priors/`: Census centers, adjacency, distance, and LODES-derived
  matrices for all 64 Louisiana counties.
- `data/external/`: public CDC weekly cases for neighboring states.
- `results/tables/`: aggregate paper and supplement tables.
- `results/matrices/`: 12-run mean learned matrices and aggregate distances.
- `figures/paper/`: the three figures directly included by the camera-ready
  LaTeX source.
- `figures/supplementary/`: model comparisons, prior sensitivity, and learned
  matrix diagnostics.
- `notebooks/`: clean English notebooks that use only included files.
- `experiments/`: real-data training protocols; these require a separately
  obtained local study file.
- `src/geoid_pinn/`: reusable data, path, prior, and significance helpers.
- `scripts/`: figure regeneration and public-release auditing.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
python scripts/plot_release_results.py
python scripts/plot_release_matrices.py
python scripts/audit_public_release.py
jupyter lab
```

The four summary notebooks run entirely from released files. The synthetic
identifiability notebook is self-contained but computationally heavier.

## Optional real-data training

The real-data scripts are retained for method transparency, but the original
Louisiana tract-level weekly case/testing file is not redistributed. After
obtaining it from the original data steward, point the code to a local copy:

```powershell
$env:LOUISIANA_DATA_ROOT = "C:\path\to\local\study-data"
$env:GEOID_ALL64_MODE = "smoke"
python experiments/real_data_geoid_all64_theta2_combined_masked_origin.py
```