# Public-release audit

The release is designed around data minimization. Only files needed to inspect
the paper or reproduce aggregate visualizations are committed.

## Automated checks

Run:

```powershell
python scripts/audit_public_release.py
```

The audit fails on:

- restricted-data and model-artifact extensions;
- risky raw/prediction/checkpoint filenames;
- absolute local filesystem paths;
- common credential and private-key patterns;
- notebook outputs or execution counts;
- restricted case/testing fields in released CSV schemas.

Use `--write-manifest` after a successful audit to regenerate
`RELEASE_MANIFEST.csv` with SHA-256 hashes.

## Reviewed exceptions

- `data/external/cdc_neighbor_states_weekly_cases_2020.csv` contains public,
  state-level CDC case counts.
- `results/matrices/*_mean.csv` are 12-run aggregate learned parameters.
- `figures/paper/selected6_main_no_geo_nb_ar_trajectories.pdf` is a fixed paper
  artifact; no machine-readable county-week source table is released.
- experiment source code names expected columns in the separately obtained
  local study file, but it does not contain the records themselves.

## Before publishing

1. Run the audit and tests from a fresh clone.
2. Inspect `git status` and `git diff --cached`.
3. Confirm no ignored local files were force-added.
4. Confirm the intended software and data licenses with all authors.
5. Add the final repository URL and paper DOI when available.
