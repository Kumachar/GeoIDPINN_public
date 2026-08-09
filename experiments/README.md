# Real-data experiment protocols

These scripts preserve the training protocols used for the paper. They contain
code only and require the original Louisiana tract-level weekly case/testing
file to be supplied locally through `LOUISIANA_DATA_ROOT`.

The public metadata intentionally omits observed totals. When an experiment
selects a high-case subset, it computes `total_cases` and `total_tests` at run
time from the local study file. Those values and all row-level outputs are
ignored and must not be committed.

Primary all-64 scripts:

- `real_data_geoid_all64_theta2_combined_masked_origin.py`: main combined
  observed-outside plus mixed-loss PINN.
- `real_data_geoid_all64_theta2_no_geo_updated_training.py`: identity-fixed
  no-county-network ablation.
- `real_data_geoid_all64_theta2_neighbor_vs_data_inferred.py`: Neighbor Network
  versus Data-Inferred Network.
- `real_data_geoid_all64_theta2_updated_prior_experiment.py`: spatial-prior
  sensitivity.
- `real_data_fair_forecast_spatial_nb_experiment.py`: NB and NB-AR baselines.

Use smoke mode first. Full paper runs are computationally expensive.
