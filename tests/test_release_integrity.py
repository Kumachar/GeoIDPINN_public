from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_public_county_metadata_has_no_study_fields() -> None:
    path = ROOT / "data" / "public_priors" / "county_metadata_with_centroids.csv"
    metadata = pd.read_csv(path, dtype={"county_fips": str})
    assert len(metadata) == 64
    assert metadata["county_fips"].str.zfill(5).str.startswith("22").all()
    assert set(metadata.columns) == {
        "county_fips",
        "county_name",
        "state_name",
        "cenpop2020_population",
        "latitude",
        "longitude",
    }


def test_released_mean_matrices_are_row_stochastic_and_consistent() -> None:
    matrix_dir = ROOT / "results" / "matrices"
    neighbor = pd.read_csv(
        matrix_dir / "neighbor_network_learned_matrix_mean.csv", index_col=0
    )
    inferred = pd.read_csv(
        matrix_dir / "data_inferred_network_learned_matrix_mean.csv", index_col=0
    )
    difference = pd.read_csv(
        matrix_dir / "data_inferred_minus_neighbor_learned_matrix.csv", index_col=0
    )
    assert neighbor.shape == inferred.shape == difference.shape == (64, 64)
    assert np.allclose(neighbor.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(inferred.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(difference, inferred - neighbor, atol=1e-8)


def test_release_notebooks_have_no_saved_execution_state() -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None, path
            assert not cell.get("outputs"), path
