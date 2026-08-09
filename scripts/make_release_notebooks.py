"""Create output-free, release-facing notebooks from aggregate repository files."""

from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(text: str):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
    }


SETUP = """
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_repo_root(start=Path.cwd()):
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the repository.")

ROOT = find_repo_root()
TABLE_DIR = ROOT / "results" / "tables"
MATRIX_DIR = ROOT / "results" / "matrices"
"""


def write_notebook(filename: str, cells: list) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = NOTEBOOK_DIR / filename
    destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def sanitize_synthetic_notebook() -> None:
    path = NOTEBOOK_DIR / "05_synthetic_identifiability_experiment.ipynb"
    if not path.exists():
        return
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        legacy_container_path = "/" + "mnt" + "/data/full_pinn_synthetic_outputs"
        source = source.replace(legacy_container_path, "artifacts/synthetic")
        source = source.replace("full_pinn_synthetic_outputs", "artifacts/synthetic")
        cell["source"] = source.splitlines(keepends=True)
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook.get("metadata", {}).pop("widgets", None)
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def model_comparison() -> None:
    write_notebook(
        "01_all64_model_comparison.ipynb",
        [
            markdown("""
            # Louisiana 64-County Model Comparison

            This notebook compares the Neighbor Network, identity-fixed No
            County Network, and NB-AR using released test summaries only.
            """),
            code(SETUP),
            code("""
            results = pd.read_csv(TABLE_DIR / "table_6_all64_three_model_comparison.csv")
            results
            """),
            code("""
            order = ["Neighbor Network", "No County Network", "NB-AR"]
            results = results.set_index("model").loc[order].reset_index()
            specs = [("test_nll", "test_nll_sd", "Test NLL"),
                     ("mae", "mae_sd", "MAE"),
                     ("mse", "mse_sd", "MSE")]
            colors = ["#177E89", "#73777F", "#D97706"]
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for axis, (metric, error, label) in zip(axes, specs):
                axis.bar(results["model"], results[metric], yerr=results[error],
                         color=colors, capsize=4)
                axis.set_ylabel(label)
                axis.tick_params(axis="x", rotation=20)
                axis.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            """),
            code("""
            indexed = results.set_index("model")
            neighbor = indexed.loc["Neighbor Network"]
            no_network = indexed.loc["No County Network"]
            nb_ar = indexed.loc["NB-AR"]
            pd.Series({
                "MSE reduction vs No County Network (%)": 100 * (1 - neighbor.mse / no_network.mse),
                "MAE reduction vs No County Network (%)": 100 * (1 - neighbor.mae / no_network.mae),
                "MSE reduction vs NB-AR (%)": 100 * (1 - neighbor.mse / nb_ar.mse),
                "MAE reduction vs NB-AR (%)": 100 * (1 - neighbor.mae / nb_ar.mae),
            }).round(2)
            """),
        ],
    )


def prior_sensitivity() -> None:
    write_notebook(
        "02_spatial_prior_sensitivity.ipynb",
        [
            markdown("""
            # Spatial-Prior Sensitivity

            Compare all 64-county PINN priors using test MSE. Neighbor and
            Data-Inferred Networks are emphasized, while the remaining priors
            act as sensitivity and permutation controls.
            """),
            code(SETUP),
            code("""
            priors = pd.read_csv(TABLE_DIR / "table_8_all64_prior_mse_comparison.csv")
            priors.sort_values("test_mse_mean")
            """),
            code("""
            ordered = priors.sort_values("test_mse_mean").reset_index(drop=True)
            colors = ordered["prior_label"].map({
                "Neighbor Network": "#4C6A88",
                "Data-Inferred Network": "#2A9D8F",
            }).fillna("#C8CDD2")
            fig, axis = plt.subplots(figsize=(9, 6.5))
            axis.barh(ordered["prior_label"], ordered["test_mse_mean"],
                      xerr=ordered["test_mse_sd_across_origins"],
                      color=colors, capsize=3)
            axis.invert_yaxis()
            axis.set_xlabel("Test MSE")
            axis.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            """),
            code("""
            ordered[["prior_label", "test_mse_mean", "test_mse_sd_across_origins", "mse_rank"]]
            """),
        ],
    )


def matrix_comparison() -> None:
    write_notebook(
        "03_learned_matrix_comparison.ipynb",
        [
            markdown("""
            # Learned Coupling-Matrix Comparison

            These matrices are averages over 12 fits. Rows are recipient
            counties, columns are source counties, and every row sums to one.
            """),
            code(SETUP),
            code("""
            neighbor = pd.read_csv(MATRIX_DIR / "neighbor_network_learned_matrix_mean.csv", index_col=0)
            inferred = pd.read_csv(MATRIX_DIR / "data_inferred_network_learned_matrix_mean.csv", index_col=0)
            difference = pd.read_csv(MATRIX_DIR / "data_inferred_minus_neighbor_learned_matrix.csv", index_col=0)
            diagnostics = pd.read_csv(MATRIX_DIR / "all64_learned_matrix_distance_summary.csv")
            diagnostics.T
            """),
            code("""
            checks = pd.DataFrame({
                "neighbor_row_sum": neighbor.sum(axis=1),
                "inferred_row_sum": inferred.sum(axis=1),
            })
            checks.describe()
            """),
            code("""
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            panels = [(neighbor, "Neighbor Network", "viridis", 0, 1),
                      (inferred, "Data-Inferred Network", "viridis", 0, 1)]
            limit = np.abs(difference.to_numpy()).max()
            panels.append((difference, "Data-Inferred minus Neighbor", "RdBu_r", -limit, limit))
            for axis, (matrix, title, cmap, vmin, vmax) in zip(axes, panels):
                image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
                axis.set_title(title)
                axis.set_xlabel("Source county index")
                axis.set_ylabel("Recipient county index")
                fig.colorbar(image, ax=axis, fraction=0.046)
            fig.tight_layout()
            """),
        ],
    )


def paper_index() -> None:
    write_notebook(
        "04_paper_results_index.ipynb",
        [
            markdown("""
            # Released Paper Results Index

            This notebook loads every released aggregate table. It is a quick
            consistency check and a compact index of the paper-facing results.
            """),
            code(SETUP),
            code("""
            table_files = sorted(TABLE_DIR.glob("*.csv"))
            [(path.name, pd.read_csv(path).shape) for path in table_files]
            """),
            code("""
            tables = {path.stem: pd.read_csv(path) for path in table_files}
            for name, table in tables.items():
                print(f"\\n{name} ({len(table)} rows)")
                display(table)
            """),
        ],
    )


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    model_comparison()
    prior_sensitivity()
    matrix_comparison()
    paper_index()
    sanitize_synthetic_notebook()
    print(f"Wrote release notebooks to {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
