"""Regenerate heatmaps from the released 12-run mean coupling matrices."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = ROOT / "results" / "matrices"
OUTPUT_DIR = ROOT / "figures" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_matrix(filename: str) -> pd.DataFrame:
    matrix = pd.read_csv(MATRIX_DIR / filename, index_col=0)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{filename} is not square: {matrix.shape}")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"Rows of {filename} do not sum to one")
    return matrix


def heatmap(matrix: pd.DataFrame, title: str, stem: str, difference: bool = False) -> None:
    values = matrix.to_numpy(float)
    if difference:
        limit = float(np.max(np.abs(values)))
        cmap, vmin, vmax = "RdBu_r", -limit, limit
    else:
        cmap, vmin, vmax = "viridis", 0.0, 1.0

    fig, axis = plt.subplots(figsize=(11.0, 9.6))
    image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    positions = np.arange(len(matrix))
    axis.set_xticks(positions, matrix.columns, rotation=90, fontsize=5)
    axis.set_yticks(positions, matrix.index, fontsize=5)
    axis.set_xlabel("Source county")
    axis.set_ylabel("Recipient county")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    neighbor = read_matrix("neighbor_network_learned_matrix_mean.csv")
    inferred = read_matrix("data_inferred_network_learned_matrix_mean.csv")
    difference = pd.read_csv(
        MATRIX_DIR / "data_inferred_minus_neighbor_learned_matrix.csv",
        index_col=0,
    )
    if not np.allclose(difference.to_numpy(float), inferred - neighbor, atol=1e-8):
        raise ValueError("Released difference matrix does not match inferred - neighbor")
    heatmap(neighbor, "Neighbor Network: mean learned matrix", "neighbor_mean_matrix")
    heatmap(inferred, "Data-Inferred Network: mean learned matrix", "inferred_mean_matrix")
    heatmap(
        difference,
        "Data-Inferred minus Neighbor mean matrix",
        "inferred_minus_neighbor_matrix",
        difference=True,
    )
    print(f"Saved matrix heatmaps to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
