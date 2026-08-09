"""Regenerate aggregate model and prior comparison figures from released tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"
OUTPUT_DIR = ROOT / "figures" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "Neighbor Network": "#177E89",
    "No County Network": "#73777F",
    "NB-AR": "#D97706",
}


def plot_model_metrics() -> None:
    table = pd.read_csv(TABLE_DIR / "table_6_all64_three_model_comparison.csv")
    order = ["Neighbor Network", "No County Network", "NB-AR"]
    table = table.set_index("model").loc[order].reset_index()
    specs = [
        ("test_nll", "test_nll_sd", "Test NLL", ".3f"),
        ("mae", "mae_sd", "MAE", ".1f"),
        ("mse", "mse_sd", "MSE", ",.0f"),
    ]
    labels = ["Neighbor\nNetwork", "No County\nNetwork", "NB-AR"]

    for metric, error_column, ylabel, value_format in specs:
        values = table[metric].to_numpy(float)
        errors = table[error_column].to_numpy(float)
        x = np.arange(len(table))
        fig, axis = plt.subplots(figsize=(6.3, 4.7))
        bars = axis.bar(
            x,
            values,
            yerr=errors,
            capsize=5,
            width=0.64,
            color=[MODEL_COLORS[name] for name in order],
            edgecolor="white",
        )
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        upper = float(np.max(values + errors))
        axis.set_ylim(0, upper * 1.2)
        for bar, value, error in zip(bars, values, errors):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + upper * 0.035,
                format(value, value_format),
                ha="center",
                va="bottom",
                fontsize=10,
            )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"all64_{metric}_comparison.png", dpi=300)
        fig.savefig(OUTPUT_DIR / f"all64_{metric}_comparison.pdf")
        plt.close(fig)


def plot_prior_mse() -> None:
    table = pd.read_csv(TABLE_DIR / "table_8_all64_prior_mse_comparison.csv")
    table = table.sort_values("test_mse_mean").reset_index(drop=True)
    colors = table["prior_label"].map(
        {
            "Neighbor Network": "#4C6A88",
            "Data-Inferred Network": "#2A9D8F",
        }
    ).fillna("#C8CDD2")
    y = np.arange(len(table))
    fig, axis = plt.subplots(figsize=(9.4, 7.0))
    axis.barh(
        y,
        table["test_mse_mean"],
        xerr=table["test_mse_sd_across_origins"],
        color=colors,
        edgecolor="#263238",
        capsize=3,
    )
    axis.set_yticks(y, table["prior_label"])
    axis.invert_yaxis()
    axis.set_xlabel("Test MSE")
    axis.grid(axis="x", alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all64_prior_mse_comparison.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "all64_prior_mse_comparison.pdf")
    plt.close(fig)


def main() -> None:
    plot_model_metrics()
    plot_prior_mse()
    print(f"Saved aggregate figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
