from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geoid_pinn.paths import data_root, project_root, public_prior_dir, results_root

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:
    raise RuntimeError("This experiment requires PyTorch.") from exc


warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
np.set_printoptions(precision=4, suppress=True)

PROJECT_ROOT = Path(os.environ["FAIR_GEO_PROJECT_ROOT"]).resolve() if os.environ.get("FAIR_GEO_PROJECT_ROOT") else project_root()
DATA_ROOT = data_root()
RESULTS_ROOT = results_root()
SCOPE = os.environ.get("FAIR_GEO_SCOPE", "all64").lower()
RUN_MODE = os.environ.get("FAIR_GEO_MODE", "paper").lower()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

if SCOPE not in {"top15", "all64"}:
    raise ValueError("FAIR_GEO_SCOPE must be 'top15' or 'all64'.")
if RUN_MODE not in {"smoke", "paper"}:
    raise ValueError("FAIR_GEO_MODE must be 'smoke' or 'paper'.")

CONFIG = {
    "smoke": {
        "splits": ["roll_2020-09-03"],
        "epochs": 8,
        "test_weeks": 3,
        "models": ["nb_ar", "adjacency", "lodes_sym", "lodes_permuted"],
    },
    "paper": {
        "splits": [
            "roll_2020-07-23",
            "roll_2020-09-03",
            "roll_2020-10-15",
            "roll_2020-11-26",
        ],
        "epochs": 1500,
        "test_weeks": 6,
        "models": ["nb_ar", "adjacency", "distance", "lodes_sym", "lodes_permuted"],
    },
}[RUN_MODE]

START_DATE = pd.Timestamp("2020-03-05")
END_DATE = pd.Timestamp("2020-12-31")
OUTPUT_DIR = RESULTS_ROOT / "real_data_result" / f"fair_forecast_spatial_nb_{SCOPE}_outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
for directory in [OUTPUT_DIR, TABLE_DIR, FIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"scope={SCOPE} mode={RUN_MODE} device={DEVICE}")
print(json.dumps(CONFIG, indent=2))


def find_louisiana_dta(root: Path) -> Path:
    candidates = [
        root / "Raw_data" / "LousianaWeeklyCasesByCensusTract2022_date_20231001.dta",
        root / "LousianaWeeklyCasesByCensusTract2022_date_20231001.dta",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = list(root.rglob("LousianaWeeklyCasesByCensusTract2022_date_20231001.dta"))
    if found:
        return found[0]
    raise FileNotFoundError("Could not locate the Louisiana weekly case .dta file.")


def load_weekly_data(dta_path: Path) -> pd.DataFrame:
    raw = pd.read_stata(dta_path, convert_categoricals=False)
    required = ["tract_fips10", "dateforstartofweek", "weeklycasecount", "weeklytestcount"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Raw data missing required columns: {missing}")
    raw["parish_fips"] = raw["tract_fips10"].astype(str).str[:5].str.zfill(5)
    raw["week"] = pd.to_datetime(raw["dateforstartofweek"])
    raw["cases"] = pd.to_numeric(raw["weeklycasecount"], errors="coerce").fillna(0).clip(lower=0)
    raw["tests"] = pd.to_numeric(raw["weeklytestcount"], errors="coerce").fillna(0).clip(lower=0)
    return raw.groupby(["parish_fips", "week"], as_index=False).agg(cases=("cases", "sum"), tests=("tests", "sum"))


def build_matrices(
    weekly: pd.DataFrame,
    fips_order: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset = weekly[(weekly["week"] >= start) & (weekly["week"] <= end)].copy()
    weeks = pd.Index(sorted(subset["week"].unique()), name="week")
    cases = (
        subset.pivot_table(index="week", columns="parish_fips", values="cases", aggfunc="sum")
        .reindex(index=weeks, columns=fips_order)
        .fillna(0.0)
    )
    tests = (
        subset.pivot_table(index="week", columns="parish_fips", values="tests", aggfunc="sum")
        .reindex(index=weeks, columns=fips_order)
        .fillna(0.0)
    )
    return cases, tests


def read_square_matrix(path: Path, fips_order: List[str]) -> np.ndarray:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str).str.zfill(5)
    frame.columns = frame.columns.astype(str).str.zfill(5)
    return frame.loc[fips_order, fips_order].to_numpy(dtype=np.float64)


def off_diagonal_row_normalize(matrix: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(matrix, dtype=np.float64).copy(), 0.0)
    np.fill_diagonal(result, 0.0)
    row_sums = result.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 0
    result[valid] = result[valid] / row_sums[valid]
    result[~valid] = 0.0
    return result


def load_analysis_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, Path]:
    prior_dir = public_prior_dir("all64")
    metadata_file = prior_dir / "parish_metadata_with_centroids.csv"
    top_n = 64 if SCOPE == "all64" else 15
    metadata = pd.read_csv(metadata_file)
    metadata["parish_fips"] = metadata["parish_fips"].astype(str).str.zfill(5)
    weekly = load_weekly_data(find_louisiana_dta(DATA_ROOT))
    study_totals = (
        weekly.groupby("parish_fips", as_index=False)
        .agg(total_cases=("cases", "sum"), total_tests=("tests", "sum"))
    )
    metadata = metadata.merge(study_totals, on="parish_fips", how="left")
    metadata = metadata.sort_values("total_cases", ascending=False).head(top_n).reset_index(drop=True)
    fips_order = metadata["parish_fips"].tolist()
    cases, tests = build_matrices(weekly, fips_order, START_DATE, END_DATE)
    population = metadata.set_index("parish_fips").loc[fips_order, "cenpop2020_population"].astype(float)
    return cases, tests, population, metadata, prior_dir


cases_df, tests_df, population_s, parish_metadata, PRIOR_DIR = load_analysis_data()
FIPS = cases_df.columns.tolist()
PARISH_NAMES = parish_metadata.set_index("parish_fips").loc[FIPS, "parish"].tolist()
CASES = cases_df.to_numpy(dtype=np.float64)
TESTS = tests_df.to_numpy(dtype=np.float64)
POPULATION = population_s.to_numpy(dtype=np.float64)
T, P = CASES.shape

print(f"data shape={CASES.shape}, weeks={cases_df.index.min().date()} to {cases_df.index.max().date()}")
display(parish_metadata[["parish_fips", "parish", "total_cases", "total_tests", "cenpop2020_population"]].head(10))


def load_spatial_weights() -> Dict[str, Optional[np.ndarray]]:
    weights: Dict[str, Optional[np.ndarray]] = {"nb_ar": None}
    weights["adjacency"] = off_diagonal_row_normalize(
        read_square_matrix(PRIOR_DIR / "C0_adjacency_diag80.csv", FIPS)
    )
    weights["distance"] = off_diagonal_row_normalize(
        read_square_matrix(PRIOR_DIR / "C0_distance_diag80_scale75km.csv", FIPS)
    )

    commute_path = PRIOR_DIR / "commute_jobs_work_by_home_LODES2019_JT01.csv"
    if commute_path.exists():
        commute = read_square_matrix(commute_path, FIPS)
        weights["lodes_sym"] = off_diagonal_row_normalize(0.5 * (commute + commute.T))
    else:
        commute_prior = read_square_matrix(PRIOR_DIR / "C0_commute_diag80_2019_JT01.csv", FIPS)
        weights["lodes_sym"] = off_diagonal_row_normalize(0.5 * (commute_prior + commute_prior.T))

    permuted_path = PRIOR_DIR / "C0_wrong_permuted_lodes_diag80_2019_JT01.csv"
    weights["lodes_permuted"] = off_diagonal_row_normalize(read_square_matrix(permuted_path, FIPS))
    return {key: weights[key] for key in CONFIG["models"]}


SPATIAL_WEIGHTS = load_spatial_weights()
weight_diagnostics = []
for name, weight in SPATIAL_WEIGHTS.items():
    if weight is None:
        continue
    weight_diagnostics.append({
        "model": name,
        "mean_nonzero_sources": float(np.mean(np.sum(weight > 0, axis=1))),
        "zero_rows": int(np.sum(weight.sum(axis=1) == 0)),
        "mean_row_sum": float(weight.sum(axis=1).mean()),
    })
display(pd.DataFrame(weight_diagnostics))


def make_split(weeks: pd.Index, split_name: str, test_weeks: int) -> Tuple[np.ndarray, np.ndarray, int]:
    cutoff = pd.Timestamp(split_name.replace("roll_", ""))
    train_end = int(np.searchsorted(weeks.to_numpy(), cutoff.to_datetime64(), side="right"))
    train_end = min(max(train_end, 8), len(weeks) - test_weeks)
    train_indices = np.arange(1, train_end)
    test_indices = np.arange(train_end, min(train_end + test_weeks, len(weeks)))
    return train_indices, test_indices, train_end


def make_time_basis(length: int, n_rbf: int = 5) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length)
    columns = [np.ones(length), x, x ** 2]
    for center in np.linspace(0.0, 1.0, n_rbf):
        columns.append(np.exp(-0.5 * ((x - center) / 0.25) ** 2))
    return np.vstack(columns).T.astype(np.float64)


TIME_BASIS = make_time_basis(T)


@dataclass
class Standardizer:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        scale = float(np.std(values))
        return cls(float(np.mean(values)), scale if scale > 1e-6 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


def lag_features(previous_cases: np.ndarray, current_tests: np.ndarray, weight: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    self_feature = np.log1p(np.maximum(previous_cases, 0.0))
    tests_per_1000 = np.maximum(current_tests, 0.0) / np.maximum(POPULATION, 1.0) * 1000.0
    test_feature = np.log1p(tests_per_1000)
    if weight is None:
        geo_feature = np.zeros(P, dtype=np.float64)
    else:
        lag_rate_per_100k = np.maximum(previous_cases, 0.0) / np.maximum(POPULATION, 1.0) * 100000.0
        geo_feature = weight @ np.log1p(lag_rate_per_100k)
    return self_feature, test_feature, geo_feature


def negbin_nll(mu: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    mu = torch.clamp(mu, min=1e-6, max=1e8)
    theta = torch.clamp(theta, min=1e-4, max=1e5)
    return -(
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + y * (torch.log(mu) - torch.log(theta + mu))
    )


class SpatialNBAutoregression(nn.Module):
    def __init__(self, n_parishes: int, n_basis: int, use_geo: bool):
        super().__init__()
        self.use_geo = use_geo
        self.parish_intercept = nn.Parameter(torch.zeros(n_parishes))
        self.time_beta = nn.Parameter(torch.zeros(n_basis))
        self.beta_self = nn.Parameter(torch.tensor(0.25))
        self.beta_tests = nn.Parameter(torch.tensor(0.25))
        self.beta_geo = nn.Parameter(torch.tensor(0.0)) if use_geo else None
        self.raw_theta = nn.Parameter(torch.tensor(2.0))

    def theta(self) -> torch.Tensor:
        return F.softplus(self.raw_theta) + 1e-3

    def forward(
        self,
        basis: torch.Tensor,
        self_lag: torch.Tensor,
        tests: torch.Tensor,
        geo_lag: torch.Tensor,
    ) -> torch.Tensor:
        log_mu = (
            self.parish_intercept[None, :]
            + (basis @ self.time_beta)[:, None]
            + self.beta_self * self_lag
            + self.beta_tests * tests
        )
        if self.use_geo:
            log_mu = log_mu + self.beta_geo * geo_lag
        return torch.exp(torch.clamp(log_mu, min=-12.0, max=14.0))


@dataclass
class FittedModel:
    model: SpatialNBAutoregression
    self_scaler: Standardizer
    test_scaler: Standardizer
    geo_scaler: Standardizer
    train_nll: float


def to_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=DEVICE)


def fit_model(model_name: str, weight: Optional[np.ndarray], train_indices: np.ndarray) -> FittedModel:
    self_rows, test_rows, geo_rows = [], [], []
    for time_index in train_indices:
        self_feature, test_feature, geo_feature = lag_features(CASES[time_index - 1], TESTS[time_index], weight)
        self_rows.append(self_feature)
        test_rows.append(test_feature)
        geo_rows.append(geo_feature)

    self_raw = np.vstack(self_rows)
    test_raw = np.vstack(test_rows)
    geo_raw = np.vstack(geo_rows)
    self_scaler = Standardizer.fit(self_raw)
    test_scaler = Standardizer.fit(test_raw)
    geo_scaler = Standardizer.fit(geo_raw) if weight is not None else Standardizer(0.0, 1.0)

    basis_t = to_tensor(TIME_BASIS[train_indices])
    self_t = to_tensor(self_scaler.transform(self_raw))
    tests_t = to_tensor(test_scaler.transform(test_raw))
    geo_t = to_tensor(geo_scaler.transform(geo_raw))
    y_t = to_tensor(CASES[train_indices])

    torch.manual_seed(0)
    model = SpatialNBAutoregression(P, TIME_BASIS.shape[1], use_geo=weight is not None).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    best_loss = math.inf
    best_state = None

    for epoch in range(CONFIG["epochs"]):
        optimizer.zero_grad()
        prediction = model(basis_t, self_t, tests_t, geo_t)
        elementwise = negbin_nll(prediction, y_t, model.theta())
        slope_penalty = model.beta_self.square() + model.beta_tests.square()
        if model.use_geo:
            slope_penalty = slope_penalty + model.beta_geo.square()
        regularization = 1e-3 * torch.mean(model.time_beta.square()) + 1e-3 * slope_penalty
        loss = torch.mean(elementwise) + regularization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(basis_t, self_t, tests_t, geo_t)
        train_nll = float(torch.mean(negbin_nll(prediction, y_t, model.theta())).cpu())
    return FittedModel(model, self_scaler, test_scaler, geo_scaler, train_nll)


def predict_week(
    fitted: FittedModel,
    time_index: int,
    previous_cases: np.ndarray,
    current_tests: np.ndarray,
    weight: Optional[np.ndarray],
) -> np.ndarray:
    self_raw, test_raw, geo_raw = lag_features(previous_cases, current_tests, weight)
    with torch.no_grad():
        prediction = fitted.model(
            to_tensor(TIME_BASIS[time_index:time_index + 1]),
            to_tensor(fitted.self_scaler.transform(self_raw)[None, :]),
            to_tensor(fitted.test_scaler.transform(test_raw)[None, :]),
            to_tensor(fitted.geo_scaler.transform(geo_raw)[None, :]),
        )
    return prediction.detach().cpu().numpy()[0]


PROTOCOLS = {
    "rolling_1step_observed_tests": {"recursive": False, "future_tests": "observed"},
    "fixed_origin_observed_tests": {"recursive": True, "future_tests": "observed"},
    "fixed_origin_last_observed_tests": {"recursive": True, "future_tests": "last_observed"},
}


def evaluate_forecast(
    split: str,
    model_name: str,
    weight: Optional[np.ndarray],
    fitted: FittedModel,
    test_indices: np.ndarray,
    train_end: int,
    protocol_name: str,
    protocol: Dict[str, object],
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[Dict[str, object]]]:
    predictions, observations = [], []
    horizon_rows: List[Dict[str, float]] = []
    parish_rows: List[Dict[str, object]] = []
    previous_recursive = CASES[train_end - 1].copy()
    theta = fitted.model.theta().detach().cpu()

    for horizon, time_index in enumerate(test_indices, start=1):
        if protocol["recursive"]:
            previous_cases = previous_recursive
        else:
            previous_cases = CASES[time_index - 1]
        if protocol["future_tests"] == "observed":
            current_tests = TESTS[time_index]
        else:
            current_tests = TESTS[train_end - 1]

        predicted = predict_week(fitted, time_index, previous_cases, current_tests, weight)
        observed = CASES[time_index]
        if protocol["recursive"]:
            previous_recursive = predicted.copy()

        nll_values = negbin_nll(to_tensor(predicted), to_tensor(observed), theta.to(DEVICE)).detach().cpu().numpy()
        predictions.append(predicted)
        observations.append(observed)
        horizon_rows.append({
            "scope": SCOPE,
            "split": split,
            "protocol": protocol_name,
            "model": model_name,
            "horizon": horizon,
            "week": str(pd.Timestamp(cases_df.index[time_index]).date()),
            "nll": float(np.mean(nll_values)),
            "mae": float(np.mean(np.abs(predicted - observed))),
            "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        })
        for parish_index, parish_fips in enumerate(FIPS):
            parish_rows.append({
                "scope": SCOPE,
                "split": split,
                "protocol": protocol_name,
                "model": model_name,
                "horizon": horizon,
                "week": str(pd.Timestamp(cases_df.index[time_index]).date()),
                "parish_fips": parish_fips,
                "parish": PARISH_NAMES[parish_index],
                "observed": float(observed[parish_index]),
                "predicted": float(predicted[parish_index]),
                "nll": float(nll_values[parish_index]),
            })

    predicted_all = np.vstack(predictions)
    observed_all = np.vstack(observations)
    nll_all = negbin_nll(to_tensor(predicted_all), to_tensor(observed_all), theta.to(DEVICE)).detach().cpu().numpy()
    run_row = {
        "scope": SCOPE,
        "split": split,
        "protocol": protocol_name,
        "model": model_name,
        "n_test_weeks": len(test_indices),
        "train_nll": fitted.train_nll,
        "test_nll": float(np.mean(nll_all)),
        "mae": float(np.mean(np.abs(predicted_all - observed_all))),
        "rmse": float(np.sqrt(np.mean((predicted_all - observed_all) ** 2))),
        "theta": float(fitted.model.theta().detach().cpu()),
        "beta_self": float(fitted.model.beta_self.detach().cpu()),
        "beta_tests": float(fitted.model.beta_tests.detach().cpu()),
        "beta_geo": float(fitted.model.beta_geo.detach().cpu()) if fitted.model.use_geo else 0.0,
    }
    return run_row, horizon_rows, parish_rows


run_rows: List[Dict[str, float]] = []
horizon_rows_all: List[Dict[str, float]] = []
parish_rows_all: List[Dict[str, object]] = []

for split in CONFIG["splits"]:
    train_indices, test_indices, train_end = make_split(cases_df.index, split, CONFIG["test_weeks"])
    print(
        f"\n{split}: train through {cases_df.index[train_end - 1].date()}, "
        f"test {cases_df.index[test_indices[0]].date()} to {cases_df.index[test_indices[-1]].date()}"
    )
    for model_name, weight in SPATIAL_WEIGHTS.items():
        fitted = fit_model(model_name, weight, train_indices)
        print(
            f"  {model_name:18s} train_nll={fitted.train_nll:.3f} "
            f"theta={float(fitted.model.theta().detach().cpu()):.2f} "
            f"beta_geo={float(fitted.model.beta_geo.detach().cpu()) if fitted.model.use_geo else 0.0:.3f}"
        )
        for protocol_name, protocol in PROTOCOLS.items():
            run_row, horizon_rows, parish_rows = evaluate_forecast(
                split,
                model_name,
                weight,
                fitted,
                test_indices,
                train_end,
                protocol_name,
                protocol,
            )
            run_rows.append(run_row)
            horizon_rows_all.extend(horizon_rows)
            parish_rows_all.extend(parish_rows)


run_metrics = pd.DataFrame(run_rows)
horizon_metrics = pd.DataFrame(horizon_rows_all)
parish_predictions = pd.DataFrame(parish_rows_all)

baseline = run_metrics[run_metrics["model"] == "nb_ar"][["split", "protocol", "test_nll", "mae"]].rename(
    columns={"test_nll": "baseline_nll", "mae": "baseline_mae"}
)
paired = run_metrics.merge(baseline, on=["split", "protocol"], how="left")
paired["delta_nll_vs_nb_ar"] = paired["test_nll"] - paired["baseline_nll"]
paired["delta_mae_vs_nb_ar"] = paired["mae"] - paired["baseline_mae"]

comparison_summary = (
    paired.groupby(["scope", "protocol", "model"], as_index=False)
    .agg(
        n_splits=("split", "nunique"),
        test_nll_mean=("test_nll", "mean"),
        test_nll_sd=("test_nll", "std"),
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
        delta_nll_vs_nb_ar_mean=("delta_nll_vs_nb_ar", "mean"),
        delta_nll_vs_nb_ar_sd=("delta_nll_vs_nb_ar", "std"),
        spatial_win_rate=("delta_nll_vs_nb_ar", lambda values: float(np.mean(values < 0))),
        theta_mean=("theta", "mean"),
        beta_self_mean=("beta_self", "mean"),
        beta_tests_mean=("beta_tests", "mean"),
        beta_geo_mean=("beta_geo", "mean"),
    )
    .sort_values(["protocol", "test_nll_mean"])
)

horizon_baseline = horizon_metrics[horizon_metrics["model"] == "nb_ar"][["split", "protocol", "horizon", "nll"]].rename(
    columns={"nll": "baseline_nll"}
)
horizon_paired = horizon_metrics.merge(horizon_baseline, on=["split", "protocol", "horizon"], how="left")
horizon_paired["delta_nll_vs_nb_ar"] = horizon_paired["nll"] - horizon_paired["baseline_nll"]
horizon_summary = (
    horizon_paired.groupby(["scope", "protocol", "model", "horizon"], as_index=False)
    .agg(
        nll_mean=("nll", "mean"),
        nll_sd=("nll", "std"),
        mae_mean=("mae", "mean"),
        delta_nll_vs_nb_ar_mean=("delta_nll_vs_nb_ar", "mean"),
    )
)

run_metrics.to_csv(TABLE_DIR / "run_metrics.csv", index=False)
paired.to_csv(TABLE_DIR / "paired_run_metrics.csv", index=False)
horizon_metrics.to_csv(TABLE_DIR / "horizon_metrics.csv", index=False)
horizon_summary.to_csv(TABLE_DIR / "horizon_summary.csv", index=False)
parish_predictions.to_csv(TABLE_DIR / "parish_predictions.csv", index=False)
comparison_summary.to_csv(TABLE_DIR / "model_comparison_summary.csv", index=False)

print("\nModel comparison (negative delta favors the spatial model):")
display(
    comparison_summary[
        [
            "protocol",
            "model",
            "test_nll_mean",
            "mae_mean",
            "delta_nll_vs_nb_ar_mean",
            "spatial_win_rate",
            "beta_geo_mean",
        ]
    ].round(4)
)


model_order = [name for name in CONFIG["models"] if name in comparison_summary["model"].unique()]
protocol_order = list(PROTOCOLS)
fig, axes = plt.subplots(1, len(protocol_order), figsize=(16, 4.2), sharey=False)
for axis, protocol_name in zip(axes, protocol_order):
    subset = comparison_summary[comparison_summary["protocol"] == protocol_name].set_index("model").reindex(model_order)
    colors = ["#444444" if name == "nb_ar" else "#2B6F8A" if name != "lodes_permuted" else "#A34A28" for name in model_order]
    axis.bar(np.arange(len(model_order)), subset["test_nll_mean"], color=colors)
    axis.set_xticks(np.arange(len(model_order)))
    axis.set_xticklabels(model_order, rotation=35, ha="right")
    axis.set_title(protocol_name.replace("_", " "))
    axis.set_ylabel("Held-out negative-binomial NLL")
    axis.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "protocol_model_nll_comparison.png", dpi=220, bbox_inches="tight")
plt.show()


strict_protocol = "fixed_origin_last_observed_tests"
strict_horizon = horizon_summary[horizon_summary["protocol"] == strict_protocol]
fig, axis = plt.subplots(figsize=(8.5, 4.5))
for model_name in model_order:
    if model_name == "nb_ar":
        continue
    subset = strict_horizon[strict_horizon["model"] == model_name].sort_values("horizon")
    axis.plot(subset["horizon"], subset["delta_nll_vs_nb_ar_mean"], marker="o", label=model_name)
axis.axhline(0.0, color="black", linewidth=1)
axis.set_xlabel("Forecast horizon (weeks)")
axis.set_ylabel("NLL difference versus NB-AR")
axis.set_title("Strict fixed-origin forecast: incremental value of geography")
axis.grid(alpha=0.25)
axis.legend(ncol=2)
fig.tight_layout()
fig.savefig(FIG_DIR / "strict_fixed_origin_horizon_delta_nll.png", dpi=220, bbox_inches="tight")
plt.show()


interpretation_lines = [
    f"Scope: {SCOPE}; mode: {RUN_MODE}.",
    "Negative delta NLL means that the spatial lag improves on the otherwise identical NB-AR model.",
]
for protocol_name in protocol_order:
    subset = comparison_summary[comparison_summary["protocol"] == protocol_name].sort_values("test_nll_mean")
    best = subset.iloc[0]
    interpretation_lines.append(
        f"{protocol_name}: best={best['model']}, mean NLL={best['test_nll_mean']:.4f}, "
        f"delta vs NB-AR={best['delta_nll_vs_nb_ar_mean']:.4f}."
    )

strict = comparison_summary[comparison_summary["protocol"] == strict_protocol].set_index("model")
plausible = [name for name in ["adjacency", "distance", "lodes_sym"] if name in strict.index]
plausible_winners = [name for name in plausible if strict.loc[name, "delta_nll_vs_nb_ar_mean"] < 0]
if "lodes_permuted" in strict.index:
    negative_control_delta = float(strict.loc["lodes_permuted", "delta_nll_vs_nb_ar_mean"])
    interpretation_lines.append(f"Permuted-LODES negative-control delta NLL={negative_control_delta:.4f} in the strict protocol.")
if plausible_winners:
    interpretation_lines.append("Plausible spatial priors with negative strict-protocol delta: " + ", ".join(plausible_winners) + ".")
else:
    interpretation_lines.append("No plausible spatial prior improves mean strict fixed-origin NLL over NB-AR.")

interpretation = "\n".join(interpretation_lines)
(OUTPUT_DIR / "interpretation.txt").write_text(interpretation, encoding="utf-8")
print("\n" + interpretation)
print(f"\nSaved tables to: {TABLE_DIR}")
print(f"Saved figures to: {FIG_DIR}")
