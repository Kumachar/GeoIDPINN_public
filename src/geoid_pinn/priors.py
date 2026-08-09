"""Training-only residualized, prewhitened cross-correlation priors.

The returned matrix follows the GeoID convention: rows are recipients,
columns are sources, and every row sums to one. The prior is empirical and
must be rebuilt independently inside every rolling-origin training split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


EPS = 1e-10


@dataclass
class PWCCFPriorResult:
    C0: np.ndarray
    raw_score: np.ndarray
    weighted_score: np.ndarray
    best_lag: np.ndarray
    stability: np.ndarray
    pearson_residuals: np.ndarray
    fitted_mean: np.ndarray
    ar1_phi: np.ndarray
    theta: float
    edge_table: pd.DataFrame
    diagnostics: Dict[str, float]


def _time_basis(length: int, n_rbf: int = 5) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, length)
    columns = [x, x ** 2]
    for center in np.linspace(-1.0, 1.0, n_rbf):
        columns.append(np.exp(-0.5 * ((x - center) / 0.50) ** 2))
    return np.vstack(columns).T


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (values - mean) / scale


def _fit_pooled_nb_residuals(
    cases: np.ndarray,
    tests: np.ndarray,
    population: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit a stable pooled NB-AR mean model and return Pearson residuals.

    The design contains parish intercepts, a shared smooth time basis,
    log lagged local cases, and log testing rate. A Poisson fit supplies a
    method-of-moments NB2 dispersion estimate, followed by one NB-GLM refit.
    """
    n_time, n_parish = cases.shape
    if n_time < 8:
        raise ValueError("PWCCF prior requires at least eight training weeks.")

    response = np.asarray(cases[1:], dtype=np.float64).reshape(-1)
    lag_cases = np.log1p(np.asarray(cases[:-1], dtype=np.float64))
    test_rate = np.asarray(tests[1:], dtype=np.float64) / np.maximum(population[None, :], 1.0) * 1000.0
    log_test_rate = np.log1p(np.maximum(test_rate, 0.0))

    parish_intercepts = np.tile(np.eye(n_parish, dtype=np.float64), (n_time - 1, 1))
    temporal = np.repeat(_time_basis(n_time)[1:], n_parish, axis=0)
    design = np.column_stack([
        parish_intercepts,
        temporal,
        _standardize(lag_cases).reshape(-1),
        _standardize(log_test_rate).reshape(-1),
    ])

    poisson = sm.GLM(response, design, family=sm.families.Poisson()).fit(maxiter=150, disp=0)
    poisson_mean = np.maximum(poisson.predict(design), 1e-6)
    alpha_numerator = np.sum((response - poisson_mean) ** 2 - response)
    alpha_denominator = np.sum(poisson_mean ** 2)
    alpha = float(np.clip(alpha_numerator / max(alpha_denominator, EPS), 1e-4, 10.0))

    nb_fit = sm.GLM(
        response,
        design,
        family=sm.families.NegativeBinomial(alpha=alpha),
    ).fit(maxiter=200, disp=0)
    fitted = np.maximum(nb_fit.predict(design), 1e-6).reshape(n_time - 1, n_parish)
    observed = cases[1:]

    # Refresh alpha once using the NB fitted mean. This avoids an overly
    # confident residual scale when the Poisson initialization overfits.
    alpha_numerator = np.sum((observed - fitted) ** 2 - fitted)
    alpha_denominator = np.sum(fitted ** 2)
    alpha = float(np.clip(alpha_numerator / max(alpha_denominator, EPS), 1e-4, 10.0))
    variance = fitted + alpha * fitted ** 2
    residuals = (observed - fitted) / np.sqrt(np.maximum(variance, 1e-6))
    return residuals, fitted, 1.0 / alpha


def _estimate_ar1(series: np.ndarray) -> float:
    previous = np.asarray(series[:-1], dtype=np.float64)
    current = np.asarray(series[1:], dtype=np.float64)
    denominator = float(previous @ previous)
    if denominator < EPS:
        return 0.0
    return float(np.clip((previous @ current) / denominator, -0.80, 0.80))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 6:
        return 0.0
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt((x @ x) * (y @ y)))
    if denominator < EPS:
        return 0.0
    return float(np.clip((x @ y) / denominator, -1.0, 1.0))


def _pwccf_scores(
    residuals: np.ndarray,
    lags: Sequence[int],
    candidate_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive maximum CCF scores, best lags, and source AR(1)."""
    n_parish = residuals.shape[1]
    scores = np.zeros((n_parish, n_parish), dtype=np.float64)
    best_lags = np.full((n_parish, n_parish), -1, dtype=np.int64)
    phi = np.zeros(n_parish, dtype=np.float64)

    for source in range(n_parish):
        phi[source] = _estimate_ar1(residuals[:, source])
        source_white = residuals[1:, source] - phi[source] * residuals[:-1, source]
        for recipient in range(n_parish):
            if recipient == source or not candidate_mask[recipient, source]:
                continue
            recipient_white = residuals[1:, recipient] - phi[source] * residuals[:-1, recipient]
            correlations = []
            for lag in lags:
                if lag == 0:
                    corr = _safe_corr(recipient_white, source_white)
                elif lag < len(source_white) - 5:
                    # Positive lag means the source leads the recipient.
                    corr = _safe_corr(recipient_white[lag:], source_white[:-lag])
                else:
                    corr = 0.0
                correlations.append(corr)
            best_index = int(np.argmax(correlations))
            best_corr = float(correlations[best_index])
            if best_corr > 0.0:
                scores[recipient, source] = best_corr
                best_lags[recipient, source] = int(lags[best_index])
    return scores, best_lags, phi


def _moving_block_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    block_length = int(np.clip(block_length, 1, length))
    starts = np.arange(max(length - block_length + 1, 1))
    sampled = []
    while len(sampled) < length:
        start = int(rng.choice(starts))
        sampled.extend(range(start, min(start + block_length, length)))
    return np.asarray(sampled[:length], dtype=np.int64)


def _bootstrap_stability(
    residuals: np.ndarray,
    original_scores: np.ndarray,
    original_lags: np.ndarray,
    lags: Sequence[int],
    candidate_mask: np.ndarray,
    repetitions: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    if repetitions <= 0:
        return (original_scores > 0).astype(np.float64)
    rng = np.random.default_rng(seed)
    support = np.zeros_like(original_scores, dtype=np.float64)
    eligible = original_scores > 0
    for _ in range(repetitions):
        indices = _moving_block_indices(len(residuals), block_length, rng)
        bootstrap_scores, bootstrap_lags, _ = _pwccf_scores(residuals[indices], lags, candidate_mask)
        same_or_nearby_lag = np.abs(bootstrap_lags - original_lags) <= 1
        support += eligible & (bootstrap_scores > 0) & same_or_nearby_lag
    return support / float(repetitions)


def _make_prior(
    weighted_scores: np.ndarray,
    diagonal_mass: float,
    top_k: Optional[int],
) -> np.ndarray:
    n_parish = weighted_scores.shape[0]
    C0 = np.zeros_like(weighted_scores, dtype=np.float64)
    diagonal_mass = float(np.clip(diagonal_mass, 0.0, 1.0))
    for recipient in range(n_parish):
        row = np.maximum(weighted_scores[recipient].copy(), 0.0)
        row[recipient] = 0.0
        if top_k is not None and 0 < top_k < n_parish - 1:
            keep = np.argpartition(row, -top_k)[-top_k:]
            mask = np.zeros(n_parish, dtype=bool)
            mask[keep] = True
            row[~mask] = 0.0
        total = float(row.sum())
        if total <= EPS:
            C0[recipient, recipient] = 1.0
            continue
        C0[recipient, recipient] = diagonal_mass
        C0[recipient] += (1.0 - diagonal_mass) * row / total
    return C0


def permute_prior_offdiagonal(C0: np.ndarray, seed: int = 2026) -> np.ndarray:
    """Permute source labels within each row while preserving row weights."""
    rng = np.random.default_rng(seed)
    result = np.zeros_like(C0, dtype=np.float64)
    n_parish = C0.shape[0]
    for recipient in range(n_parish):
        result[recipient, recipient] = C0[recipient, recipient]
        off_indices = np.delete(np.arange(n_parish), recipient)
        result[recipient, off_indices] = rng.permutation(C0[recipient, off_indices])
    return result


def build_nb_residual_pwccf_prior(
    cases: np.ndarray,
    tests: np.ndarray,
    population: np.ndarray,
    train_mask: Optional[np.ndarray] = None,
    diagonal_mass: float = 0.80,
    lags: Iterable[int] = (0, 1, 2),
    bootstrap_repetitions: int = 200,
    block_length: int = 3,
    stability_threshold: float = 0.60,
    top_k: Optional[int] = 8,
    candidate_mask: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
    seed: int = 2026,
) -> PWCCFPriorResult:
    """Build a split-specific NB-residual PWCCF prior.

    `cases`, `tests`, and `train_mask` may cover the full observation window,
    but only the leading contiguous training block is used. Rows of C0 are
    recipients and columns are sources.
    """
    cases = np.asarray(cases, dtype=np.float64)
    tests = np.asarray(tests, dtype=np.float64)
    population = np.asarray(population, dtype=np.float64)
    if cases.shape != tests.shape:
        raise ValueError("cases and tests must have the same shape")
    if cases.ndim != 2 or population.shape != (cases.shape[1],):
        raise ValueError("Expected cases/tests=(time, parish) and population=(parish,)")

    if train_mask is None:
        train_end = len(cases)
    else:
        train_mask = np.asarray(train_mask, dtype=bool)
        train_indices = np.flatnonzero(train_mask)
        if len(train_indices) == 0 or not np.array_equal(train_indices, np.arange(train_indices[-1] + 1)):
            raise ValueError("train_mask must select a non-empty leading contiguous block")
        train_end = int(train_indices[-1] + 1)

    cases_train = cases[:train_end]
    tests_train = tests[:train_end]
    residuals, fitted, theta = _fit_pooled_nb_residuals(cases_train, tests_train, population)

    n_parish = cases.shape[1]
    if candidate_mask is None:
        candidate_mask = np.ones((n_parish, n_parish), dtype=bool)
        np.fill_diagonal(candidate_mask, False)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool).copy()
        if candidate_mask.shape != (n_parish, n_parish):
            raise ValueError("candidate_mask must have shape (parish, parish)")
        np.fill_diagonal(candidate_mask, False)

    lags = tuple(sorted({int(lag) for lag in lags if int(lag) >= 0}))
    if not lags:
        raise ValueError("At least one non-negative lag is required")

    raw_score, best_lag, phi = _pwccf_scores(residuals, lags, candidate_mask)
    stability = _bootstrap_stability(
        residuals,
        raw_score,
        best_lag,
        lags,
        candidate_mask,
        repetitions=bootstrap_repetitions,
        block_length=block_length,
        seed=seed,
    )
    weighted_score = raw_score * stability
    weighted_score[stability < stability_threshold] = 0.0
    C0 = _make_prior(weighted_score, diagonal_mass=diagonal_mass, top_k=top_k)

    labels = list(labels) if labels is not None else [str(i) for i in range(n_parish)]
    edge_rows = []
    for recipient in range(n_parish):
        for source in range(n_parish):
            if recipient == source or C0[recipient, source] <= 0:
                continue
            edge_rows.append({
                "recipient_index": recipient,
                "recipient": labels[recipient],
                "source_index": source,
                "source": labels[source],
                "raw_correlation": raw_score[recipient, source],
                "best_lag_weeks": int(best_lag[recipient, source]),
                "bootstrap_stability": stability[recipient, source],
                "weighted_score": weighted_score[recipient, source],
                "C0_weight": C0[recipient, source],
            })
    edge_table = pd.DataFrame(edge_rows)
    if not edge_table.empty:
        edge_table = edge_table.sort_values("C0_weight", ascending=False).reset_index(drop=True)

    diagnostics = {
        "train_weeks": float(train_end),
        "residual_weeks": float(len(residuals)),
        "theta": float(theta),
        "bootstrap_repetitions": float(bootstrap_repetitions),
        "stable_edges_before_top_k": float(np.sum(weighted_score > 0)),
        "stable_edges": float(np.sum((C0 > 0) & (~np.eye(n_parish, dtype=bool)))),
        "rows_with_spatial_edges": float(np.sum((1.0 - np.diag(C0)) > EPS)),
        "mean_diagonal_mass": float(np.diag(C0).mean()),
        "mean_effective_sources": float(np.mean(np.exp(-np.sum(C0 * np.log(C0 + EPS), axis=1)))),
    }
    return PWCCFPriorResult(
        C0=C0,
        raw_score=raw_score,
        weighted_score=weighted_score,
        best_lag=best_lag,
        stability=stability,
        pearson_residuals=residuals,
        fitted_mean=fitted,
        ar1_phi=phi,
        theta=theta,
        edge_table=edge_table,
        diagnostics=diagnostics,
    )
