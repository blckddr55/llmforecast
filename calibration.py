"""Hierarchical Platt scaling for forecast calibration.

Calibrates raw probabilities with a shared logistic transform plus a per-source
intercept offset:

    p_cal = expit(a * logit(p_hat) + b + delta_s)

`a` (slope) and `b` (intercept) are global; `delta_s` is a per-source offset that
lets each source carry its own bias correction. The per-source offsets are L2
regularized toward zero, so sources with little data shrink back to the pooled
global fit — the "hierarchical" part.

Parameters are fit by minimizing the regularized negative log-likelihood, and the
calibration is evaluated honestly with leave-one-out cross-validation.

Run with:  uv run calibration.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

# Clip raw/calibrated probabilities away from {0, 1} so logit/log stay finite.
EPS = 1e-6


def safe_logit(p, eps: float = EPS):
    """logit with the input clipped to [eps, 1 - eps] to avoid domain errors."""
    return logit(np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps))


def calibrate(logit_p, a, b, delta, source_idx):
    """Apply the calibration map, returning probabilities clipped to (0, 1)."""
    z = a * logit_p + b + delta[source_idx]
    return np.clip(expit(z), EPS, 1.0 - EPS)


def negative_log_likelihood(params, logit_p, y, source_idx, n_sources, lam):
    """Regularized NLL. params = [a, b, delta_0, ..., delta_{n_sources-1}].

    The L2 penalty applies ONLY to the per-source offsets delta_s — not to the
    global slope a or intercept b.
    """
    a, b = params[0], params[1]
    delta = params[2:]
    p_cal = calibrate(logit_p, a, b, delta, source_idx)
    nll = -np.sum(y * np.log(p_cal) + (1.0 - y) * np.log1p(-p_cal))
    penalty = lam * np.sum(delta**2)
    return nll + penalty


def fit(logit_p, y, source_idx, n_sources, lam=1.0):
    """Fit (a, b, delta) by minimizing the regularized NLL with scipy."""
    x0 = np.concatenate(([1.0, 0.0], np.zeros(n_sources)))
    result = minimize(
        negative_log_likelihood,
        x0,
        args=(logit_p, y, source_idx, n_sources, lam),
        method="L-BFGS-B",
    )
    a, b = result.x[0], result.x[1]
    delta = result.x[2:]
    return a, b, delta


def loocv_calibrate(p_hat, y, source_idx, lam=1.0):
    """Leave-one-out calibrated probabilities.

    For each of the N questions: train (a, b, delta) on the other N-1, then
    calibrate the held-out question with those parameters. The number of sources
    is fixed from the full dataset so per-source offsets stay aligned across
    folds (a source absent from a fold's training set keeps delta ~ 0 via the L2
    penalty). Returns an array of N calibrated probabilities.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    source_idx = np.asarray(source_idx, dtype=int)
    n = len(p_hat)
    n_sources = int(source_idx.max()) + 1 if n else 0
    logit_p = safe_logit(p_hat)

    calibrated = np.empty(n)
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        a, b, delta = fit(logit_p[train], y[train], source_idx[train], n_sources, lam)
        calibrated[i] = calibrate(
            logit_p[i : i + 1], a, b, delta, source_idx[i : i + 1]
        )[0]
    return calibrated


def _log_loss(p, y):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def _brier(p, y):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def main() -> None:
    rng = np.random.default_rng(0)
    n, n_sources = 200, 3

    # Latent "true" probabilities and sampled binary outcomes.
    true_logit = rng.normal(0.0, 1.5, size=n)
    true_p = expit(true_logit)
    y = (rng.random(n) < true_p).astype(int)

    # Raw forecaster: overconfident (slope > 1) with a per-source bias.
    source_idx = rng.integers(0, n_sources, size=n)
    source_bias = np.array([-0.8, 0.0, 0.8])
    raw_logit = (
        1.6 * true_logit + 0.2 + source_bias[source_idx] + rng.normal(0.0, 0.3, size=n)
    )
    p_hat = expit(raw_logit)

    calibrated = loocv_calibrate(p_hat, y, source_idx, lam=1.0)

    print("Leave-one-out calibration (first 8 questions):")
    print(f"  {'src':>3} {'p_hat':>7} {'p_cal':>7} {'y':>3}")
    for i in range(8):
        print(f"  {source_idx[i]:>3} {p_hat[i]:>7.3f} {calibrated[i]:>7.3f} {y[i]:>3}")

    print("\nCalibration quality vs outcomes (lower is better):")
    print(
        f"  log loss : raw {_log_loss(p_hat, y):.4f}  ->  "
        f"calibrated {_log_loss(calibrated, y):.4f}"
    )
    print(
        f"  Brier    : raw {_brier(p_hat, y):.4f}  ->  "
        f"calibrated {_brier(calibrated, y):.4f}"
    )


if __name__ == "__main__":
    main()
