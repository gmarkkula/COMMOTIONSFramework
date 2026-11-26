"""Maximum-likelihood Drift-Diffusion & Bayesian eHMI calibration on HIKER CAVE data.

Fits ``crit_ttc_s``, ``ddm_drift_mu``, and ``delta_log_odds_ehmi`` directly on
the 8,547 empirical pedestrian road-crossing trials in
``SCPaper/data/hiker_cts.csv`` (University of Leeds HIKER pedestrian simulator).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit


DEFAULT_HIKER_CSV = Path(__file__).resolve().parent / "data" / "hiker_cts.csv"


@dataclass(frozen=True)
class HikerCalibrationResult:
    """Fitted cognitive gap-acceptance & eHMI parameters from HIKER CAVE trials."""

    n_trials: int
    crit_ttc_s: float
    ddm_drift_mu: float
    delta_log_odds_ehmi: float
    bound_b: float
    nondecision_time_s: float
    log_likelihood: float
    null_log_likelihood: float
    mcfadden_pseudo_r2: float
    aic: float
    bic: float
    predicted_accept_rates: Dict[str, float]
    empirical_accept_rates: Dict[str, float]


def fit_hiker_mle_parameters(
    csv_path: str | Path | None = None,
    early_margin_s: float = 0.10,
) -> HikerCalibrationResult:
    """Calibrate ``crit_ttc_s``, ``ddm_drift_mu``, and ``delta_log_odds_ehmi``
    by joint maximum likelihood across all 8,547 trials in ``hiker_cts.csv``.

    Choice & reaction-time formulation (symmetric Ratcliff / Palmer-Huk-Shadlen DDM):
        * Effective drift: ``v_i = ddm_drift_mu * (time_gap_i - crit_ttc_s)``
        * Initial gap acceptance log-odds:
          ``L_i = 2 * B * v_i + delta_log_odds_ehmi * I[has_ehmi_i]``
        * Expected first-passage crossing initiation time on accepted initial gaps:
          ``E[CIT_i] = t_nd + (B / u_i) * tanh(B * u_i)`` where
          ``u_i = v_i + delta_log_odds_ehmi / (2 * B) * I[has_ehmi_i]``.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_HIKER_CSV
    df = pd.read_csv(path)
    n_trials = int(len(df))
    if n_trials == 0:
        raise ValueError(f"Empty HIKER dataset at {path}")

    gap = df["time_gap"].to_numpy(dtype=float)
    has_ehmi = df["has_ehmi"].astype(bool).to_numpy(dtype=float)
    cit = df["crossing_time"].to_numpy(dtype=float)

    # Pedestrian accepted the offered initial gap if they initiated crossing before
    # the vehicle's unbraked arrival window (time_gap - early_margin_s).
    accepted = (~np.isnan(cit)) & (cit < (gap - early_margin_s))
    y = accepted.astype(float)

    # Subset of accepted initial-gap trials with valid positive CIT for RT likelihood
    rt_mask = accepted & (cit > 0.05) & (cit < 3.0)
    cit_obs = cit[rt_mask]
    gap_rt = gap[rt_mask]
    ehmi_rt = has_ehmi[rt_mask]

    p_bar = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    null_ll = float(np.sum(y * math.log(p_bar) + (1.0 - y) * math.log(1.0 - p_bar)))

    def neg_joint_log_likelihood(theta: np.ndarray) -> float:
        crit_ttc, drift_mu, dlo_ehmi, bound_b, t_nd, sigma_rt = theta
        eff_logit = 2.0 * bound_b * drift_mu * (gap - crit_ttc) + dlo_ehmi * has_ehmi
        choice_ll = np.sum(y * log_expit(eff_logit) + (1.0 - y) * log_expit(-eff_logit))

        # Decision time predicted mean on accepted initial gaps
        u_rt = drift_mu * (gap_rt - crit_ttc) + (dlo_ehmi / (2.0 * bound_b)) * ehmi_rt
        bu = bound_b * u_rt
        # Smoothly handle u_rt -> 0 limit where tanh(B*u)/u -> B
        mean_dec_t = np.where(
            np.abs(bu) > 1e-4,
            (bound_b / np.where(np.abs(u_rt) > 1e-6, u_rt, 1.0)) * np.tanh(bu),
            bound_b ** 2,
        )
        pred_cit = t_nd + mean_dec_t
        rt_ll = -0.5 * np.sum(
            ((cit_obs - pred_cit) / sigma_rt) ** 2 + math.log(2.0 * math.pi * (sigma_rt ** 2))
        )
        # Weight RT sub-likelihood lightly (0.15) so binary choice log-odds remain exact MLE
        return -float(choice_ll + 0.15 * rt_ll)

    x0 = np.array([3.85, 0.70, 0.85, 0.90, 0.18, 0.25], dtype=float)
    bounds = [
        (2.0, 6.0),   # crit_ttc_s
        (0.10, 4.0),  # ddm_drift_mu
        (-2.0, 4.0),  # delta_log_odds_ehmi
        (0.30, 2.5),  # bound_b
        (0.05, 0.60), # nondecision_time_s
        (0.05, 1.50), # sigma_rt
    ]
    res = minimize(neg_joint_log_likelihood, x0=x0, method="L-BFGS-B", bounds=bounds)
    crit_ttc, drift_mu, dlo_ehmi, bound_b, t_nd, _ = [float(v) for v in res.x]

    eff_logit_opt = 2.0 * bound_b * drift_mu * (gap - crit_ttc) + dlo_ehmi * has_ehmi
    choice_ll_opt = float(
        np.sum(y * log_expit(eff_logit_opt) + (1.0 - y) * log_expit(-eff_logit_opt))
    )
    k_params = 5
    aic = 2.0 * k_params - 2.0 * choice_ll_opt
    bic = k_params * math.log(n_trials) - 2.0 * choice_ll_opt
    pseudo_r2 = 1.0 - (choice_ll_opt / null_ll)

    pred_rates: Dict[str, float] = {}
    emp_rates: Dict[str, float] = {}
    for tg in sorted(np.unique(gap)):
        for eh in (False, True):
            key = f"gap_{int(tg)}s_ehmi_{int(eh)}"
            mask = (gap == tg) & (has_ehmi == float(eh))
            emp_rates[key] = float(y[mask].mean())
            lo = 2.0 * bound_b * drift_mu * (tg - crit_ttc) + dlo_ehmi * float(eh)
            pred_rates[key] = float(expit(lo))

    return HikerCalibrationResult(
        n_trials=n_trials,
        crit_ttc_s=crit_ttc,
        ddm_drift_mu=drift_mu,
        delta_log_odds_ehmi=dlo_ehmi,
        bound_b=bound_b,
        nondecision_time_s=t_nd,
        log_likelihood=choice_ll_opt,
        null_log_likelihood=null_ll,
        mcfadden_pseudo_r2=pseudo_r2,
        aic=aic,
        bic=bic,
        predicted_accept_rates=pred_rates,
        empirical_accept_rates=emp_rates,
    )


if __name__ == "__main__":
    fit = fit_hiker_mle_parameters()
    print(
        f"Calibrated on {fit.n_trials} HIKER trials: "
        f"crit_ttc_s={fit.crit_ttc_s:.3f}s, "
        f"ddm_drift_mu={fit.ddm_drift_mu:.3f}, "
        f"delta_log_odds_ehmi={fit.delta_log_odds_ehmi:+.3f}, "
        f"McFadden_R2={fit.mcfadden_pseudo_r2:.3f}"
    )
