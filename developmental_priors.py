"""Developmental and accessibility human-factors priors for vulnerable road user cohorts.

Provides calibrated perceptual, motor, anthropometric, and curb-decision
parameters across nine vulnerable road user (VRU) and baseline cohorts:
``child_6_8yr``, ``child_7_9yr``, ``child_9_11yr``, ``child_12_14yr``,
``older_adult``, ``wheelchair_manual``, ``blind_low_vision``, ``adult_phone``,
and ``adult_baseline``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional
import numpy as np


@dataclass(frozen=True)
class CohortPerceptualMotorPrior:
    """Perceptual, motor, and curb-safety prior for a pedestrian cohort."""

    cohort_id: str
    eye_height_m: float
    weber_looming_fraction: float
    motor_lag_s: float
    free_walking_speed_mps: float
    critical_ttc_s: float
    distance_heuristic_weight: float
    curb_stopping_margin_m: float
    requires_curb_ramp: bool = False
    requires_tactile_or_aps: bool = False
    primary_modality: str = "visual"
    default_head_pitch_deg: float = 5.0


DEVELOPMENTAL_COHORT_PRIORS: Dict[str, CohortPerceptualMotorPrior] = {
    "child_6_8yr": CohortPerceptualMotorPrior(
        cohort_id="child_6_8yr",
        eye_height_m=1.15,
        weber_looming_fraction=0.28,
        motor_lag_s=0.34,
        free_walking_speed_mps=1.12,
        critical_ttc_s=3.10,
        distance_heuristic_weight=0.65,
        curb_stopping_margin_m=0.35,
    ),
    "child_7_9yr": CohortPerceptualMotorPrior(
        cohort_id="child_7_9yr",
        eye_height_m=1.24,
        weber_looming_fraction=0.24,
        motor_lag_s=0.30,
        free_walking_speed_mps=1.18,
        critical_ttc_s=3.25,
        distance_heuristic_weight=0.50,
        curb_stopping_margin_m=0.42,
    ),
    "child_9_11yr": CohortPerceptualMotorPrior(
        cohort_id="child_9_11yr",
        eye_height_m=1.38,
        weber_looming_fraction=0.18,
        motor_lag_s=0.25,
        free_walking_speed_mps=1.26,
        critical_ttc_s=3.45,
        distance_heuristic_weight=0.30,
        curb_stopping_margin_m=0.50,
    ),
    "child_12_14yr": CohortPerceptualMotorPrior(
        cohort_id="child_12_14yr",
        eye_height_m=1.55,
        weber_looming_fraction=0.13,
        motor_lag_s=0.20,
        free_walking_speed_mps=1.34,
        critical_ttc_s=3.55,
        distance_heuristic_weight=0.10,
        curb_stopping_margin_m=0.58,
    ),
    "older_adult": CohortPerceptualMotorPrior(
        cohort_id="older_adult",
        eye_height_m=1.62,
        weber_looming_fraction=0.19,
        motor_lag_s=0.32,
        free_walking_speed_mps=0.90,
        critical_ttc_s=4.80,
        distance_heuristic_weight=0.20,
        curb_stopping_margin_m=0.75,
    ),
    "wheelchair_manual": CohortPerceptualMotorPrior(
        cohort_id="wheelchair_manual",
        eye_height_m=1.22,
        weber_looming_fraction=0.12,
        motor_lag_s=0.38,
        free_walking_speed_mps=0.80,
        critical_ttc_s=5.20,
        distance_heuristic_weight=0.05,
        curb_stopping_margin_m=0.90,
        requires_curb_ramp=True,
    ),
    "blind_low_vision": CohortPerceptualMotorPrior(
        cohort_id="blind_low_vision",
        eye_height_m=1.65,
        weber_looming_fraction=0.55,
        motor_lag_s=0.28,
        free_walking_speed_mps=0.82,
        critical_ttc_s=5.80,
        distance_heuristic_weight=0.15,
        curb_stopping_margin_m=0.85,
        requires_tactile_or_aps=True,
        primary_modality="acoustic",
    ),
    "adult_phone": CohortPerceptualMotorPrior(
        cohort_id="adult_phone",
        eye_height_m=1.66,
        weber_looming_fraction=0.17,
        motor_lag_s=0.26,
        free_walking_speed_mps=1.22,
        critical_ttc_s=3.90,
        distance_heuristic_weight=0.12,
        curb_stopping_margin_m=0.48,
        default_head_pitch_deg=45.0,
    ),
    "adult_baseline": CohortPerceptualMotorPrior(
        cohort_id="adult_baseline",
        eye_height_m=1.68,
        weber_looming_fraction=0.10,
        motor_lag_s=0.18,
        free_walking_speed_mps=1.38,
        critical_ttc_s=3.65,
        distance_heuristic_weight=0.0,
        curb_stopping_margin_m=0.60,
    ),
}


def get_cohort_prior(cohort: str) -> CohortPerceptualMotorPrior:
    """Return the calibrated :class:`CohortPerceptualMotorPrior` for ``cohort``."""
    if cohort not in DEVELOPMENTAL_COHORT_PRIORS:
        raise KeyError(
            f"Unknown developmental/VRU cohort {cohort!r}. "
            f"Available: {sorted(DEVELOPMENTAL_COHORT_PRIORS)}"
        )
    return DEVELOPMENTAL_COHORT_PRIORS[cohort]


def compute_perceived_effective_ttc(
    distance_m: float,
    speed_mps: float,
    cohort: str | CohortPerceptualMotorPrior = "adult_baseline",
    ref_speed_mps: float = 11.176,
    include_weber_noise: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute cohort-specific perceived Time-to-Collision (TTC) combining true
    optical tau (``D / V``) with the developmental distance-over-speed heuristic
    (``D / V_ref``; Wann et al., 2011; Connelly et al., 1998) and optional
    multiplicative Weber looming noise.
    """
    prior = (
        cohort
        if isinstance(cohort, CohortPerceptualMotorPrior)
        else get_cohort_prior(cohort)
    )
    if distance_m <= 0.0:
        return 0.0
    if speed_mps <= 1e-6:
        return math.inf

    true_ttc = float(distance_m) / float(speed_mps)
    dist_ttc = float(distance_m) / max(float(ref_speed_mps), 0.5)
    w_dist = prior.distance_heuristic_weight
    eff_ttc = (1.0 - w_dist) * true_ttc + w_dist * dist_ttc

    if include_weber_noise and prior.weber_looming_fraction > 0.0:
        gen = rng if rng is not None else np.random.default_rng()
        # Multiplicative log-normal Weber sensory noise with median 1.0
        sigma = prior.weber_looming_fraction
        eff_ttc *= float(gen.lognormal(mean=0.0, sigma=sigma))

    return max(0.0, eff_ttc)


def evaluate_curb_crossing_rule(
    distance_m: float,
    speed_mps: float,
    crossing_width_m: float = 7.0,
    cohort: str | CohortPerceptualMotorPrior = "adult_baseline",
    has_curb_ramp: bool = True,
    has_tactile_or_aps: bool = True,
    ref_speed_mps: float = 11.176,
) -> Dict[str, float | bool | str]:
    """Evaluate curb gap-acceptance rule accounting for developmental TTC bias,
    initiation motor lag, traversal duration, and physical curb accessibility.
    """
    prior = (
        cohort
        if isinstance(cohort, CohortPerceptualMotorPrior)
        else get_cohort_prior(cohort)
    )
    infrastructure_accessible = True
    if prior.requires_curb_ramp and not has_curb_ramp:
        infrastructure_accessible = False
    if prior.requires_tactile_or_aps and not has_tactile_or_aps:
        infrastructure_accessible = False

    perceived_ttc_s = compute_perceived_effective_ttc(
        distance_m=distance_m,
        speed_mps=speed_mps,
        cohort=prior,
        ref_speed_mps=ref_speed_mps,
        include_weber_noise=False,
    )
    true_ttc_s = (
        math.inf if speed_mps <= 1e-6 else max(0.0, float(distance_m) / float(speed_mps))
    )
    traversal_time_s = prior.motor_lag_s + (
        max(0.0, float(crossing_width_m)) / prior.free_walking_speed_mps
    )
    required_gap_s = max(prior.critical_ttc_s, traversal_time_s)
    accepts_gap = bool(infrastructure_accessible and (perceived_ttc_s >= required_gap_s))
    true_post_encroachment_s = true_ttc_s - traversal_time_s

    return {
        "cohort_id": prior.cohort_id,
        "infrastructure_accessible": infrastructure_accessible,
        "perceived_ttc_s": perceived_ttc_s,
        "true_ttc_s": true_ttc_s,
        "traversal_time_s": traversal_time_s,
        "required_gap_s": required_gap_s,
        "accepts_gap": accepts_gap,
        "true_post_encroachment_s": true_post_encroachment_s,
        "is_hazardous_misjudgment": bool(accepts_gap and true_post_encroachment_s < 0.5),
    }
