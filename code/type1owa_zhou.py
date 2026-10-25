"""
Python port of the Type1OWA 1.0 R implementation by Francisco Mata,
based on the alpha-level Type-1 OWA algorithms of Zhou et al.

This module intentionally preserves the computational flow of the supplied
R package and the FuzzyIEEE 2015 example:

  trimf2 -> alphaCutsGeneralMF -> itvOWA_withNullSets
          -> memDegByAlphaCuts_withNullCheck -> centroid

Only NumPy is required.

References cited by the original R package:
- S.-M. Zhou, F. Chiclana, R. I. John, J. M. Garibaldi,
  "Alpha-level aggregation: a practical approach to type-1 OWA operation
   for aggregating uncertain information with applications to breast cancer
   treatments", IEEE TKDE 23(10), 1455-1468, 2011.
- S.-M. Zhou, F. Chiclana, R. I. John, J. M. Garibaldi,
  "Type-1 OWA operators for aggregating uncertain information with uncertain
   weights induced by type-2 linguistic quantifiers", Fuzzy Sets and Systems
   159(24), 3281-3296, 2008.

The source R package supplied by the user is GPL-licensed. This port is a research-oriented translation of that supplied implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np


@dataclass
class FuzzySet:
    domain: np.ndarray
    mf_value: np.ndarray

    def __post_init__(self) -> None:
        self.domain = np.asarray(self.domain, dtype=float)
        self.mf_value = np.asarray(self.mf_value, dtype=float)
        if self.domain.ndim != 1 or self.mf_value.ndim != 1:
            raise ValueError("domain and mf_value must be one-dimensional")
        if self.domain.size != self.mf_value.size:
            raise ValueError("domain and mf_value must have the same length")
        if self.domain.size == 0:
            raise ValueError("fuzzy set cannot be empty")


@dataclass
class IntervalCut:
    itv_l: float
    itv_r: float
    alpha: float | None = None

    @property
    def is_null(self) -> bool:
        return bool(np.isnan(self.itv_l) or np.isnan(self.itv_r))


def trimf2(x: Sequence[float], mf_params: Sequence[float]) -> np.ndarray:
    """Port of R ``trimf2``: triangular membership function.

    Parameters
    ----------
    x:
        Domain samples.
    mf_params:
        ``(a, b, c, max_membership)``.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(mf_params, dtype=float)
    if p.size != 4:
        raise ValueError("mf_params must contain (a, b, c, max_membership)")

    a, b, c, max_degree = p
    if not (a <= b <= c):
        raise ValueError("Expected a <= b <= c")

    # Equivalent to the original R formulas, with explicit shoulder handling
    # to avoid 0/0 if a==b or b==c.
    if b == a:
        fs1 = np.where(x >= b, max_degree, 0.0)
    else:
        fs1 = np.minimum(np.maximum((x - a) / (b - a), 0.0), max_degree)

    if c == b:
        fs2 = np.where(x <= b, max_degree, 0.0)
    else:
        fs2 = np.minimum(np.maximum((c - x) / (c - b), 0.0), max_degree)

    return np.minimum(fs1, fs2)


def alpha_cuts_general_mf(a: FuzzySet, alpha: float) -> IntervalCut:
    """Port of R ``alphaCutsGeneralMF``.

    The alpha-cut endpoints are selected from the *discretized domain*, just
    as in the R package; no interpolation is introduced.
    """
    alpha = float(alpha)
    max_mf = float(np.max(a.mf_value))

    if alpha <= max_mf:
        ind = a.mf_value >= alpha
        if not np.any(ind):
            return IntervalCut(np.nan, np.nan, alpha)
        left = float(np.min(a.domain[ind]))
        right = float(np.max(a.domain[ind]))
    else:
        left = np.nan
        right = np.nan

    return IntervalCut(left, right, alpha)


def centroid(domain: Sequence[float], mf_value: Sequence[float]) -> float:
    """Port of R ``centroid``: discrete center of gravity ``sum(x*mu)/sum(mu)``."""
    x = np.asarray(domain, dtype=float)
    mf = np.asarray(mf_value, dtype=float)
    if x.size != mf.size:
        raise ValueError("domain and mf_value must have the same length")
    denominator = float(np.sum(mf))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(x * mf) / denominator)


def fuzzy_set_weights(
    num_obj: int,
    wty: str = "crisp",
    specific_wty: str = "singletonAvg",
    partition_x: str = "equal",
) -> list[FuzzySet]:
    """Port of the behavior implemented by R ``fuzzySetWeights``.

    The supplied Type1OWA 1.0 package implements only the branch used by the
    conference example: equal crisp singleton weights ``1/num_obj``.
    ``partition_x`` is retained for API correspondence with the R function.
    """
    if num_obj <= 0:
        raise ValueError("num_obj must be positive")
    if wty != "crisp" or specific_wty != "singletonAvg":
        raise NotImplementedError(
            "The supplied R fuzzySetWeights implementation only defines "
            "wty='crisp', specificWty='singletonAvg'."
        )

    q = 1.0 / num_obj
    return [FuzzySet(np.array([q]), np.array([1.0])) for _ in range(num_obj)]


def crisp_owa_weights(weights: Sequence[float]) -> list[FuzzySet]:
    """Create arbitrary crisp singleton positional OWA weights.

    This is a Python convenience extension. The original supplied R helper
    only generates equal singleton weights, but ``itvOWA_withNullSets`` itself
    accepts arbitrary interval/singleton weights.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional sequence")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if not np.isclose(np.sum(w), 1.0):
        raise ValueError("crisp OWA weights must sum to 1")
    return [FuzzySet(np.array([wi]), np.array([1.0])) for wi in w]


def itv_owa_with_null_sets(
    w: Sequence[IntervalCut],
    a: Sequence[IntervalCut],
) -> IntervalCut:
    """Faithful Python port of R ``itvOWA_withNullSets``.

    Aggregates interval-valued objects with interval-valued OWA weights using
    the alpha-level endpoint algorithm used in the Zhou/Mata implementation.
    """
    if len(w) != len(a):
        raise ValueError("w and a must have the same length")
    n = len(w)
    if n == 0:
        raise ValueError("at least one object is required")

    if any(cut.is_null for cut in a):
        return IntervalCut(np.nan, np.nan)

    wl = np.asarray([cut.itv_l for cut in w], dtype=float)
    wr = np.asarray([cut.itv_r for cut in w], dtype=float)
    al = np.asarray([cut.itv_l for cut in a], dtype=float)
    ar = np.asarray([cut.itv_r for cut in a], dtype=float)

    if np.any(np.isnan(wl)) or np.any(np.isnan(wr)):
        return IntervalCut(np.nan, np.nan)

    # Sort the lower and upper input endpoints independently in decreasing
    # order, exactly as in the supplied R implementation.
    al = np.sort(al)[::-1]
    ar = np.sort(ar)[::-1]

    if n == 1:
        # Direct one-object specialization of the same normalized formula.
        if wr[0] == 0 or wl[0] == 0:
            return IntervalCut(np.nan, np.nan)
        return IntervalCut(float(al[0]), float(ar[0]))

    # ---- Right endpoint rhoR ----
    tmp_h = wr[0] + np.sum(wl[1:])
    if tmp_h == 0:
        return IntervalCut(np.nan, np.nan)
    rho_r = (wr[0] * ar[0] + np.dot(wl[1:], ar[1:])) / tmp_h

    k = 1  # R k=2, converted to 0-based indexing
    while k < n and rho_r < ar[k]:
        k += 1
        tmp_h = np.sum(wr[:k]) + np.sum(wl[k:])
        if tmp_h == 0:
            return IntervalCut(np.nan, np.nan)
        rho_r = (
            np.dot(wr[:k], ar[:k]) + np.dot(wl[k:], ar[k:])
        ) / tmp_h

    # ---- Left endpoint rhoL ----
    tmp_h = wl[0] + np.sum(wr[1:])
    if tmp_h == 0:
        return IntervalCut(np.nan, np.nan)
    rho_l = (wl[0] * al[0] + np.dot(wr[1:], al[1:])) / tmp_h

    k = 1
    while k < n and rho_l < al[k]:
        k += 1
        tmp_h = np.sum(wl[:k]) + np.sum(wr[k:])
        if tmp_h == 0:
            return IntervalCut(np.nan, np.nan)
        rho_l = (
            np.dot(wl[:k], al[:k]) + np.dot(wr[k:], al[k:])
        ) / tmp_h

    return IntervalCut(float(rho_l), float(rho_r))


def mem_deg_by_alpha_cuts_with_null_check(
    domain_x: Sequence[float],
    alpha_cuts: Sequence[IntervalCut],
) -> FuzzySet:
    """Port of R ``memDegByAlphaCuts_withNullCheck``."""
    domain = np.asarray(domain_x, dtype=float)
    mf_value = np.zeros(domain.size, dtype=float)

    for k, x in enumerate(domain):
        containing_alphas = [0.0]  # matches R vector("numeric",1)
        for cut in alpha_cuts:
            if not cut.is_null and cut.itv_l <= x <= cut.itv_r:
                containing_alphas.append(float(cut.alpha or 0.0))
        mf_value[k] = max(containing_alphas)

    return FuzzySet(domain, mf_value)


def aggregate_type1_owa(
    fuzzy_sets: Sequence[FuzzySet],
    weight_sets: Sequence[FuzzySet],
    domain_x: Sequence[float],
    alpha_levels: Sequence[float],
    *,
    reproduce_r_alpha0: bool = True,
) -> tuple[FuzzySet, list[IntervalCut]]:
    """High-level alpha-level Type-1 OWA aggregation.

    Parameters mirror the supplied conference example. When
    ``reproduce_r_alpha0=True`` the alpha=0 branch deliberately assigns every
    input cut the full ``domain_x`` range, exactly as the R example does.
    """
    if len(fuzzy_sets) != len(weight_sets):
        raise ValueError("fuzzy_sets and weight_sets must have the same length")

    domain = np.asarray(domain_x, dtype=float)
    alphas = np.asarray(alpha_levels, dtype=float)
    alpha_cut_agg: list[IntervalCut] = []

    for alpha in alphas:
        alpha_cut_a: list[IntervalCut] = []
        alpha_cut_w: list[IntervalCut] = []

        for fs, ws in zip(fuzzy_sets, weight_sets):
            if reproduce_r_alpha0 and np.isclose(alpha, 0.0):
                alpha_cut_a.append(
                    IntervalCut(float(np.min(domain)), float(np.max(domain)), float(alpha))
                )
                alpha_cut_w.append(
                    IntervalCut(
                        float(np.min(ws.domain)),
                        float(np.max(ws.domain)),
                        float(alpha),
                    )
                )
            else:
                alpha_cut_a.append(alpha_cuts_general_mf(fs, float(alpha)))
                alpha_cut_w.append(alpha_cuts_general_mf(ws, float(alpha)))

        agg_cut = itv_owa_with_null_sets(alpha_cut_w, alpha_cut_a)
        agg_cut.alpha = float(alpha)
        alpha_cut_agg.append(agg_cut)

    agg_result = mem_deg_by_alpha_cuts_with_null_check(domain, alpha_cut_agg)
    return agg_result, alpha_cut_agg


def aggregate_triangular_t1owa(
    triangular_params: Sequence[Sequence[float]],
    *,
    domain_x: Sequence[float] | None = None,
    alpha_levels: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    reproduce_r_alpha0: bool = True,
) -> tuple[FuzzySet, list[IntervalCut], float]:
    """Convenience wrapper for triangular input fuzzy sets.

    Returns ``(aggregated_fuzzy_set, aggregated_alpha_cuts, centroid)``.
    """
    if domain_x is None:
        domain_x = np.linspace(0.0, 1.0, 100)
    if alpha_levels is None:
        alpha_levels = np.linspace(0.0, 1.0, 50)

    domain = np.asarray(domain_x, dtype=float)
    fuzzy_sets = [FuzzySet(domain, trimf2(domain, p)) for p in triangular_params]

    if weights is None:
        weight_sets = fuzzy_set_weights(len(fuzzy_sets))
    else:
        if len(weights) != len(fuzzy_sets):
            raise ValueError("weights length must equal number of fuzzy sets")
        weight_sets = crisp_owa_weights(weights)

    agg, cuts = aggregate_type1_owa(
        fuzzy_sets,
        weight_sets,
        domain,
        alpha_levels,
        reproduce_r_alpha0=reproduce_r_alpha0,
    )
    return agg, cuts, centroid(agg.domain, agg.mf_value)


# ---------------------------------------------------------------------------
# R-name aliases for direct correspondence with the supplied implementation.
# ---------------------------------------------------------------------------
alphaCutsGeneralMF = alpha_cuts_general_mf
fuzzySetWeights = fuzzy_set_weights
itvOWA_withNullSets = itv_owa_with_null_sets
memDegByAlphaCuts_withNullCheck = mem_deg_by_alpha_cuts_with_null_check
