"""
Plateau detection with adaptive gap threshold.

A 'plateau' is a maximal run of indices where consecutive gaps
g_i = lambda_i - lambda_{i+1} are small compared to the typical gap.

Threshold tau = median(gaps) + alpha * MAD(gaps), with absolute floor.
Adapts to both flat (Pre-LN+skip) and spread (Post-LN) spectra.
"""
from __future__ import annotations
import numpy as np


def detect_plateaus(lambdas_sorted, alpha=3.0, min_width=2, abs_floor=1e-6):
    n = len(lambdas_sorted)
    if n < 2:
        return [], np.array([]), 0.0
    gaps = np.clip(-np.diff(lambdas_sorted), 0, None)
    med = float(np.median(gaps))
    mad = float(np.median(np.abs(gaps - med))) + 1e-30
    tau = max(med + alpha * mad, abs_floor)

    plateaus = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and gaps[j] < tau:
            j += 1
        width = j - i + 1
        if width >= min_width:
            plateaus.append(dict(
                start=int(i), end=int(j), width=int(width),
                value=float(np.mean(lambdas_sorted[i:j + 1])),
            ))
        i = j + 1
    return plateaus, gaps, tau


def staircase_summary(lambdas_sorted, **kwargs):
    plats, gaps, tau = detect_plateaus(lambdas_sorted, **kwargs)
    n = len(lambdas_sorted)
    n_steps = int(np.sum(gaps >= tau)) if len(gaps) else 0
    return {
        "n_plateaus": len(plats),
        "n_step_gaps": n_steps,
        "plateau_widths": [p["width"] for p in plats],
        "plateau_values": [p["value"] for p in plats],
        "total_width": sum(p["width"] for p in plats),
        "fraction_in_plateaus": sum(p["width"] for p in plats) / max(1, n),
        "spread": float(lambdas_sorted.max() - lambdas_sorted.min()),
        "tau": float(tau),
    }
