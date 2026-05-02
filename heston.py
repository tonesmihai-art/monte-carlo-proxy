"""
Heston stochastic volatility model — math core.
Folosit de: routers/heston_routes.py, routers/iv.py (indirect prin _bs_iv)
"""

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
import time as _time_mod

# ── Cache calibrare (1 ora) ───────────────────────────
_heston_cache: dict = {}
HESTON_CACHE_TTL = 3600

# Grid fix de phi pentru integrare trapezoidala (96 puncte — echilibru viteza/precizie)
_PHI_VEC = np.linspace(1e-4, 150.0, 96)


def _heston_cf_vec(phi, T, r, v0, kappa, theta, xi, rho, logS):
    """Heston characteristic function (Albrecher 'little trap') vectorizata pe phi."""
    i   = 1j
    d   = np.sqrt((rho * xi * phi * i - kappa) ** 2 + xi ** 2 * (phi * i + phi ** 2))
    g   = (kappa - rho * xi * phi * i - d) / (kappa - rho * xi * phi * i + d)
    edt = np.exp(-d * T)
    # Evita log(0)
    denom = np.where(np.abs(1.0 - g) < 1e-12, 1e-12, 1.0 - g)
    numer = np.where(np.abs(1.0 - g * edt) < 1e-12, 1e-12, 1.0 - g * edt)
    C = r * phi * i * T + (kappa * theta / xi ** 2) * (
        (kappa - rho * xi * phi * i - d) * T - 2.0 * np.log(numer / denom)
    )
    D = (kappa - rho * xi * phi * i - d) / xi ** 2 * (1.0 - edt) / np.where(
        np.abs(1.0 - g * edt) < 1e-12, 1e-12, 1.0 - g * edt
    )
    return np.exp(C + D * v0 + i * phi * logS)


def _heston_call(S, K, T, r, v0, kappa, theta, xi, rho):
    """Pret call Heston via Lewis (2001) — integrare trapezoidala pe grila fixa."""
    if T <= 0:
        return max(float(S) - float(K), 0.0)
    try:
        logS      = np.log(float(S))
        phi_shift = _PHI_VEC - 0.5j
        cf        = _heston_cf_vec(phi_shift, float(T), float(r),
                                   float(v0), float(kappa), float(theta),
                                   float(xi), float(rho), logS)
        if not np.all(np.isfinite(cf)):
            return max(float(S) - float(K) * np.exp(-r * T), 0.0)
        integ = (np.exp(-1j * _PHI_VEC * np.log(float(K) / float(S))) * cf
                 / (_PHI_VEC ** 2 + 0.25)).real
        I    = np.trapz(integ, _PHI_VEC)
        call = float(S) - np.sqrt(float(S) * float(K)) * np.exp(-r * T) / np.pi * I
        lb   = max(float(S) - float(K) * np.exp(-r * T), 0.0)
        return max(float(call), lb)
    except Exception:
        return max(float(S) - float(K) * np.exp(-r * T), 0.0)


def _bs_iv(price, S, K, T, r):
    """Volatilitate implicita Black-Scholes via metoda Brent."""
    if T <= 0 or price <= 0:
        return None
    lb = max(float(S) - float(K) * np.exp(-r * T), 0.0)
    if price < lb - 1e-6:
        return None

    def bs_call(sigma):
        if sigma <= 0:
            return lb
        sq = sigma * np.sqrt(T)
        d1 = (np.log(float(S) / float(K)) + (r + 0.5 * sigma ** 2) * T) / sq
        d2 = d1 - sq
        return float(S) * norm.cdf(d1) - float(K) * np.exp(-r * T) * norm.cdf(d2)

    try:
        lo, hi = 1e-4, 10.0
        if bs_call(lo) - price > 0 or bs_call(hi) - price < 0:
            return None
        iv = brentq(lambda s: bs_call(s) - price, lo, hi, xtol=1e-5, maxiter=60)
        return float(iv) if 0.01 <= iv <= 8.0 else None
    except Exception:
        return None


def _calibration_loss(params, S, option_data, r):
    """RMSE ponderat IV model vs IV piata."""
    v0, kappa, theta, xi, rho = params
    if v0 <= 1e-5 or kappa <= 0 or theta <= 1e-5 or xi <= 0 or abs(rho) >= 0.999:
        return 1e6
    sse = 0.0
    wt  = 0.0
    for row in option_data:
        K, T, miv, w = row['K'], row['T'], row['iv'], row['w']
        try:
            mp   = _heston_call(S, K, T, r, v0, kappa, theta, xi, rho)
            iv_h = _bs_iv(mp, S, K, T, r)
            sse += w * (0.25 if iv_h is None else (iv_h - miv) ** 2)
        except Exception:
            sse += w * 0.25
        wt += w
    return float(np.sqrt(sse / max(wt, 1e-10)))
