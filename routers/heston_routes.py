"""
Router: GET /heston-calibrate/{ticker}
Calibrare model Heston pe suprafata IV reala din optiuni Yahoo Finance.
"""

import asyncio
import time as _time_mod

import httpx
import numpy as np
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from scipy.optimize import minimize

from heston import _heston_cache, HESTON_CACHE_TTL, _calibration_loss
from yahoo_client import _yahoo_get

router = APIRouter()


@router.get("/heston-calibrate/{ticker}")
async def heston_calibrate(ticker: str, price: float = Query(0)):
    """
    Calibreaza modelul Heston pe suprafata IV reala din optiuni Yahoo Finance.
    Returneaza: v0, kappa, theta, xi, rho + RMSE + statistici.
    Cache 1 ora.
    """
    cache_key = ticker.upper()
    cached    = _heston_cache.get(cache_key)
    if cached and (_time_mod.time() - cached["ts"] < HESTON_CACHE_TTL):
        return JSONResponse(content=cached["data"])

    S = float(price)
    r = 0.04  # rata risk-free aproximativa

    async with httpx.AsyncClient(follow_redirects=True) as client:
        # ── Step 1: expiry dates ──────────────────────────
        url1  = f"https://query1.finance.yahoo.com/v7/finance/options/{ticker}"
        data1 = await _yahoo_get(client, url1)
        if not data1:
            raise HTTPException(status_code=404, detail="Optiuni indisponibile")

        res_chain = data1.get("optionChain", {}).get("result", [])
        if not res_chain:
            raise HTTPException(status_code=404, detail="Nu exista optiuni")

        if S <= 0:
            S = float(res_chain[0].get("quote", {}).get("regularMarketPrice", 0) or 0)
        if S <= 0:
            raise HTTPException(status_code=400, detail="Pret indisponibil")

        expiry_dates = res_chain[0].get("expirationDates", [])
        now_ts       = _time_mod.time()

        # Selectam pana la 5 expirari: ~30, 60, 90, 180, 365 zile
        valid    = [d for d in expiry_dates if d > now_ts + 14 * 86400]
        if len(valid) < 2:
            raise HTTPException(status_code=404, detail="Expirari insuficiente")

        targets  = [30, 60, 90, 180, 365]
        selected = []
        used     = set()
        for t_days in targets:
            t_target = now_ts + t_days * 86400
            best     = min(valid, key=lambda d: abs(d - t_target))
            if best not in used:
                selected.append(best)
                used.add(best)
            if len(selected) >= 5:
                break

        # ── Step 2: fetch optiuni pentru fiecare expirare ──
        async def _fetch_chain(exp_ts):
            url  = f"https://query1.finance.yahoo.com/v7/finance/options/{ticker}?date={int(exp_ts)}"
            data = await _yahoo_get(client, url)
            if not data:
                return []
            opts_list = data.get("optionChain", {}).get("result", [{}])[0].get("options", [])
            if not opts_list:
                return []
            opts0 = opts_list[0]
            T     = (exp_ts - now_ts) / (365.25 * 86400)
            rows  = []
            for side in ("calls", "puts"):
                for c in opts0.get(side, []):
                    K   = float(c.get("strike", 0) or 0)
                    iv  = float(c.get("impliedVolatility", 0) or 0)
                    vol = int(c.get("volume") or 0)
                    oi  = int(c.get("openInterest") or 0)
                    if K <= 0 or iv <= 0.01 or iv > 4.0:
                        continue
                    mon = K / S
                    if not (0.72 < mon < 1.28):
                        continue
                    dist  = abs(mon - 1.0)
                    w_mon = float(np.exp(-dist * 6))
                    w_liq = min(1.0, (vol + oi) / 500) if (vol + oi) > 0 else 0.3
                    rows.append({"K": K, "T": T, "iv": iv, "w": w_mon * w_liq + 0.05})
            return rows

        chain_results = await asyncio.gather(
            *[_fetch_chain(e) for e in selected], return_exceptions=True
        )
        option_data = []
        for cr in chain_results:
            if isinstance(cr, list):
                option_data.extend(cr)

        if len(option_data) < 5:
            raise HTTPException(status_code=404, detail=f"Prea putine puncte IV: {len(option_data)}")

        # ── Step 3: calibrare in thread executor ──────────
        atm_ivs = [row["iv"] for row in option_data if abs(row["K"] / S - 1.0) < 0.06]
        atm_iv  = float(np.median(atm_ivs)) if atm_ivs else 0.25
        v0_init = atm_iv ** 2
        bounds  = [(1e-4, 2.0), (0.1, 15.0), (1e-4, 2.0), (0.01, 5.0), (-0.99, 0.99)]

        def _run():
            best   = None
            starts = [
                [v0_init, 2.0, v0_init, 0.5, -0.7],
                [v0_init, 4.0, max(v0_init * 0.8, 0.01), 0.8, -0.5],
                [min(v0_init * 1.5, 1.5), 1.5, v0_init, 1.2, -0.4],
            ]
            for x0 in starts:
                try:
                    res = minimize(
                        _calibration_loss, x0,
                        args=(S, option_data, r),
                        method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 400, "ftol": 1e-9, "gtol": 1e-7},
                    )
                    if best is None or res.fun < best.fun:
                        best = res
                except Exception:
                    pass
            return best

        loop  = asyncio.get_event_loop()
        calib = await loop.run_in_executor(None, _run)

        if calib is None:
            raise HTTPException(status_code=500, detail="Calibrare esuat")

        v0, kappa, theta, xi, rho = calib.x
        result_data = {
            "ticker":      ticker,
            "v0":          round(float(v0),    6),
            "kappa":       round(float(kappa), 4),
            "theta":       round(float(theta), 6),
            "xi":          round(float(xi),    4),
            "rho":         round(float(rho),   4),
            "rmse":        round(float(calib.fun), 4),
            "nPoints":     len(option_data),
            "nExpiries":   len(selected),
            "convergence": bool(calib.success),
            "r":           r,
        }
        _heston_cache[cache_key] = {"ts": _time_mod.time(), "data": result_data}
        return JSONResponse(content=result_data)
