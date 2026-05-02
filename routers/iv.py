"""
Router: GET /iv/{ticker}
IV real din optiuni Yahoo Finance (ATM + skew OTM).
"""

import math
import time
import httpx
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from yahoo_client import _yahoo_get

router = APIRouter()


@router.get("/iv/{ticker}")
async def get_iv(ticker: str, price: float = Query(0)):
    """IV real din optiuni Yahoo Finance — fara CORS, cu crumb."""
    async with httpx.AsyncClient(follow_redirects=True) as client:

        # ── Step 1: expiry dates ──────────────────────────
        url1  = f"https://query1.finance.yahoo.com/v7/finance/options/{ticker}"
        data1 = await _yahoo_get(client, url1)
        if not data1:
            raise HTTPException(status_code=404, detail="Options indisponibile")

        result = data1.get("optionChain", {}).get("result", [])
        if not result:
            raise HTTPException(status_code=404, detail="Nu exista optiuni")

        expiry_dates = result[0].get("expirationDates", [])
        now   = time.time()
        valid = [d for d in expiry_dates if d > now + 7 * 86400]
        if not valid:
            raise HTTPException(status_code=404, detail="Nu exista expirari valide")

        t30     = now + 30 * 86400
        nearest = min(valid, key=lambda d: abs(d - t30))
        days_to_exp = round((nearest - now) / 86400)

        # ── Step 2: options chain ─────────────────────────
        url2  = f"https://query1.finance.yahoo.com/v7/finance/options/{ticker}?date={int(nearest)}"
        data2 = await _yahoo_get(client, url2)
        if not data2:
            raise HTTPException(status_code=502, detail="Options chain indisponibil")

        opts_list = data2.get("optionChain", {}).get("result", [{}])[0].get("options", [])
        if not opts_list:
            raise HTTPException(status_code=404, detail="Options goale")

        opts  = opts_list[0]
        calls = [c for c in opts.get("calls", []) if 0.01 < c.get("impliedVolatility", 0) < 5]
        puts  = [p for p in opts.get("puts",  []) if 0.01 < p.get("impliedVolatility", 0) < 5]
        if not calls and not puts:
            raise HTTPException(status_code=404, detail="IV invalid in options")

        # ── ATM strike ────────────────────────────────────
        all_strikes = list(set([c["strike"] for c in calls] + [p["strike"] for p in puts]))
        current     = price if price > 0 else all_strikes[len(all_strikes) // 2]
        atm_strike  = min(all_strikes, key=lambda s: abs(s - current))

        atm_call = next((c for c in calls if c["strike"] == atm_strike), None)
        atm_put  = next((p for p in puts  if p["strike"] == atm_strike), None)
        ivs = [x["impliedVolatility"] for x in [atm_call, atm_put]
               if x and 0.01 < x.get("impliedVolatility", 0) < 5]
        if not ivs:
            raise HTTPException(status_code=404, detail="IV ATM negasit")

        iv_annual = sum(ivs) / len(ivs)
        iv_daily  = iv_annual / math.sqrt(252)

        # ── Skew: OTM put (~7% sub pret) vs OTM call (~7% peste) ──
        skew_data   = None
        put_target  = current * 0.93
        call_target = current * 1.07
        otm_puts    = [p for p in puts  if current * 0.70 < p["strike"] < current * 0.98]
        otm_calls   = [c for c in calls if current * 1.02 < c["strike"] < current * 1.30]
        if otm_puts and otm_calls:
            otm_put  = min(otm_puts,  key=lambda p: abs(p["strike"] - put_target))
            otm_call = min(otm_calls, key=lambda c: abs(c["strike"] - call_target))
            piv = otm_put.get("impliedVolatility",  0)
            civ = otm_call.get("impliedVolatility", 0)
            if piv > 0.01 and civ > 0.01:
                skew_data = {
                    "skew":       round(piv - civ, 4),
                    "putIV":      round(piv, 4),
                    "callIV":     round(civ, 4),
                    "putStrike":  otm_put["strike"],
                    "callStrike": otm_call["strike"],
                }

        return JSONResponse(content={
            "ticker":    ticker,
            "ivAnnual":  round(iv_annual, 4),
            "ivDaily":   round(iv_daily,  4),
            "atmStrike": atm_strike,
            "daysToExp": days_to_exp,
            "skewData":  skew_data,
        })
