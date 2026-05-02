"""
Router: GET /finnhub/{ticker}
Date fundamentale Finnhub — cheia API ramane pe server.
"""

import os
import asyncio

import httpx
from fastapi import APIRouter, HTTPException

from yahoo_client import _to_finnhub_ticker

router = APIRouter()


@router.get("/finnhub/{ticker}")
async def get_finnhub(ticker: str):
    """Date fundamentale Finnhub — cheia ramane pe server, nu in browser."""
    finnhub_key = os.environ.get("FINNHUB_KEY")
    if not finnhub_key:
        raise HTTPException(status_code=500, detail="FINNHUB_KEY lipsa pe server")

    fh_ticker = _to_finnhub_ticker(ticker)
    base      = "https://finnhub.io/api/v1"

    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            met_r, prof_r = await asyncio.gather(
                client.get(f"{base}/stock/metric?symbol={fh_ticker}&metric=all&token={finnhub_key}", timeout=9.0),
                client.get(f"{base}/stock/profile2?symbol={fh_ticker}&token={finnhub_key}", timeout=7.0),
                return_exceptions=True,
            )
            m = met_r.json().get("metric", {})  if not isinstance(met_r,  Exception) and met_r.status_code  == 200 else {}
            p = prof_r.json()                    if not isinstance(prof_r, Exception) and prof_r.status_code == 200 else {}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Finnhub eroare: {e}")

    if not m and not p:
        raise HTTPException(status_code=404, detail="Date Finnhub indisponibile")

    def _n(v):
        return v if v is not None and isinstance(v, (int, float)) else None

    eps           = _n(m.get("epsTTM"))          or _n(m.get("epsAnnual"))
    pe            = _n(m.get("peTTM"))            or _n(m.get("peAnnual"))
    fcf_per_share = _n(m.get("freeCashFlowPerShareTTM")) or _n(m.get("freeCashFlowPerShareAnnual"))
    growth        = next((v for v in [
                        _n(m.get("epsGrowth3Y")),
                        _n(m.get("epsGrowthTTMYoy")),
                        _n(m.get("revenueGrowth3Y")),
                        _n(m.get("revenueGrowthTTMYoy")),
                        _n(m.get("revenueGrowthQuarterlyYoy")),
                    ] if v is not None), None)
    shares       = _n(p.get("shareOutstanding"))
    total_assets = _n(m.get("totalAssets"))
    cash         = _n(m.get("cashAndEquivalents"))
    debt         = _n(m.get("totalDebt"))

    # Sector derivat din finnhubIndustry
    sector   = None
    industry = p.get("finnhubIndustry")
    if industry:
        ind = industry.lower()
        if any(k in ind for k in ["software", "semiconductor", "internet", "tech", "data",
                                   "cloud", "cyber", "artificial", "saas", "electronic compon"]):
            sector = "Technology"
        elif any(k in ind for k in ["telecom", "communication", "media", "broadcast", "wireless"]):
            sector = "Communication Services"
        elif "insurance" in ind:
            sector = "Insurance"
        elif any(k in ind for k in ["bank", "financial services", "asset management",
                                     "capital market", "credit service"]):
            sector = "Financial Services"
        elif any(k in ind for k in ["reit", "real estate"]):
            sector = "Real Estate"
        elif any(k in ind for k in ["oil", "gas", "energy", "petroleum", "coal", "pipeline", "lng"]):
            sector = "Energy"
        elif any(k in ind for k in ["utilit", "electric power", "water util", "renewable"]):
            sector = "Utilities"
        elif any(k in ind for k in ["drug", "pharma", "biotech", "medical", "hospital",
                                     "health plan", "diagnostics", "life science"]):
            sector = "Healthcare"
        elif any(k in ind for k in ["gold", "silver", "steel", "mining", "material",
                                     "chemical", "aluminum", "copper", "lithium"]):
            sector = "Basic Materials"
        elif any(k in ind for k in ["auto manufacturer", "automobile", "vehicle"]):
            sector = "Auto Manufacturers"
        elif any(k in ind for k in ["tobacco", "cigarette"]):
            sector = "Consumer Defensive"
        elif any(k in ind for k in ["food", "beverage", "grocery", "consumer staple", "household"]):
            sector = "Consumer Defensive"
        elif any(k in ind for k in ["shipping", "freight", "marine", "logistics", "courier",
                                     "aerospace", "defense", "industrial", "machinery", "equipment",
                                     "electrical", "manufacture", "construct"]):
            sector = "Industrials"

    return {
        "eps": eps, "pe": pe, "fcfPerShare": fcf_per_share,
        "growth": growth, "shares": shares,
        "totalAssets": total_assets, "cash": cash, "debt": debt,
        "sector": sector, "industry": industry,
    }
