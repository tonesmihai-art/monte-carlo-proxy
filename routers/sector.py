"""
Router: GET /sector/{ticker}
Returneaza sector si industrie via Yahoo assetProfile + fallback yfinance.
"""

import asyncio
import httpx
from fastapi import APIRouter, HTTPException

from yahoo_client import _yahoo_get

router = APIRouter()


@router.get("/sector/{ticker}")
async def get_sector(ticker: str):
    sector   = None
    industry = None

    # Incearca 1: Yahoo assetProfile cu crumb (rapid)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        url  = (
            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            f"?modules=assetProfile&formatted=false"
        )
        data = await _yahoo_get(client, url)
        try:
            profile  = data["quoteSummary"]["result"][0]["assetProfile"]
            sector   = profile.get("sector")
            industry = profile.get("industry")
        except Exception:
            pass

    # Incearca 2: yfinance (functioneaza EU server-side, fara CORS)
    if not sector:
        try:
            import yfinance as yf
            loop = asyncio.get_event_loop()

            def _get_info():
                t = yf.Ticker(ticker)
                return t.info or {}

            info     = await loop.run_in_executor(None, _get_info)
            sector   = info.get("sector")
            industry = info.get("industry")
        except Exception as e:
            print(f"[sector/yfinance] {ticker}: {e}")

    if not sector:
        raise HTTPException(status_code=404, detail="Sector necunoscut")

    return {"ticker": ticker, "sector": sector, "industry": industry or sector}
