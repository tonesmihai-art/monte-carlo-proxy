"""
Proxy FastAPI — Monte Carlo Stocks
Yahoo: foloseste yfinance (gestioneaza sesiunea intern, recunoscut de Yahoo)
Altele: httpx direct
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import yfinance as yf
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

WHITELIST = [
    "financialmodelingprep.com",
    "finnhub.io",
    "query1.finance.yahoo.com",
    "query2.finance.yahoo.com",
    "finance.yahoo.com",
    "data.sec.gov",
    "www.sec.gov",
    "api.nasdaq.com",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _is_allowed(url: str) -> bool:
    return any(d in url for d in WHITELIST)


def _is_yahoo(url: str) -> bool:
    return "yahoo.com" in url


def _extract_ticker(url: str) -> str | None:
    """Extrage ticker-ul din orice URL Yahoo Finance."""
    # /v8/finance/chart/NESN.SW
    # /v7/finance/options/NESN.SW
    # /v10/finance/quoteSummary/NESN.SW
    # /v11/finance/quoteSummary/NESN.SW
    m = re.search(r'/finance/(?:chart|options|quote(?:Summary)?)/([^/?&]+)', url)
    if m:
        return m.group(1)
    # ?symbols=NESN.SW  sau  ?symbol=NESN.SW
    m = re.search(r'[?&]symbols?=([^&]+)', url)
    if m:
        return m.group(1).split(',')[0]
    return None


@app.get("/proxy")
async def proxy(url: str = Query(...)):
    if not _is_allowed(url):
        raise HTTPException(status_code=403, detail="Domeniu nepermis")

    # ── Yahoo: folosim yfinance care gestioneaza auth intern ──
    if _is_yahoo(url):
        ticker_sym = _extract_ticker(url)

        # chart endpoint → date istorice
        if ticker_sym and '/chart/' in url:
            try:
                t    = yf.Ticker(ticker_sym)
                hist = t.history(period="1y", interval="1d", auto_adjust=False)
                if hist.empty:
                    raise HTTPException(status_code=404, detail="Date indisponibile")

                closes     = [round(float(v), 4) for v in hist['Close'].tolist()]
                volumes    = [int(v) for v in hist['Volume'].tolist()]
                timestamps = [int(ts.timestamp()) for ts in hist.index.to_pydatetime()]

                info = t.info or {}
                return JSONResponse({
                    "chart": {"result": [{
                        "meta": {
                            "symbol":                ticker_sym,
                            "currency":              info.get("currency", "USD"),
                            "longName":              info.get("longName", ticker_sym),
                            "shortName":             info.get("shortName", ticker_sym),
                            "sharesOutstanding":     info.get("sharesOutstanding"),
                            "epsTrailingTwelveMonths": info.get("trailingEps"),
                            "trailingPE":            info.get("trailingPE"),
                            "forwardPE":             info.get("forwardPE"),
                        },
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [{
                                "close":  closes,
                                "volume": volumes,
                            }]
                        }
                    }], "error": None}
                })
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # quoteSummary / quote endpoint → date fundamentale
        if ticker_sym:
            try:
                t    = yf.Ticker(ticker_sym)
                info = t.info or {}
                if not info:
                    raise HTTPException(status_code=404, detail="Date indisponibile")

                shares = info.get("sharesOutstanding")
                fcf    = info.get("freeCashflow")

                # totalAssets e in balance_sheet, nu in info
                total_assets = None
                try:
                    bs = t.balance_sheet
                    if bs is not None and not bs.empty:
                        for label in ["Total Assets", "TotalAssets"]:
                            if label in bs.index:
                                val = bs.loc[label].dropna()
                                if not val.empty:
                                    total_assets = float(val.iloc[0])
                                    break
                except Exception:
                    pass

                return JSONResponse({
                    "quoteSummary": {"result": [{
                        "financialData": {
                            "totalCash":      {"raw": info.get("totalCash")},
                            "totalDebt":      {"raw": info.get("totalDebt")},
                            "freeCashflow":   {"raw": fcf},
                            "earningsGrowth": {"raw": info.get("earningsGrowth")},
                            "revenueGrowth":  {"raw": info.get("revenueGrowth")},
                            "totalAssets":    {"raw": total_assets},
                        },
                        "defaultKeyStatistics": {
                            "sharesOutstanding": {"raw": shares},
                            "trailingEps":       {"raw": info.get("trailingEps")},
                            "forwardEps":        {"raw": info.get("forwardEps")},
                        },
                        "summaryDetail": {
                            "trailingPE": {"raw": info.get("trailingPE")},
                            "forwardPE":  {"raw": info.get("forwardPE")},
                        },
                    }], "error": None}
                })
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    # ── Non-Yahoo: httpx direct ───────────────────────
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=HEADERS) as client:
            r = await client.get(url)
            try:
                data = r.json()
            except Exception:
                data = r.text
            return JSONResponse(content=data, status_code=r.status_code)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
