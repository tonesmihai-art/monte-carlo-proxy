"""
Proxy FastAPI — Monte Carlo Stocks
Yahoo: yfinance (gestioneaza auth intern)
Altele: httpx direct
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import traceback

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


def _extract_ticker(url: str) -> str:
    import re
    m = re.search(r'/finance/(?:chart|options|quote(?:Summary)?)/([^/?&]+)', url)
    if m:
        return m.group(1)
    m = re.search(r'[?&]symbols?=([^&]+)', url)
    if m:
        return m.group(1).split(',')[0]
    return ""


def _yf_ticker_data(ticker_sym: str) -> dict:
    """Extrage date din yfinance — complet izolat, nu crapa niciodata."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker_sym)

        # info — cel mai important
        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        # balance_sheet — optional
        total_assets = None
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                for label in ["Total Assets", "TotalAssets", "totalAssets"]:
                    if label in bs.index:
                        val = bs.loc[label].dropna()
                        if not val.empty:
                            total_assets = float(val.iloc[0])
                            break
        except Exception:
            pass

        return {"info": info, "total_assets": total_assets}
    except Exception as e:
        print(f"[yfinance] Eroare pentru {ticker_sym}: {e}")
        return {"info": {}, "total_assets": None}


@app.get("/proxy")
async def proxy(url: str = Query(...)):
    if not _is_allowed(url):
        raise HTTPException(status_code=403, detail="Domeniu nepermis")

    # ── Yahoo: yfinance ───────────────────────────────
    if _is_yahoo(url):
        ticker_sym = _extract_ticker(url)
        if not ticker_sym:
            raise HTTPException(status_code=400, detail="Ticker negasit in URL")

        # Ruleaza yfinance intr-un thread separat (e sincron)
        import asyncio
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _yf_ticker_data, ticker_sym)

        info        = data["info"]
        total_assets = data["total_assets"]

        # chart endpoint → date istorice
        if "/chart/" in url:
            try:
                import yfinance as yf
                t    = yf.Ticker(ticker_sym)
                hist = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: t.history(period="1y", interval="1d", auto_adjust=False)
                )
                if hist.empty:
                    raise HTTPException(status_code=404, detail="Date istorice indisponibile")

                closes     = [round(float(v), 4) for v in hist["Close"].tolist()]
                volumes    = [int(v) for v in hist["Volume"].tolist()]
                timestamps = [int(ts.timestamp()) for ts in hist.index.to_pydatetime()]

                return JSONResponse({
                    "chart": {"result": [{
                        "meta": {
                            "symbol":   ticker_sym,
                            "currency": info.get("currency", "USD"),
                            "longName":  info.get("longName",  ticker_sym),
                            "shortName": info.get("shortName", ticker_sym),
                            "sharesOutstanding":       info.get("sharesOutstanding"),
                            "epsTrailingTwelveMonths": info.get("trailingEps"),
                            "trailingPE": info.get("trailingPE"),
                            "forwardPE":  info.get("forwardPE"),
                        },
                        "timestamp": timestamps,
                        "indicators": {"quote": [{"close": closes, "volume": volumes}]}
                    }], "error": None}
                })
            except HTTPException:
                raise
            except Exception as e:
                print(traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e))

        # quoteSummary / quote → date fundamentale
        shares  = info.get("sharesOutstanding")
        fcf     = info.get("freeCashflow")
        fcfps   = (fcf / shares) if (fcf and shares and shares > 0) else None

        return JSONResponse({
            "quoteSummary": {"result": [{
                "financialData": {
                    "totalCash":    {"raw": info.get("totalCash")},
                    "totalDebt":    {"raw": info.get("totalDebt")},
                    "freeCashflow": {"raw": fcf},
                    "totalAssets":  {"raw": total_assets},
                    "earningsGrowth": {"raw": info.get("earningsGrowth")},
                    "revenueGrowth":  {"raw": info.get("revenueGrowth")},
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

    # ── Non-Yahoo: httpx direct ───────────────────────
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=HEADERS) as client:
            r = await client.get(url)
            try:
                return JSONResponse(content=r.json(), status_code=r.status_code)
            except Exception:
                return JSONResponse(content=r.text, status_code=r.status_code)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/test/{ticker}")
async def test_ticker(ticker: str):
    """Endpoint de debug — testeaza yfinance direct."""
    data = _yf_ticker_data(ticker)
    return {"ticker": ticker, "fields": list(data["info"].keys())[:20], "total_assets": data["total_assets"]}
