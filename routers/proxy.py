"""
Router: GET /proxy
Proxy general pentru Yahoo Finance (cu crumb), non-Yahoo (whitelist).
"""

import asyncio
import httpx
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from yahoo_client import (
    _is_allowed, _is_yahoo, _extract_ticker,
    _yahoo_get, _yf_ticker_data, HEADERS,
)

router = APIRouter()


@router.get("/proxy")
async def proxy(url: str = Query(...)):
    if not _is_allowed(url):
        raise HTTPException(status_code=403, detail="Domeniu nepermis")

    async with httpx.AsyncClient(follow_redirects=True) as client:

        if _is_yahoo(url):
            ticker_sym = _extract_ticker(url)
            if not ticker_sym:
                raise HTTPException(status_code=400, detail="Ticker negasit in URL")

            is_eu = "." in ticker_sym   # TTE.PA, NESN.SW etc.

            # ── Fundamentals Timeseries endpoint ──────────
            if "fundamentals-timeseries" in url:
                data = await _yahoo_get(client, url)
                if data and data.get("timeseries"):
                    return JSONResponse(content=data)
                raise HTTPException(status_code=502, detail="Timeseries indisponibil")

            # ── Chart endpoint ────────────────────────────
            if "/chart/" in url:
                chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_sym}?interval=1d&range=1y"
                data = await _yahoo_get(client, chart_url)
                if data and data.get("chart", {}).get("result"):
                    return JSONResponse(content=data)

                # Fallback yfinance (US)
                if not is_eu:
                    import yfinance as yf
                    loop    = asyncio.get_event_loop()
                    yf_data = await loop.run_in_executor(None, _yf_ticker_data, ticker_sym)
                    info    = yf_data["info"]
                    t       = yf.Ticker(ticker_sym)
                    hist    = await loop.run_in_executor(
                        None, lambda: t.history(period="1y", interval="1d", auto_adjust=False)
                    )
                    if not hist.empty:
                        return JSONResponse({"chart": {"result": [{
                            "meta": {
                                "symbol":                  ticker_sym,
                                "currency":                info.get("currency", "USD"),
                                "longName":                info.get("longName", ticker_sym),
                                "shortName":               info.get("shortName", ticker_sym),
                                "sharesOutstanding":       info.get("sharesOutstanding"),
                                "epsTrailingTwelveMonths": info.get("trailingEps"),
                                "trailingPE":              info.get("trailingPE"),
                            },
                            "timestamp": [int(ts.timestamp()) for ts in hist.index.to_pydatetime()],
                            "indicators": {"quote": [{
                                "close":  [round(float(v), 4) for v in hist["Close"].tolist()],
                                "volume": [int(v) for v in hist["Volume"].tolist()],
                            }]},
                        }], "error": None}})

                raise HTTPException(status_code=502, detail="Date istorice indisponibile")

            # ── quoteSummary / quote endpoint ─────────────
            modules = "financialData,defaultKeyStatistics,summaryDetail"
            qs_url  = (
                f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker_sym}"
                f"?modules={modules}&formatted=false"
            )
            data = await _yahoo_get(client, qs_url)

            if data and data.get("quoteSummary", {}).get("result"):
                # Extrage totalAssets si totalLiabilities din balance sheet separat
                total_assets      = None
                total_liabilities = None
                bs_url  = (
                    f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker_sym}"
                    f"?modules=balanceSheetHistory&formatted=false"
                )
                bs_data = await _yahoo_get(client, bs_url)
                try:
                    stmts = bs_data["quoteSummary"]["result"][0]["balanceSheetHistory"]["balanceSheetStatements"]
                    stmt0 = stmts[0]

                    raw = stmt0.get("totalAssets", {})
                    total_assets = raw.get("raw") if isinstance(raw, dict) else (
                        raw if isinstance(raw, (int, float)) else None
                    )

                    for liab_key in ("totalLiabilitiesNetMinorityInterest", "totalLiab"):
                        raw_l = stmt0.get(liab_key, {})
                        val_l = raw_l.get("raw") if isinstance(raw_l, dict) else (
                            raw_l if isinstance(raw_l, (int, float)) else None
                        )
                        if val_l is not None:
                            total_liabilities = val_l
                            break
                except Exception:
                    pass

                # Verifica si financialData (uneori exista acolo)
                if not total_assets:
                    try:
                        fd  = data["quoteSummary"]["result"][0].get("financialData", {})
                        raw = fd.get("totalAssets", {})
                        if isinstance(raw, dict):
                            total_assets = raw.get("raw")
                        elif isinstance(raw, (int, float)):
                            total_assets = raw
                    except Exception:
                        pass

                # Fallback yfinance
                if not total_assets:
                    loop    = asyncio.get_event_loop()
                    yf_data = await loop.run_in_executor(None, _yf_ticker_data, ticker_sym)
                    if yf_data.get("total_assets"):
                        total_assets = yf_data["total_assets"]

                # Injecteaza totalAssets + totalLiabilities in financialData
                try:
                    result0 = data["quoteSummary"]["result"][0]
                    fd = result0.setdefault("financialData", {})
                    if total_assets:
                        fd["totalAssets"] = {"raw": total_assets}
                    if total_liabilities:
                        fd["totalLiabilities"] = {"raw": total_liabilities}

                    def _wrap(d):
                        for k, v in d.items():
                            if isinstance(v, (int, float)):
                                d[k] = {"raw": v}
                        return d
                    _wrap(fd)
                    _wrap(result0.get("defaultKeyStatistics", {}))
                    _wrap(result0.get("summaryDetail", {}))
                except Exception:
                    pass

                return JSONResponse(content=data)

            # Fallback yfinance — EU + US
            loop    = asyncio.get_event_loop()
            yf_data = await loop.run_in_executor(None, _yf_ticker_data, ticker_sym)
            info    = yf_data["info"]
            ta      = yf_data["total_assets"]
            tl      = yf_data.get("total_liabilities")
            if info:
                shares = info.get("sharesOutstanding")
                fcf    = info.get("freeCashflow")
                return JSONResponse({"quoteSummary": {"result": [{
                    "financialData": {
                        "totalCash":        {"raw": info.get("totalCash")},
                        "totalDebt":        {"raw": info.get("totalDebt")},
                        "freeCashflow":     {"raw": fcf},
                        "totalAssets":      {"raw": ta},
                        "totalLiabilities": {"raw": tl},
                        "earningsGrowth":   {"raw": info.get("earningsGrowth")},
                        "revenueGrowth":    {"raw": info.get("revenueGrowth")},
                    },
                    "defaultKeyStatistics": {
                        "sharesOutstanding": {"raw": shares},
                        "trailingEps":       {"raw": info.get("trailingEps")},
                    },
                    "summaryDetail": {
                        "trailingPE":    {"raw": info.get("trailingPE")},
                        "forwardPE":     {"raw": info.get("forwardPE")},
                        "dividendRate":  {"raw": info.get("dividendRate")},
                        "dividendYield": {"raw": info.get("dividendYield")},
                    },
                }], "error": None}})

            raise HTTPException(status_code=502, detail="Date fundamentale indisponibile")

        # ── Non-Yahoo ─────────────────────────────────────
        try:
            r = await client.get(url, headers=HEADERS, timeout=15.0)
            try:
                return JSONResponse(content=r.json(), status_code=r.status_code)
            except Exception:
                return JSONResponse(content=r.text, status_code=r.status_code)
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
