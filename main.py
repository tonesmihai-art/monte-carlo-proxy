"""
Proxy FastAPI — Monte Carlo Stocks
Yahoo EU: sesiune httpx cu crumb (yfinance nu merge pentru EU)
Yahoo US: yfinance
Altele: httpx direct
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import asyncio
import re
import traceback
import os
import anthropic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

WHITELIST = [
    "financialmodelingprep.com", "finnhub.io",
    "query1.finance.yahoo.com", "query2.finance.yahoo.com",
    "finance.yahoo.com", "data.sec.gov", "www.sec.gov", "api.nasdaq.com",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
}

# ── Sesiune Yahoo cu crumb (singleton) ───────────────
_yahoo_session: dict = {"crumb": None, "cookies": {}}


async def _refresh_yahoo_session(client: httpx.AsyncClient) -> bool:
    try:
        r1 = await client.get(
            "https://finance.yahoo.com/",
            headers={**HEADERS, "Accept": "text/html,*/*"},
            follow_redirects=True, timeout=12.0,
        )
        cookies = dict(r1.cookies)
        r2 = await client.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=HEADERS, cookies=cookies, timeout=8.0,
        )
        crumb = r2.text.strip()
        if crumb and len(crumb) > 3 and "<" not in crumb:
            _yahoo_session["crumb"]   = crumb
            _yahoo_session["cookies"] = cookies
            return True
    except Exception as e:
        print(f"[crumb] refresh esuat: {e}")
    return False


async def _yahoo_get(client: httpx.AsyncClient, url: str) -> dict | None:
    """Fetch Yahoo cu crumb — reincearca o data daca 401."""
    for attempt in range(2):
        if not _yahoo_session["crumb"]:
            ok = await _refresh_yahoo_session(client)
            if not ok:
                return None
        sep      = "&" if "?" in url else "?"
        full_url = f"{url}{sep}crumb={_yahoo_session['crumb']}"
        try:
            r = await client.get(full_url, cookies=_yahoo_session["cookies"],
                                 headers=HEADERS, timeout=12.0)
            if r.status_code == 401:
                _yahoo_session["crumb"] = None   # forteaza refresh
                continue
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"[yahoo_get] {e}")
            return None
    return None


def _is_allowed(url: str) -> bool:
    return any(d in url for d in WHITELIST)


def _is_yahoo(url: str) -> bool:
    return "yahoo.com" in url


def _extract_ticker(url: str) -> str:
    m = re.search(r'/finance/(?:chart|options|quote(?:Summary)?)/([^/?&]+)', url)
    if m:
        return m.group(1)
    m = re.search(r'[?&]symbols?=([^&]+)', url)
    if m:
        return m.group(1).split(',')[0]
    return ""


def _yf_ticker_data(ticker_sym: str) -> dict:
    """yfinance — functioneaza pentru US, poate esua pentru EU."""
    try:
        import yfinance as yf
        t    = yf.Ticker(ticker_sym)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass
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
        return {"info": info, "total_assets": total_assets}
    except Exception as e:
        print(f"[yfinance] {ticker_sym}: {e}")
        return {"info": {}, "total_assets": None}


@app.get("/proxy")
async def proxy(url: str = Query(...)):
    if not _is_allowed(url):
        raise HTTPException(status_code=403, detail="Domeniu nepermis")

    async with httpx.AsyncClient(follow_redirects=True) as client:

        if _is_yahoo(url):
            ticker_sym = _extract_ticker(url)
            if not ticker_sym:
                raise HTTPException(status_code=400, detail="Ticker negasit in URL")

            is_eu = "." in ticker_sym   # TTE.PA, NESN.SW etc.

            # ── Chart endpoint ────────────────────────────
            if "/chart/" in url:
                # Incearca direct cu crumb
                chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_sym}?interval=1d&range=1y"
                data = await _yahoo_get(client, chart_url)
                if data and data.get("chart", {}).get("result"):
                    return JSONResponse(content=data)

                # Fallback yfinance (US)
                if not is_eu:
                    loop = asyncio.get_event_loop()
                    yf_data = await loop.run_in_executor(None, _yf_ticker_data, ticker_sym)
                    info = yf_data["info"]
                    import yfinance as yf
                    t    = yf.Ticker(ticker_sym)
                    hist = await loop.run_in_executor(
                        None, lambda: t.history(period="1y", interval="1d", auto_adjust=False)
                    )
                    if not hist.empty:
                        return JSONResponse({"chart": {"result": [{
                            "meta": {
                                "symbol": ticker_sym,
                                "currency": info.get("currency", "USD"),
                                "longName": info.get("longName", ticker_sym),
                                "shortName": info.get("shortName", ticker_sym),
                                "sharesOutstanding": info.get("sharesOutstanding"),
                                "epsTrailingTwelveMonths": info.get("trailingEps"),
                                "trailingPE": info.get("trailingPE"),
                            },
                            "timestamp": [int(ts.timestamp()) for ts in hist.index.to_pydatetime()],
                            "indicators": {"quote": [{
                                "close":  [round(float(v), 4) for v in hist["Close"].tolist()],
                                "volume": [int(v) for v in hist["Volume"].tolist()],
                            }]}
                        }], "error": None}})

                raise HTTPException(status_code=502, detail="Date istorice indisponibile")

            # ── quoteSummary / quote endpoint ─────────────
            modules = "financialData,defaultKeyStatistics,summaryDetail"
            qs_url  = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker_sym}?modules={modules}&formatted=false"
            data    = await _yahoo_get(client, qs_url)

            if data and data.get("quoteSummary", {}).get("result"):
                # Extrage totalAssets din balance sheet separat
                total_assets = None
                bs_url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker_sym}?modules=balanceSheetHistory&formatted=false"
                bs_data = await _yahoo_get(client, bs_url)
                try:
                    stmts = bs_data["quoteSummary"]["result"][0]["balanceSheetHistory"]["balanceSheetStatements"]
                    raw = stmts[0].get("totalAssets", {})
                    total_assets = raw.get("raw") if isinstance(raw, dict) else (raw if isinstance(raw, (int, float)) else None)
                except Exception:
                    pass

                # Injecteaza totalAssets in financialData
                try:
                    result0 = data["quoteSummary"]["result"][0]
                    fd = result0.setdefault("financialData", {})
                    if total_assets:
                        fd["totalAssets"] = {"raw": total_assets}

                    # Normalizeaza toate campurile numerice sa aiba format {raw: value}
                    # (unele versiuni Yahoo returneaza plain numbers, altele {raw,fmt})
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

            # Fallback yfinance pentru US
            if not is_eu:
                loop    = asyncio.get_event_loop()
                yf_data = await loop.run_in_executor(None, _yf_ticker_data, ticker_sym)
                info    = yf_data["info"]
                ta      = yf_data["total_assets"]
                if info:
                    shares = info.get("sharesOutstanding")
                    fcf    = info.get("freeCashflow")
                    return JSONResponse({"quoteSummary": {"result": [{
                        "financialData": {
                            "totalCash":    {"raw": info.get("totalCash")},
                            "totalDebt":    {"raw": info.get("totalDebt")},
                            "freeCashflow": {"raw": fcf},
                            "totalAssets":  {"raw": ta},
                            "earningsGrowth": {"raw": info.get("earningsGrowth")},
                            "revenueGrowth":  {"raw": info.get("revenueGrowth")},
                        },
                        "defaultKeyStatistics": {
                            "sharesOutstanding": {"raw": shares},
                            "trailingEps":       {"raw": info.get("trailingEps")},
                        },
                        "summaryDetail": {
                            "trailingPE": {"raw": info.get("trailingPE")},
                            "forwardPE":  {"raw": info.get("forwardPE")},
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


@app.get("/health")
async def health():
    return {"status": "ok", "crumb": _yahoo_session["crumb"] is not None}


# ── AI Validator — Anthropic Haiku ───────────────────

class ValidateRequest(BaseModel):
    ticker: str
    sector: str = "tech"
    currency: str = "USD"
    currentPrice: float = 0
    fields: dict = {}


@app.post("/validate-fundamentals")
async def validate_fundamentals(req: ValidateRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY lipsa")

    sym = "$" if req.currency == "USD" else f"{req.currency} "
    field_lines = "\n".join(
        f"  {k}: {v}" for k, v in req.fields.items() if v is not None
    )

    prompt = f"""Esti un analist financiar. Verifica daca valorile fundamentale extrase automat pentru {req.ticker} (sector: {req.sector}, pret curent: {sym}{req.currentPrice}) sunt realiste.

Valori extrase:
{field_lines}

Reguli de validare:
- EPS: intre -50 si 500 pentru actiuni normale
- P/E: intre 3 si 200; negativ = companie pe pierdere (acceptabil)
- FCF/actiune: intre -100 si 500; daca e negativ si compania e profitabila → suspect
- Crestere (%): intre -50 si 50; peste 100 e aproape sigur o eroare Yahoo
- WACC: intre 5 si 20
- Active/Cash/Datorii (milioane): verifica ordinul de marime pentru companie
- LTV (REIT): intre 10 si 70
- Ocupare (REIT): intre 50 si 100
- Dividend: yield implicit (dividend/pret) intre 0 si 20%

Raspunde DOAR cu JSON valid, fara text suplimentar, fara markdown:
{{
  "valid": true,
  "corrections": {{
    "eps": null,
    "pe": null,
    "fcf": null,
    "growth": null,
    "wacc": null,
    "assets": null,
    "cash": null,
    "debt": null,
    "ltv": null,
    "dividend": null
  }},
  "issues": [],
  "verdict": "Valorile sunt corecte"
}}
Pune null pentru campurile corecte. Corecteaza DOAR valorile evident gresite."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw  = message.content[0].text.strip()
        import json
        # curata eventuale backtick-uri
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return JSONResponse(content=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping():
     async with httpx.AsyncClient(follow_redirects=True) as client:
        ok = await _refresh_yahoo_session(client)
    return {
        "status": "ok",
        "proxy": "online",
        "yahoo_session": "activa" if ok else "esuat",
        "crumb": _yahoo_session["crumb"] is not None,
    }

@app.get("/test/{ticker}")
async def test_ticker(ticker: str):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        await _refresh_yahoo_session(client)
        modules = "financialData,defaultKeyStatistics,summaryDetail"
        url  = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={modules}&formatted=false"
        data = await _yahoo_get(client, url)
        return {"ticker": ticker, "crumb": _yahoo_session["crumb"], "result": data}
