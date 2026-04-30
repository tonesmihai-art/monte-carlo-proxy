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
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
import time as _time_mod

app = FastAPI()

# ── Heston calibration cache (1 ora) ──────────────────
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
    m = re.search(r'/finance/(?:chart|options|quote(?:Summary)?|timeseries)/([^/?&]+)', url)
    if m:
        return m.group(1)
    # fundamentals-timeseries: /ws/fundamentals-timeseries/v1/finance/timeseries/COV.PA
    m = re.search(r'/timeseries/([^/?&]+)', url)
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

            # ── Fundamentals Timeseries endpoint ──────────
            if "fundamentals-timeseries" in url:
                data = await _yahoo_get(client, url)
                if data and data.get("timeseries"):
                    return JSONResponse(content=data)
                raise HTTPException(status_code=502, detail="Timeseries indisponibil")

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


                # 🔥 EXTRA: verifica si financialData (uneori exista acolo)
                if not total_assets:
                    try:
                        fd = data["quoteSummary"]["result"][0].get("financialData", {})
                        raw = fd.get("totalAssets", {})
                        if isinstance(raw, dict):
                            total_assets = raw.get("raw")
                        elif isinstance(raw, (int, float)):
                            total_assets = raw
                    except Exception:
                        pass

                # 🔥 FALLBACK yfinance (ACUM pentru EU + US)
                if not total_assets:
                    loop = asyncio.get_event_loop()
                    yf_data = await loop.run_in_executor(None, _yf_ticker_data, ticker_sym)
                    if yf_data.get("total_assets"):
                        total_assets = yf_data["total_assets"]

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

            # Fallback yfinance — EU + US (yfinance merge server-side fara CORS)
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
                        "trailingPE":   {"raw": info.get("trailingPE")},
                        "forwardPE":    {"raw": info.get("forwardPE")},
                        "dividendRate": {"raw": info.get("dividendRate")},
                        "dividendYield":{"raw": info.get("dividendYield")},
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


# ── Sector via yfinance (functioneaza EU + US server-side) ─
@app.get("/sector/{ticker}")
async def get_sector(ticker: str):
    sector   = None
    industry = None

    # Incearca 1: Yahoo assetProfile cu crumb (rapid)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        url  = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=assetProfile&formatted=false"
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


@app.get("/health")
async def health():
    return {"status": "ok", "crumb": _yahoo_session["crumb"] is not None}

@app.get("/gemini-models")
async def list_gemini_models():
    """Debug: listeaza modelele Gemini disponibile pentru acest API key."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or not _GEMINI_OK:
        raise HTTPException(status_code=503, detail="Gemini indisponibil")
    try:
        client_g = google_genai.Client(api_key=api_key, http_options={"api_version": "v1"})
        models = [m.name for m in client_g.models.list()]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Helper: fetch date REIT din surse financiare ──────

EXCHANGE_MAP = {
    ".AS": "AMS", ".DE": "XETRA", ".L": "LSE", ".PA": "EPA",
    ".MI": "BIT", ".SW": "SWX", ".BR": "EBR", ".LS": "ELI",
    ".MC": "BME", ".HE": "HEL", ".ST": "STO", ".CO": "CPH",
    ".OL": "OSL", ".VI": "VIE",
}

def _to_finnhub_ticker(ticker: str) -> str:
    for suffix, exchange in EXCHANGE_MAP.items():
        if ticker.endswith(suffix):
            return f"{exchange}:{ticker[:-len(suffix)]}"
    return ticker

async def _fetch_reit_live_data(ticker: str) -> dict:
    """
    Cauta date operationale REIT (occupancy, LTV) din:
    1. Finnhub basicFinancials (metric)
    2. Yahoo Finance quoteSummary financialData
    Returneaza dict cu valorile gasite (poate fi gol daca nu gaseste nimic).
    """
    found = {}
    finnhub_key = os.environ.get("FINNHUB_KEY")
    fh_ticker   = _to_finnhub_ticker(ticker)

    async with httpx.AsyncClient(follow_redirects=True, timeout=8.0) as client:
        # ── 1. Finnhub metric ──────────────────────────────
        if finnhub_key:
            try:
                r = await client.get(
                    f"https://finnhub.io/api/v1/stock/metric?symbol={fh_ticker}&metric=all&token={finnhub_key}"
                )
                if r.status_code == 200:
                    m = r.json().get("metric", {})
                    # Câmpuri posibile pentru occupancy în Finnhub
                    for key in ["occupancyRate", "occupancy", "netOccupancy",
                                "physicalOccupancy", "economicOccupancy"]:
                        val = m.get(key)
                        if val is not None and isinstance(val, (int, float)) and 50 <= val <= 100:
                            found["occupancy"] = round(float(val), 1)
                            break
                    # LTV / debt-to-assets
                    if "occupancy" not in found:
                        ltv = m.get("longtermDebtTotalAssetRatio") or m.get("debtToTotalAssetRatio")
                        if ltv is not None and isinstance(ltv, (int, float)) and 0.05 < ltv < 1:
                            found["ltv_finnhub"] = round(float(ltv) * 100, 1)
            except Exception as e:
                print(f"[REIT fetch] Finnhub eroare: {e}")

        # ── 2. Yahoo Finance financialData (fallback) ──────
        if "occupancy" not in found:
            try:
                yahoo_url = (
                    f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
                    f"?modules=financialData,defaultKeyStatistics"
                )
                if _yahoo_session["crumb"]:
                    yahoo_url += f"&crumb={_yahoo_session['crumb']}"
                r = await client.get(yahoo_url, headers=HEADERS,
                                     cookies=_yahoo_session.get("cookies", {}))
                if r.status_code == 200:
                    fd = (r.json()
                          .get("quoteSummary", {})
                          .get("result", [{}])[0]
                          .get("financialData", {}))
                    # Yahoo uneori include occupancyRate pentru REIT-uri US
                    occ = fd.get("occupancyRate", {})
                    if isinstance(occ, dict):
                        occ = occ.get("raw")
                    if occ is not None and isinstance(occ, (int, float)) and 50 <= occ <= 100:
                        found["occupancy"] = round(float(occ), 1)
            except Exception as e:
                print(f"[REIT fetch] Yahoo eroare: {e}")

    return found


# ── AI Validator — Claude Haiku + Gemini 2.5 Flash-Lite ──

class ValidateRequest(BaseModel):
    ticker: str
    sector: str = "tech"
    currency: str = "USD"
    currentPrice: float = 0
    fields: dict = {}
    estimatedFields: list = []
    provider: str = "claude"   # "claude" | "gemini"


@app.post("/validate-fundamentals")
async def validate_fundamentals(req: ValidateRequest):
    if req.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY lipsa pe server")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY lipsa")

    sym = "$" if req.currency == "USD" else f"{req.currency} "

    # Campuri cu valori
    field_lines = "\n".join(
        f"  {k}: {v}" for k, v in req.fields.items() if v is not None
    )
    # Campuri lipsa — AI sa furnizeze valori reale
    missing_fields = [k for k, v in req.fields.items() if v is None]
    if req.estimatedFields:
        missing_fields = list(set(missing_fields + req.estimatedFields))
    missing_lines = (
        "\nCampuri LIPSA (furnizeaza valori reale din cunostintele tale despre " + req.ticker + "):\n" +
        "\n".join(f"  {k}: LIPSA" for k in missing_fields)
    ) if missing_fields else ""

    # Limita FCF relativa la EPS — previne sugestii aberante
    eps_val = req.fields.get("eps")
    if eps_val and eps_val > 0:
        fcf_max = round(eps_val * 3, 2)
        fcf_min = round(eps_val * -2, 2)
        fcf_rule = f"- FCF/actiune: pentru aceasta companie (EPS={eps_val}), intervalul realist este [{fcf_min}, {fcf_max}]; orice valoare in afara acestui interval este o eroare de date, NU o sugera"
    else:
        fcf_rule = "- FCF/actiune: intre -10 si 20 pentru companii obisnuite; valori sub 0.05 sau peste 30 sunt aproape sigur erori Yahoo"

    is_reit = "reit" in req.sector.lower() or "imobiliar" in req.sector.lower()

    # ── Fetch live date REIT (occupancy, LTV) din Finnhub/Yahoo ──
    live_reit = {}
    if is_reit and req.fields.get("occupancy") is None:
        try:
            live_reit = await _fetch_reit_live_data(req.ticker)
        except Exception as e:
            print(f"[REIT live fetch] eroare: {e}")

    # Injecteaza valorile gasite ca date verificate (nu LIPSA)
    verified_lines = ""
    if live_reit.get("occupancy") is not None:
        occ_val = live_reit["occupancy"]
        verified_lines += f"\n  occupancy (verificat din surse financiare): {occ_val}%"
        missing_fields  = [f for f in missing_fields if f != "occupancy"]
        field_lines    += verified_lines
        print(f"[REIT live] {req.ticker}: occupancy={occ_val}%")

    reit_note = (
        f"\nReguli speciale REIT: "
        + (f"Rata de ocupare verificata din surse financiare este deja inclusa in date. "
           if live_reit.get("occupancy") else
           f"Daca 'occupancy' lipseste, sugereaza rata de ocupare reala a companiei {req.ticker} daca o cunosti, "
           f"sau o valoare tipica de sector (85-95%) daca nu cunosti valoarea exacta — e preferabil o estimare decat lipsa. ")
        + f"LTV si dividendul sunt indicatorii principali pentru REIT, nu FCF-ul."
    ) if is_reit else ""

    system_prompt = (
        f"Esti un analist financiar strict. "
        f"Analizezi EXCLUSIV compania cu ticker-ul {req.ticker}. "
        f"NU confunda aceasta companie cu alte companii cu nume similare, din acelasi sector sau din aceeasi tara. "
        f"Pentru valori financiare precise (EPS, FCF, active, datorii): sugereaza NUMAI daca esti sigur. "
        f"Pentru metrici structurale standard (rata ocupare REIT, LTV): poti estima din cunostinte de sector daca valoarea specifica lipseste."
    )

    prompt = f"""Verifica valorile fundamentale pentru {req.ticker} (sector: {req.sector}, pret curent: {sym}{req.currentPrice}).

Valori extrase automat:
{field_lines}{missing_lines}

Reguli de validare:
- EPS: intre -50 si 500 pentru actiuni normale
- P/E: intre 3 si 200; negativ = companie pe pierdere (acceptabil)
{fcf_rule}
- Crestere (%): intre -50 si 50; peste 100 e aproape sigur o eroare Yahoo
- WACC: intre 5 si 20
- Active/Cash/Datorii (milioane): verifica ordinul de marime pentru companie
- LTV (REIT): intre 10 si 70
- Ocupare (REIT): intre 50 si 100
- Dividend: yield implicit (dividend/pret) intre 0 si 20%{reit_note}

IMPORTANT: Pentru valori financiare precise (EPS, FCF, active) furnizeaza NUMAI daca esti sigur pentru {req.ticker}. Pentru metrici structurale (ocupare, LTV) poti estima din sector daca valoarea lipseste.

Raspunde DOAR cu JSON valid, fara text suplimentar, fara markdown:
{{
  "valid": true|false,
  "corrections": {{<doar campurile gresite sau LIPSA cu valoarea corecta; omite campurile corecte si pe cele nesigure>}},
  "issues": [<string, doar daca exista probleme reale>],
  "verdict": "<max 15 cuvinte>"
}}"""

    import json, re

    def _clean_json(raw: str) -> str:
        """Curata markdown/text din jurul JSON — Gemini poate adauga text extra."""
        raw = raw.strip()
        # extrage primul bloc JSON daca exista backtick-uri
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        else:
            # curata backtick-urile simple
            raw = raw.replace("```json", "").replace("```", "").strip()
            # fallback: extrage primul { ... } din raspuns
            m = re.search(r"(\{.*\})", raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()
        # Python booleans/None → JSON
        raw = re.sub(r'\bTrue\b', 'true', raw)
        raw = re.sub(r'\bFalse\b', 'false', raw)
        raw = re.sub(r'\bNone\b', 'null', raw)
        # trailing commas inainte de } sau ] (JSON invalid, Gemini le genereaza)
        raw = re.sub(r',\s*([}\]])', r'\1', raw)
        return raw

    try:
        if req.provider == "gemini":
            if not _GEMINI_OK:
                raise HTTPException(status_code=500, detail="google-genai package nu e instalat pe server")
            client_g = google_genai.Client(api_key=api_key, http_options={"api_version": "v1"})
            gemini_models = ["gemini-2.5-flash", "gemini-2.0-flash-lite"]
            raw = None
            last_err = None
            for gm in gemini_models:
                try:
                    resp = client_g.models.generate_content(
                        model=gm,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            max_output_tokens=800,
                            temperature=0.1,
                        ),
                    )
                    # resp.text poate arunca daca raspunsul e blocat
                    try:
                        raw = resp.text.strip()
                    except Exception:
                        parts = (resp.candidates or [{}])[0].get("content", {}).get("parts", [])
                        raw = parts[0].get("text", "").strip() if parts else None
                    if raw:
                        print(f"[Gemini] model={gm} OK, {len(raw)} chars")
                        break
                except Exception as e:
                    last_err = e
                    print(f"[Gemini] model={gm} eroare: {e}")
                    continue
            if not raw:
                raise Exception(f"Gemini indisponibil (toate modelele au esuat): {last_err}")
        else:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()

        data = json.loads(_clean_json(raw))
        # adauga provider in raspuns — frontend il foloseste pentru label
        data["_provider"] = req.provider
        return JSONResponse(content=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error ({req.provider}): {e}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[validate-fundamentals/{req.provider}] EROARE: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/iv/{ticker}")
async def get_iv(ticker: str, price: float = Query(0)):
    """IV real din optiuni Yahoo Finance — fara CORS, cu crumb."""
    import time, math
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
            piv = otm_put.get("impliedVolatility", 0)
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


@app.get("/finnhub/{ticker}")
async def get_finnhub(ticker: str):
    """Date fundamentale Finnhub — cheia ramane pe server, nu in browser."""
    finnhub_key = os.environ.get("FINNHUB_KEY")
    if not finnhub_key:
        raise HTTPException(status_code=500, detail="FINNHUB_KEY lipsa pe server")

    # Convertor ticker Yahoo → Finnhub (pentru actiuni europene)
    exchange_map = {
        ".AS": "AMS", ".DE": "XETRA", ".L": "LSE", ".PA": "EPA",
        ".MI": "BIT", ".SW": "SWX", ".BR": "EBR", ".LS": "ELI",
        ".MC": "BME", ".HE": "HEL", ".ST": "STO", ".CO": "CPH",
        ".OL": "OSL", ".VI": "VIE",
    }
    fh_ticker = ticker
    for suffix, exchange in exchange_map.items():
        if ticker.endswith(suffix):
            fh_ticker = f"{exchange}:{ticker[:-len(suffix)]}"
            break

    base = "https://finnhub.io/api/v1"
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

    eps          = _n(m.get("epsTTM"))          or _n(m.get("epsAnnual"))
    pe           = _n(m.get("peTTM"))           or _n(m.get("peAnnual"))
    fcf_per_share= _n(m.get("freeCashFlowPerShareTTM")) or _n(m.get("freeCashFlowPerShareAnnual"))
    growth       = next((v for v in [
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

    # Sector din Finnhub profile2
    sector   = None
    industry = p.get("finnhubIndustry")
    if industry:
        ind = industry.lower()
        if any(k in ind for k in ["software","semiconductor","internet","tech","data","cloud","cyber","artificial","saas","electronic compon"]):
            sector = "Technology"
        elif any(k in ind for k in ["telecom","communication","media","broadcast","wireless"]):
            sector = "Communication Services"
        elif "insurance" in ind:
            sector = "Insurance"
        elif any(k in ind for k in ["bank","financial services","asset management","capital market","credit service"]):
            sector = "Financial Services"
        elif any(k in ind for k in ["reit","real estate"]):
            sector = "Real Estate"
        elif any(k in ind for k in ["oil","gas","energy","petroleum","coal","pipeline","lng"]):
            sector = "Energy"
        elif any(k in ind for k in ["utilit","electric power","water util","renewable"]):
            sector = "Utilities"
        elif any(k in ind for k in ["drug","pharma","biotech","medical","hospital","health plan","diagnostics","life science"]):
            sector = "Healthcare"
        elif any(k in ind for k in ["gold","silver","steel","mining","material","chemical","aluminum","copper","lithium"]):
            sector = "Basic Materials"
        elif any(k in ind for k in ["auto manufacturer","automobile","vehicle"]):
            sector = "Auto Manufacturers"
        elif any(k in ind for k in ["tobacco","cigarette"]):
            sector = "Consumer Defensive"
        elif any(k in ind for k in ["food","beverage","grocery","consumer staple","household"]):
            sector = "Consumer Defensive"
        elif any(k in ind for k in ["shipping","freight","marine","logistics","courier","aerospace","defense","industrial","machinery","equipment","electrical","manufacture","construct"]):
            sector = "Industrials"

    return {
        "eps": eps, "pe": pe, "fcfPerShare": fcf_per_share,
        "growth": growth, "shares": shares,
        "totalAssets": total_assets, "cash": cash, "debt": debt,
        "sector": sector, "industry": industry,
    }


@app.get("/heston-calibrate/{ticker}")
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

        # Inlocuieste S cu pretul curent din Yahoo daca nu a fost furnizat
        if S <= 0:
            S = float(res_chain[0].get("quote", {}).get("regularMarketPrice", 0) or 0)
        if S <= 0:
            raise HTTPException(status_code=400, detail="Pret indisponibil")

        expiry_dates = res_chain[0].get("expirationDates", [])
        now_ts       = _time_mod.time()

        # Selectam pana la 5 expirari: ~30, 60, 90, 180, 365 zile
        valid   = [d for d in expiry_dates if d > now_ts + 14 * 86400]
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
        atm_ivs    = [row["iv"] for row in option_data if abs(row["K"] / S - 1.0) < 0.06]
        atm_iv     = float(np.median(atm_ivs)) if atm_ivs else 0.25
        v0_init    = atm_iv ** 2
        bounds     = [(1e-4, 2.0), (0.1, 15.0), (1e-4, 2.0), (0.01, 5.0), (-0.99, 0.99)]

        def _run():
            best = None
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
