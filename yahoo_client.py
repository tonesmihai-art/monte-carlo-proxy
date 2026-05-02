"""
Yahoo Finance client — sesiune cu crumb, helpers comune.
Folosit de: routers/proxy.py, routers/sector.py, routers/validate.py,
            routers/iv.py, routers/heston_routes.py, routers/misc.py
"""

import re
import httpx

# ── Whitelist domenii permise ─────────────────────────
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

# ── Map sufixe Yahoo → coduri bursă Finnhub (unificat) ───
EXCHANGE_MAP = {
    ".AS": "AMS", ".DE": "XETRA", ".L": "LSE", ".PA": "EPA",
    ".MI": "BIT", ".SW": "SWX", ".BR": "EBR", ".LS": "ELI",
    ".MC": "BME", ".HE": "HEL", ".ST": "STO", ".CO": "CPH",
    ".OL": "OSL", ".VI": "VIE",
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
        total_assets      = None
        total_liabilities = None
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                for label in ["Total Assets", "TotalAssets"]:
                    if label in bs.index:
                        val = bs.loc[label].dropna()
                        if not val.empty:
                            total_assets = float(val.iloc[0])
                            break
                for label in [
                    "Total Liabilities Net Minority Interest",
                    "Total Liabilities",
                    "TotalLiabilitiesNetMinorityInterest",
                    "TotalLiab",
                ]:
                    if label in bs.index:
                        val = bs.loc[label].dropna()
                        if not val.empty:
                            total_liabilities = float(val.iloc[0])
                            break
        except Exception:
            pass
        return {"info": info, "total_assets": total_assets, "total_liabilities": total_liabilities}
    except Exception as e:
        print(f"[yfinance] {ticker_sym}: {e}")
        return {"info": {}, "total_assets": None, "total_liabilities": None}


def _to_finnhub_ticker(ticker: str) -> str:
    """Converteste ticker Yahoo (ex: TTE.PA) → format Finnhub (ex: EPA:TTE)."""
    for suffix, exchange in EXCHANGE_MAP.items():
        if ticker.endswith(suffix):
            return f"{exchange}:{ticker[:-len(suffix)]}"
    return ticker

