"""
Proxy FastAPI — rezolva CORS + Yahoo crumb auth pentru Monte Carlo Stocks
Deployabil gratuit pe Render.com
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

app = FastAPI()

ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

_yahoo_crumb = None
_yahoo_cookies = {}


async def _get_yahoo_crumb(client: httpx.AsyncClient):
    global _yahoo_crumb, _yahoo_cookies
    if _yahoo_crumb:
        return _yahoo_crumb
    try:
        r1 = await client.get(
            "https://finance.yahoo.com/",
            headers={**HEADERS, "Accept": "text/html,application/xhtml+xml,*/*"},
            follow_redirects=True,
            timeout=10.0,
        )
        _yahoo_cookies = dict(r1.cookies)
        r2 = await client.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=HEADERS,
            cookies=_yahoo_cookies,
            timeout=8.0,
        )
        crumb = r2.text.strip()
        if crumb and len(crumb) > 3 and "<" not in crumb:
            _yahoo_crumb = crumb
            return _yahoo_crumb
    except Exception:
        pass
    return None


def _is_allowed(url: str) -> bool:
    return any(domain in url for domain in WHITELIST)


def _is_yahoo(url: str) -> bool:
    return "yahoo.com" in url


@app.get("/proxy")
async def proxy(url: str = Query(...)):
    if not _is_allowed(url):
        raise HTTPException(status_code=403, detail="Domeniu nepermis")

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=HEADERS) as client:
        fetch_url = url
        cookies = {}

        if _is_yahoo(url):
            crumb = await _get_yahoo_crumb(client)
            if crumb:
                sep = "&" if "?" in url else "?"
                fetch_url = f"{url}{sep}crumb={crumb}"
                cookies = _yahoo_cookies

        try:
            r = await client.get(fetch_url, cookies=cookies, headers=HEADERS)

            if r.status_code == 401 and _is_yahoo(url):
                global _yahoo_crumb
                _yahoo_crumb = None
                crumb = await _get_yahoo_crumb(client)
                if crumb:
                    sep = "&" if "?" in url else "?"
                    fetch_url = f"{url}{sep}crumb={crumb}"
                    r = await client.get(fetch_url, cookies=_yahoo_cookies, headers=HEADERS)

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
    return {"status": "ok", "crumb_cached": _yahoo_crumb is not None}
