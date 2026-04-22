"""
Proxy FastAPI — rezolva CORS pentru Monte Carlo Stocks
Deployabil gratuit pe Render.com
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import os

app = FastAPI()

# Permite orice origine — sau pune domeniul tau exact pentru securitate
# ex: ["https://tonesmihai-art.github.io"]
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Domenii permise — nu lasa proxy-ul deschis la orice URL
WHITELIST = [
    "financialmodelingprep.com",
    "finnhub.io",
    "query1.finance.yahoo.com",
    "query2.finance.yahoo.com",
    "finance.yahoo.com",
    "data.sec.gov",
    "www.sec.gov",
    "api.nasdaq.com",
    "query.data.world",
    "cdn.cboe.com",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _is_allowed(url: str) -> bool:
    return any(domain in url for domain in WHITELIST)


@app.get("/proxy")
async def proxy(url: str = Query(..., description="URL-ul de fetched")):
    if not _is_allowed(url):
        raise HTTPException(status_code=403, detail=f"Domeniu nepermis: {url}")

    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers=HEADERS,
        ) as client:
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
