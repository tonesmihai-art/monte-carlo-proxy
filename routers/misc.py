"""
Router: GET /health  +  GET /ping
Endpoints de monitorizare stare server + sesiune Yahoo.
"""

import httpx
from fastapi import APIRouter

from yahoo_client import _refresh_yahoo_session, _yahoo_session

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "crumb": _yahoo_session["crumb"] is not None}


@router.get("/ping")
async def ping():
    async with httpx.AsyncClient(follow_redirects=True) as client:
        ok = await _refresh_yahoo_session(client)
    return {
        "status":        "ok",
        "proxy":         "online",
        "yahoo_session": "activa" if ok else "esuat",
        "crumb":         _yahoo_session["crumb"] is not None,
    }
