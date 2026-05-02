"""
Router: POST /validate-fundamentals
AI validator (Claude Haiku / Gemini Flash) pentru date fundamentale.
Include helper _fetch_reit_live_data pentru REIT-uri.
"""

import os
import json
import re
import traceback

import httpx
import anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False

from yahoo_client import _yahoo_session, HEADERS, _to_finnhub_ticker

router = APIRouter()


# ── Helper: fetch date REIT din surse financiare ─────

async def _fetch_reit_live_data(ticker: str) -> dict:
    """
    Cauta date operationale REIT (occupancy, LTV) din:
    1. Finnhub basicFinancials (metric)
    2. Yahoo Finance quoteSummary financialData
    Returneaza dict cu valorile gasite (poate fi gol daca nu gaseste nimic).
    """
    found       = {}
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
                    for key in ["occupancyRate", "occupancy", "netOccupancy",
                                "physicalOccupancy", "economicOccupancy"]:
                        val = m.get(key)
                        if val is not None and isinstance(val, (int, float)) and 50 <= val <= 100:
                            found["occupancy"] = round(float(val), 1)
                            break
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
                    occ = fd.get("occupancyRate", {})
                    if isinstance(occ, dict):
                        occ = occ.get("raw")
                    if occ is not None and isinstance(occ, (int, float)) and 50 <= occ <= 100:
                        found["occupancy"] = round(float(occ), 1)
            except Exception as e:
                print(f"[REIT fetch] Yahoo eroare: {e}")

    return found


# ── Helper: curata JSON din raspunsul AI ─────────────

def _clean_json(raw: str) -> str:
    """Curata markdown/text din jurul JSON — Gemini poate adauga text extra."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    else:
        raw = raw.replace("```json", "").replace("```", "").strip()
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    raw = re.sub(r'\bTrue\b',  'true',  raw)
    raw = re.sub(r'\bFalse\b', 'false', raw)
    raw = re.sub(r'\bNone\b',  'null',  raw)
    # trailing commas inainte de } sau ] (JSON invalid, Gemini le genereaza)
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    # virgule lipsa intre proprietati
    raw = re.sub(r'([}\]"])\s*\n(\s*")', r'\1,\n\2', raw)
    return raw


# ── Schema request ────────────────────────────────────

class ValidateRequest(BaseModel):
    ticker: str
    sector: str = "tech"
    currency: str = "USD"
    currentPrice: float = 0
    fields: dict = {}
    estimatedFields: list = []
    provider: str = "claude"   # "claude" | "gemini"


# ── Endpoint ──────────────────────────────────────────

@router.post("/validate-fundamentals")
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

    field_lines   = "\n".join(f"  {k}: {v}" for k, v in req.fields.items() if v is not None)
    missing_fields = [k for k, v in req.fields.items() if v is None]
    if req.estimatedFields:
        missing_fields = list(set(missing_fields + req.estimatedFields))
    missing_lines = (
        "\nCampuri LIPSA (furnizeaza valori reale din cunostintele tale despre " + req.ticker + "):\n" +
        "\n".join(f"  {k}: LIPSA" for k in missing_fields)
    ) if missing_fields else ""

    eps_val = req.fields.get("eps")
    if eps_val and eps_val > 0:
        fcf_max  = round(eps_val * 3, 2)
        fcf_min  = round(eps_val * -2, 2)
        fcf_rule = (
            f"- FCF/actiune: pentru aceasta companie (EPS={eps_val}), intervalul realist este "
            f"[{fcf_min}, {fcf_max}]; orice valoare in afara acestui interval este o eroare de date, NU o sugera"
        )
    else:
        fcf_rule = "- FCF/actiune: intre -10 si 20 pentru companii obisnuite; valori sub 0.05 sau peste 30 sunt aproape sigur erori Yahoo"

    is_reit = "reit" in req.sector.lower() or "imobiliar" in req.sector.lower()

    # ── Fetch live date REIT ──────────────────────────
    live_reit = {}
    if is_reit and req.fields.get("occupancy") is None:
        try:
            live_reit = await _fetch_reit_live_data(req.ticker)
        except Exception as e:
            print(f"[REIT live fetch] eroare: {e}")

    verified_lines = ""
    if live_reit.get("occupancy") is not None:
        occ_val         = live_reit["occupancy"]
        verified_lines += f"\n  occupancy (verificat din surse financiare): {occ_val}%"
        missing_fields  = [f for f in missing_fields if f != "occupancy"]
        field_lines    += verified_lines
        print(f"[REIT live] {req.ticker}: occupancy={occ_val}%")

    reit_note = (
        f"\nReguli speciale REIT: "
        + (
            f"Rata de ocupare verificata din surse financiare este deja inclusa in date. "
            if live_reit.get("occupancy") else
            f"Daca 'occupancy' lipseste, sugereaza rata de ocupare reala a companiei {req.ticker} daca o cunosti, "
            f"sau o valoare tipica de sector (85-95%) daca nu cunosti valoarea exacta — e preferabil o estimare decat lipsa. "
        )
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

    try:
        if req.provider == "gemini":
            if not _GEMINI_OK:
                raise HTTPException(status_code=500, detail="google-genai package nu e instalat pe server")
            client_g      = google_genai.Client(api_key=api_key, http_options={"api_version": "v1"})
            gemini_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
            full_prompt   = f"{system_prompt}\n\n{prompt}"
            raw           = None
            last_err      = None
            for gm in gemini_models:
                try:
                    resp = client_g.models.generate_content(
                        model=gm,
                        contents=full_prompt,
                        config=genai_types.GenerateContentConfig(
                            max_output_tokens=800,
                            temperature=0.1,
                        ),
                    )
                    try:
                        raw = resp.text.strip()
                    except Exception:
                        parts = (resp.candidates or [{}])[0].get("content", {}).get("parts", [])
                        raw   = parts[0].get("text", "").strip() if parts else None
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
            claude  = anthropic.Anthropic(api_key=api_key)
            message = claude.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()

        data = json.loads(_clean_json(raw))
        data["_provider"] = req.provider
        return JSONResponse(content=data)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error ({req.provider}): {e}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[validate-fundamentals/{req.provider}] EROARE: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
