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
        else:
            # Nu exista niciun JSON in raspuns — returnam string gol
            # ca sa fie clar la json.loads() ca e invalid
            return ""
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

    sym     = "$" if req.currency == "USD" else f"{req.currency} "
    is_reit = "reit" in req.sector.lower() or "imobiliar" in req.sector.lower()

    # Excludem 'occupancy' din fields/missing pentru sectoare non-REIT
    _fields_filtered = {
        k: v for k, v in req.fields.items()
        if not (k == "occupancy" and not is_reit)
    }
    field_lines   = "\n".join(f"  {k}: {v}" for k, v in _fields_filtered.items() if v is not None)
    missing_fields = [k for k, v in _fields_filtered.items() if v is None]
    if req.estimatedFields:
        missing_fields = list(set(missing_fields + [f for f in req.estimatedFields if f != "occupancy" or is_reit]))
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

    reit_rules = (
        "\n- ltv (REIT): intre 10 si 70"
        "\n- occupancy (REIT): intre 50 si 100"
    ) if is_reit else ""

    prompt = f"""Verifica valorile fundamentale pentru {req.ticker} (sector: {req.sector}, pret curent: {sym}{req.currentPrice}).

Valori extrase automat (cheile sunt cele exacte pe care TREBUIE sa le folosesti in 'corrections'):
{field_lines}{missing_lines}

Reguli de validare:
- eps: intre -50 si 500 pentru actiuni normale
- pe: intre 3 si 200; negativ = companie pe pierdere (acceptabil)
{fcf_rule}
- growth: intre -50 si 50; peste 100 e aproape sigur o eroare Yahoo
- wacc: intre 5 si 20
- assets, cash, totalLiabilities (milioane): verifica ordinul de marime pentru companie{reit_rules}
- dividend: yield implicit (dividend/pret) intre 0 si 20%{reit_note}

IMPORTANT:
- In campul 'corrections' foloseste EXACT aceste nume de chei (nu le traduce, nu le redenumi): eps, pe, fcf, growth, wacc, assets, cash, totalLiabilities, shares, ltv, occupancy, dividend
- Pentru valori financiare precise (eps, fcf, assets, totalLiabilities) furnizeaza NUMAI daca esti sigur pentru {req.ticker}.
- Pentru metrici structurale (occupancy, ltv) poti estima din sector daca valoarea lipseste.

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
                        raw_candidate = resp.text.strip()
                    except Exception:
                        parts = (resp.candidates or [{}])[0].get("content", {}).get("parts", [])
                        raw_candidate = parts[0].get("text", "").strip() if parts else None
                    if raw_candidate:
                        # Validam JSON-ul INAINTE sa acceptam raspunsul
                        # Daca e invalid, incercam urmatorul model
                        try:
                            json.loads(_clean_json(raw_candidate))
                            raw = raw_candidate
                            print(f"[Gemini] model={gm} OK, {len(raw)} chars")
                            break
                        except (json.JSONDecodeError, ValueError) as je:
                            last_err = je
                            print(f"[Gemini] model={gm} JSON invalid ({je}) — incerc urmatorul model")
                            continue
                    else:
                        print(f"[Gemini] model={gm} raspuns gol — incerc urmatorul model")
                except Exception as e:
                    last_err = e
                    print(f"[Gemini] model={gm} eroare API: {e}")
                    continue
            if not raw:
                raise HTTPException(
                    status_code=503,
                    detail=f"Gemini indisponibil — toate modelele au esuat. Ultima eroare: {last_err}"
                )
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

        # Filtru de siguranta: eliminam 'occupancy' din corrections daca nu e REIT
        if not is_reit and isinstance(data.get("corrections"), dict):
            data["corrections"].pop("occupancy", None)

        return JSONResponse(content=data)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error ({req.provider}): {e}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[validate-fundamentals/{req.provider}] EROARE: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Gemini Verdict — evaluare calitativă smart ────────────────────────────

class GeminiVerdictRequest(BaseModel):
    sims: list   # [{ticker, name, score, verdict, margin, ret, up, down, prob, vol, sent, dev, div, period}]


@router.post("/gemini-verdict")
async def gemini_verdict(req: GeminiVerdictRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY lipsa pe server")
    if not _GEMINI_OK:
        raise HTTPException(status_code=500, detail="google-genai package nu e instalat")
    if not req.sims or len(req.sims) < 2:
        raise HTTPException(status_code=400, detail="Minim 2 simulari necesare")

    def _fmt(v, decimals=1, pct=False):
        if v is None:
            return "—"
        sign = "+" if float(v) >= 0 else ""
        return f"{sign}{float(v):.{decimals}f}{'%' if pct else ''}"

    sims_text = ""
    for s in req.sims:
        ticker  = s.get("ticker", "?")
        name    = s.get("name", "")
        score   = s.get("score")
        verdict = s.get("verdict", "")
        period  = s.get("period", 30)

        sims_text += f"\n▸ {ticker}" + (f" ({name})" if name else "") + ":\n"
        if score is not None:
            sims_text += f"  Scor final: {score}/100  Semnal: {verdict}\n"
        if s.get("margin") is not None:
            m = float(s["margin"])
            label = "subevaluat semnificativ" if m > 30 else "subevaluat moderat" if m > 10 else "la valoare justa" if m > -5 else "supraevaluat"
            sims_text += f"  Marja siguranta vs DCF: {_fmt(m, pct=True)} ({label})\n"
        if s.get("ret") is not None:
            sims_text += f"  Randament P50 ({period}z): {_fmt(s['ret'], pct=True)}\n"
        if s.get("up") is not None and s.get("down") is not None:
            sims_text += f"  Asimetrie P90/P10: {_fmt(s['up'], pct=True)} / {_fmt(s['down'], pct=True)}\n"
        if s.get("prob") is not None:
            sims_text += f"  Probabilitate profit {period}z: {float(s['prob']):.1f}%\n"
        if s.get("vol") is not None:
            sims_text += f"  Volatilitate anualizata: {float(s['vol']):.1f}%/an\n"
        if s.get("div") is not None:
            sims_text += f"  Dividend yield: {float(s['div']):.2f}%\n"
        if s.get("sent") is not None:
            sims_text += f"  Sentiment: {_fmt(s['sent'], decimals=3)}\n"
        if s.get("dev") is not None:
            sims_text += f"  Deviatia fata de MA60: {_fmt(s['dev'], pct=True)}\n"

    prompt = f"""INSTRUCTIUNE CRITICA: Prima propozitie a raspunsului tau trebuie sa fie direct o concluzie despre actiuni. NU incepe cu "Ca analist", "In calitate de", "Am analizat", "Rezultatele arata" sau orice alta fraza introductiva. Incepe direct cu numele actiunii sau cu verdictul.

Date comparatie Monte Carlo (30.000 simulari), {len(req.sims)} actiuni:
{sims_text}

Scrie 4 paragrafe scurte in romana, fara titluri, fara liste:

Paragraful 1: Acord sau dezacord cu clasamentul matematic — argumentat cu numere din date (scor, marja, probabilitate).

Paragraful 2: Asimetria P90/P10 pe fiecare actiune — care are cel mai bun potential vs risc real? Integreaza volatilitatea.

Paragraful 3: Contradictii vizibile — unde marja de siguranta contrazice randamentul MC? Ce semnaleaza sentimentul negativ sau pozitiv?

Paragraful 4: O singura recomandare de actiune, directa, cu pragul de risc specific de urmarit.

Ton: direct, tehnic, propozitii scurte. Fara introduceri, fara clisee."""

    client_g = google_genai.Client(api_key=api_key, http_options={"api_version": "v1"})

    last_err = None
    for model in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        try:
            resp = client_g.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=1200,
                    temperature=0.4,
                ),
            )
            text = resp.text.strip() if resp.text else None
            if text:
                print(f"[gemini-verdict] model={model} OK, {len(text)} chars")
                return JSONResponse(content={"evaluare": text, "model": model})
            print(f"[gemini-verdict] model={model} raspuns gol")
        except Exception as e:
            last_err = e
            print(f"[gemini-verdict] model={model} eroare: {e}")
            continue

    raise HTTPException(status_code=503, detail=f"Gemini indisponibil: {last_err}")


# ── Gemini Verdict Comparație — evaluare calitativă smart ────────────────

class GeminiVerdictRequest(BaseModel):
    sims: list   # [{ticker, name, score, verdict, margin, ret, up, down, prob, vol, sent, dev, div, period}]


@router.post("/gemini-verdict")
async def gemini_verdict(req: GeminiVerdictRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY lipsa pe server")
    if not _GEMINI_OK:
        raise HTTPException(status_code=500, detail="google-genai package nu e instalat")
    if not req.sims or len(req.sims) < 2:
        raise HTTPException(status_code=400, detail="Minim 2 simulări necesare")

    def _fmt(v, decimals=1, pct=False):
        if v is None:
            return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.{decimals}f}{'%' if pct else ''}"

    sims_text = ""
    for s in req.sims:
        ticker  = s.get("ticker", "?")
        name    = s.get("name", "")
        score   = s.get("score")
        verdict = s.get("verdict", "")
        period  = s.get("period", 30)

        sims_text += f"\n▸ {ticker}" + (f" ({name})" if name else "") + ":\n"
        if score is not None:
            sims_text += f"  Scor final: {score}/100 {verdict}\n"
        if s.get("margin") is not None:
            m = s["margin"]
            sims_text += f"  Marjă siguranță vs. val. estimată DCF: {_fmt(m, pct=True)}"
            if m > 30:
                sims_text += " → subevaluat semnificativ"
            elif m > 10:
                sims_text += " → ușor subevaluat"
            elif m < -20:
                sims_text += " → supraevaluat"
            sims_text += "\n"
        if s.get("ret") is not None:
            sims_text += f"  Randament P50 ({period}z): {_fmt(s['ret'], pct=True)}\n"
        if s.get("up") is not None and s.get("down") is not None:
            sims_text += f"  Asimetrie (P90↑ / P10↓): {_fmt(s['up'], pct=True)} / {_fmt(s['down'], pct=True)}\n"
        if s.get("prob") is not None:
            sims_text += f"  Probabilitate profit {period}z: {s['prob']:.1f}%\n"
        if s.get("vol") is not None:
            sims_text += f"  Volatilitate anualizată: {s['vol']:.1f}%/an\n"
        if s.get("div") is not None:
            sims_text += f"  Dividend yield: {s['div']:.2f}%\n"
        if s.get("sent") is not None:
            sims_text += f"  Sentiment: {_fmt(s['sent'], decimals=3)}\n"
        if s.get("dev") is not None:
            sims_text += f"  Deviație față de MA60: {_fmt(s['dev'], pct=True)}\n"

    prompt = f"""Ești un analist financiar senior cu experiență în piețe europene și globale.

Ai rezultatele unei comparații Monte Carlo (30.000 simulări) între {len(req.sims)} acțiuni:
{sims_text}

Oferă o evaluare calitativă independentă. Concentrează-te pe:
1. Dacă câștigătorul matematic e justificat sau există contradicții între metrici (ex: marjă mare dar randament mic, sau invers)
2. Raportul risc/randament real — compară volatilitatea cu potențialul și asimetria P90/P10
3. Cel mai important risc ascuns sau oportunitate neevidentă în cifre
4. Verdictul tău: ești de acord cu clasamentul sau nu, și de ce — exprimă-te direct

Ton: direct, concis, ca un analist real. Fără fraze introductive. Maxim 160 de cuvinte.
Răspunde exclusiv în română. Text curgător, fără liste sau titluri."""

    client_g = google_genai.Client(api_key=api_key, http_options={"api_version": "v1"})

    last_err = None
    for model in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        try:
            resp = client_g.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=400,
                    temperature=0.35,
                ),
            )
            text = resp.text.strip() if resp.text else None
            if text:
                print(f"[gemini-verdict] model={model} OK, {len(text)} chars")
                return JSONResponse(content={"evaluare": text, "model": model})
            print(f"[gemini-verdict] model={model} raspuns gol")
        except Exception as e:
            last_err = e
            print(f"[gemini-verdict] model={model} eroare: {e}")
            continue

    raise HTTPException(status_code=503, detail=f"Gemini indisponibil: {last_err}")


# ── Gemini Verdict — evaluare calitativa smart ────────────────────────────

class GeminiVerdictRequest(BaseModel):
    sims: list   # [{ticker, name, score, verdict, margin, ret, up, down, prob, vol, sent, dev, div, period}]


@router.post("/gemini-verdict")
async def gemini_verdict(req: GeminiVerdictRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY lipsa pe server")
    if not _GEMINI_OK:
        raise HTTPException(status_code=500, detail="google-genai package nu e instalat")
    if not req.sims or len(req.sims) < 2:
        raise HTTPException(status_code=400, detail="Minim 2 simulari necesare")

    def _fmt(v, decimals=1, pct=False):
        if v is None:
            return "—"
        sign = "+" if float(v) >= 0 else ""
        return f"{sign}{float(v):.{decimals}f}{'%' if pct else ''}"

    sims_text = ""
    for s in req.sims:
        ticker  = s.get("ticker", "?")
        name    = s.get("name", "")
        score   = s.get("score")
        verdict = s.get("verdict", "")
        period  = s.get("period", 30)

        sims_text += f"\n▸ {ticker}" + (f" ({name})" if name else "") + ":\n"
        if score is not None:
            sims_text += f"  Scor final: {score}/100  Semnal: {verdict}\n"
        if s.get("margin") is not None:
            m = float(s["margin"])
            label = "subevaluat semnificativ" if m > 30 else "subevaluat moderat" if m > 10 else "la valoare justa" if m > -5 else "supraevaluat"
            sims_text += f"  Marja siguranta vs DCF: {_fmt(m, pct=True)} ({label})\n"
        if s.get("ret") is not None:
            sims_text += f"  Randament P50 ({period}z): {_fmt(s['ret'], pct=True)}\n"
        if s.get("up") is not None and s.get("down") is not None:
            sims_text += f"  Asimetrie P90/P10: {_fmt(s['up'], pct=True)} / {_fmt(s['down'], pct=True)}\n"
        if s.get("prob") is not None:
            sims_text += f"  Probabilitate profit {period}z: {float(s['prob']):.1f}%\n"
        if s.get("vol") is not None:
            sims_text += f"  Volatilitate anualizata: {float(s['vol']):.1f}%/an\n"
        if s.get("div") is not None:
            sims_text += f"  Dividend yield: {float(s['div']):.2f}%\n"
        if s.get("sent") is not None:
            sims_text += f"  Sentiment: {_fmt(s['sent'], decimals=3)}\n"
        if s.get("dev") is not None:
            sims_text += f"  Deviatia fata de MA60: {_fmt(s['dev'], pct=True)}\n"

    prompt = f"""Esti un analist financiar senior cu 20 de ani experienta pe pietele europene.

Ai in fata rezultatele unei comparatii Monte Carlo (30.000 simulari) intre {len(req.sims)} actiuni:
{sims_text}

Scrie o analiza calitativa aprofundata in 4 paragrafe clare:

PARAGRAFUL 1 — Verdict independent: Este castigatorul matematic cu adevarat superior? Exista discrepante intre scorul total si semnalele individuale? Mentioneaza explicit daca esti de acord sau nu cu clasamentul si de ce.

PARAGRAFUL 2 — Raport risc/randament real: Analizeaza asimetria P90/P10 pentru fiecare actiune — nu doar randamentul median P50. O actiune cu P50 mic dar asimetrie favorabila poate fi mai buna decat una cu P50 mare dar risc de cadere puternica. Integreaza volatilitatea si probabilitatea de profit.

PARAGRAFUL 3 — Semnale contradictorii sau ascunse: Identifica orice contradictie intre metrici (ex: marja mare dar randament negativ, volatilitate ridicata cu probabilitate mica, dividend atractiv vs supraevaluare). Ce spune sentimentul despre contextul actual al fiecarei actiuni?

PARAGRAFUL 4 — Recomandare practica: Care actiune ofera cel mai bun punct de intrare acum si de ce? Ce risc specific trebuie monitorizat pentru castigator? Fii direct si concis ca un analist real.

Scrie in romana. Fara titluri, fara bullet points — text curgator, ton profesionist."""

    client_g = google_genai.Client(api_key=api_key, http_options={"api_version": "v1"})

    last_err = None
    for model in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        try:
            resp = client_g.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=1200,
                    temperature=0.4,
                ),
            )
            text = resp.text.strip() if resp.text else None
            if text:
                print(f"[gemini-verdict] model={model} OK, {len(text)} chars")
                return JSONResponse(content={"evaluare": text, "model": model})
            print(f"[gemini-verdict] model={model} raspuns gol")
        except Exception as e:
            last_err = e
            print(f"[gemini-verdict] model={model} eroare: {e}")
            continue

    raise HTTPException(status_code=503, detail=f"Gemini indisponibil: {last_err}")
