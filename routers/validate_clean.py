"""
Router: POST /validate-fundamentals
AI validator (Claude Haiku / Gemini Flash) pentru date fundamentale.
Include helper _fetch_reit_live_data pentru REIT-uri.
"""

import os
import json
import re
import traceback
from datetime import datetime, timedelta

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


# ── Benchmarks P/E sectoriale (Damodaran 2025, piete globale) ──────────
# sursa: pages.stern.nyu.edu/~adamodar/ — actualizat anual
# (label, pe_median, div_yield_avg%)
SECTOR_PE_DAMODARAN = {
    "energy":                  ("Energie",            10.5, 4.0),
    "tech":                    ("Tehnologie",         28.0, 0.8),
    "technology":              ("Tehnologie",         28.0, 0.8),
    "communication services":  ("Comunicatii",        16.5, 1.5),
    "utilities":               ("Utilitati",          17.5, 3.8),
    "utilitati":               ("Utilitati",          17.5, 3.8),
    "financial services":      ("Servicii Financiare",13.5, 2.8),
    "banci":                   ("Banci",              12.0, 3.5),
    "insurance":               ("Asigurari",          15.0, 2.0),
    "asigurari":               ("Asigurari",          15.0, 2.0),
    "real estate":             ("Imobiliare/REIT",    35.0, 4.5),
    "reit":                    ("REIT",               35.0, 4.5),
    "industrials":             ("Industriale",        20.5, 1.8),
    "conglomerate":            ("Conglomerate",       20.5, 1.8),
    "healthcare":              ("Sanatate",           22.5, 1.6),
    "basic materials":         ("Materiale",          14.0, 3.0),
    "materiale":               ("Materiale",          14.0, 3.0),
    "consumer defensive":      ("Consum Defensiv",    20.0, 2.8),
    "consumer cyclical":       ("Consum Ciclic",      18.0, 1.2),
    "consum":                  ("Consum",             19.0, 2.0),
    "auto":                    ("Auto",               13.0, 1.5),
    "shipping":                ("Transport Naval",     8.5, 4.5),
}


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


# ── Helper: fetch recomandări analiști + știri din Finnhub ──

async def _fetch_external_data(tickers: list, finnhub_key: str) -> dict:
    """
    Fetches analyst recommendations and recent news from Finnhub for each ticker.
    Returns dict: { ticker: { 'reco': {...}, 'news': [...] } }
    """
    from yahoo_client import _to_finnhub_ticker
    base     = "https://finnhub.io/api/v1"
    today    = datetime.utcnow().date()
    from_dt  = (today - timedelta(days=30)).isoformat()
    to_dt    = today.isoformat()
    results  = {}

    async with httpx.AsyncClient(follow_redirects=True, timeout=8.0) as client:
        for ticker in tickers:
            fh = _to_finnhub_ticker(ticker)
            entry = {"reco": None, "news": []}
            try:
                reco_r, news_r = await asyncio.gather(
                    client.get(f"{base}/stock/recommendation?symbol={fh}&token={finnhub_key}"),
                    client.get(f"{base}/company-news?symbol={fh}&from={from_dt}&to={to_dt}&token={finnhub_key}"),
                    return_exceptions=True
                )
                # Recomandări analiști — luăm cea mai recentă
                if not isinstance(reco_r, Exception) and reco_r.status_code == 200:
                    reco_list = reco_r.json()
                    if reco_list and isinstance(reco_list, list):
                        r = reco_list[0]  # cel mai recent
                        entry["reco"] = {
                            "buy":        r.get("buy", 0),
                            "hold":       r.get("hold", 0),
                            "sell":       r.get("sell", 0),
                            "strongBuy":  r.get("strongBuy", 0),
                            "strongSell": r.get("strongSell", 0),
                            "period":     r.get("period", ""),
                        }
                # Știri recente — maxim 4 titluri
                if not isinstance(news_r, Exception) and news_r.status_code == 200:
                    news_list = news_r.json()
                    if isinstance(news_list, list):
                        entry["news"] = [
                            {"headline": n.get("headline", ""), "datetime": n.get("datetime", 0)}
                            for n in news_list[:4]
                            if n.get("headline")
                        ]
            except Exception as e:
                print(f"[external_data] {ticker} eroare: {e}")
            results[ticker] = entry

    return results


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
    shares_val = req.fields.get("shares")  # shares in milioane
    if eps_val and eps_val > 0 and shares_val and shares_val > 0:
        # FCF total M estimat: EPS * shares * factor realist
        fcf_max_m = round(eps_val * shares_val * 3, 0)
        fcf_min_m = round(eps_val * shares_val * -2, 0)
        fcf_rule = (
            f"- FCF total (milioane): pentru {req.ticker} (EPS={eps_val}, shares={shares_val}M), "
            f"intervalul realist este [{fcf_min_m}M, {fcf_max_m}M]; "
            f"valori foarte mici (sub 10M pt companii mari) sau enorme sunt erori de date"
        )
    else:
        fcf_rule = (
            "- FCF total (milioane): verifica ordinul de marime pentru aceasta companie; "
            "valori negative acceptabile (capex mare); valori peste 100.000M sunt aproape sigur erori"
        )

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
- fcf, assets, cash, totalLiabilities (milioane): verifica ordinul de marime pentru companie{reit_rules}
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
            client_g      = google_genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
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

    # -- Fetch date externe Finnhub --
    finnhub_key  = os.environ.get("FINNHUB_KEY", "")
    tickers_list = [s.get("ticker", "") for s in req.sims if s.get("ticker")]
    ext_data     = {}
    has_external = False
    if finnhub_key and tickers_list:
        try:
            ext_data     = await _fetch_external_data(tickers_list, finnhub_key)
            has_external = any(
                (v.get("reco") or v.get("news"))
                for v in ext_data.values()
            )
        except Exception as e:
            print(f"[gemini-verdict] external fetch eroare: {e}")

    # -- Construieste context P/E sectorial (Damodaran) --
    sector_pe_context = ""
    sector_pe_lines = []
    for s in req.sims:
        sector_raw = (s.get("sector") or "").lower().strip()
        pe_info    = SECTOR_PE_DAMODARAN.get(sector_raw)
        if pe_info:
            label, pe_med, div_med = pe_info
            line = f"  {s['ticker']}: sector {label} — P/E median sector = {pe_med}x, dividend yield mediu sector = {div_med}%"
            sector_pe_lines.append(line)
    if sector_pe_lines:
        sector_pe_context = (
            "\nDATE SECTORIALE (Damodaran 2025):\n"
            + "\n".join(sector_pe_lines)
            + "\n"
        )

    # -- Construieste sectiunea surse externe --
    ext_lines = []
    if has_external:
        ext_lines.append("")
        ext_lines.append("")
        ext_lines.append("DATE INDEPENDENTE DIN SURSE EXTERNE (Finnhub):")
        for tkr, entry in ext_data.items():
            ext_lines.append("")
            ext_lines.append(f"\u25b8 {tkr}:")
            reco = entry.get("reco")
            if reco:
                total = (reco["strongBuy"] + reco["buy"] + reco["hold"]
                         + reco["sell"] + reco["strongSell"])
                if total > 0:
                    ext_lines.append(
                        f"  Recomandari analisti ({reco['period']}): "
                        f"Strong Buy={reco['strongBuy']}, Buy={reco['buy']}, "
                        f"Hold={reco['hold']}, Sell={reco['sell']}, "
                        f"Strong Sell={reco['strongSell']} (total {total} analisti)"
                    )
            for nw in entry.get("news", []):
                ext_lines.append(f"  - {nw['headline']}")
    ext_text = "\n".join(ext_lines)

    # -- Paragraful 4 variabil --
    par4 = (
        "Paragraful 4: Opinie independenta de datele simulate — bazeaza-te EXCLUSIV"
        " pe recomandarile analistilor si stirile recente de mai sus."
        " Ce spune consensul pietei? Stirile recente sunt pozitive sau negative?"
        " Exista divergenta intre modelul MC si sursele externe?"
        if has_external else
        "Paragraful 4: Opinie calitativa independenta de datele simulate —"
        " pe baza cunostintelor tale despre aceste companii si sectoarele lor,"
        " ce factori externi (macro, sector, competitie) ar putea invalida sau confirma rezultatele MC?"
    )

    system_prompt = (
        "Esti un analist financiar senior specializat pe piete europene. "
        "Raspunzi intotdeauna in romana, cu propozitii scurte si directe. "
        "Nu folosesti fraze introductive despre tine sau despre ce urmeaza sa faci. "
        "Intri direct in analiza datelor."
    )

    par4_label = "opinie independenta din surse externe Finnhub" if has_external else "opinie calitativa independenta de simulare"
    user_prompt = (
        f"Comparatie Monte Carlo (30.000 simulari) pentru {len(req.sims)} actiuni:\n"
        f"{sims_text}"
        f"{sector_pe_context}"
        f"{ext_text}\n"
        "\nFORMAT OBLIGATORIU — respecta EXACT aceasta structura, separa fiecare element cu o linie goala:\n"
        "5 paragrafe principale (3-5 propozitii fiecare), dupa fiecare un bloc [OBJ]...[/OBJ] (1-2 propozitii validare externa), la final un bloc [GENERAL]...[/GENERAL] (3-4 propozitii analiza contextuala).\n"
        "Fara titluri, fara liste, fara text in afara acestui format.\n"
        "\nParagraf 1: Evalueaza daca scorul matematic reflecta corect realitatea."
        " Citeaza numerele concrete (scor, marja, probabilitate). Spune daca esti de acord cu castigatorul sau nu.\n"
        "[OBJ]Compara P/E si dividend yield al castigatorului cu media sectorului din datele Damodaran de mai sus. Citeaza numerele concrete.[/OBJ]\n"
        "\nParagraf 2: Compara asimetria P90 vs P10 pentru fiecare actiune."
        " Cine are cel mai bun raport potential/risc real, independent de P50?"
        " Cum se coreleaza volatilitatea cu asimetria?\n"
        "[OBJ]Raporteaza volatilitatea la contextul sectorului. Este ridicata sau normala pentru sector? Citeaza benchmark-ul sectorial.[/OBJ]\n"
        "\nParagraf 3: Identifica principalele contradictii din date."
        " Marja de siguranta contrazice randamentul MC? Sentimentul negativ sau pozitiv schimba contextul?"
        " Dividendul este sustenabil?\n"
        "[OBJ]Compara dividend yield-ul fiecarei actiuni cu media sectorului din datele de mai sus. Citeaza numerele si exprima o opinie clara.[/OBJ]\n"
        f"\nParagraf 4: {par4_label} —"
        f" {par4.split(chr(8212))[-1].strip() if chr(8212) in par4 else par4}\n"
        "[OBJ]Pe baza recomandarilor analistilor si stirilor: consensul extern confirma sau contrazice simularea? Citeaza buy/hold/sell count.[/OBJ]\n"
        "\nParagraf 5: Recomanda o singura actiune pentru intrare acum. Explica de ce. Numeste un risc concret de urmarit.\n"
        "[OBJ]Ce date externe (sectorial, analisti, stiri) sustin sau slabesc aceasta recomandare? Fii specific.[/OBJ]\n"
        "\n[GENERAL]Analiza generala contextuala: contextul macro/sectorial, trenduri de sector, factori externi relevanti"
        " pentru aceste companii bazat pe datele injectate. 3-4 propozitii.[/GENERAL]"
    )

    client_g = google_genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    last_err = None
    for model in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        try:
            resp = client_g.models.generate_content(
                model=model,
                contents=[
                    {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
                ],
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=2200,
                    temperature=0.5,
                ),
            )
            text = resp.text.strip() if resp.text else None
            if text:
                print(f"[gemini-verdict] model={model} OK, {len(text)} chars")
                return JSONResponse(content={"evaluare": text, "model": model, "has_external": has_external})
            print(f"[gemini-verdict] model={model} raspuns gol")
        except Exception as e:
            last_err = e
            print(f"[gemini-verdict] model={model} eroare: {e}")
            continue

    raise HTTPException(status_code=503, detail=f"Gemini indisponibil: {last_err}")
