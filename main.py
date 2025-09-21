# main.py : Entry point for Financial RAG (GOOGL/MSFT/NVDA | 10-K | 2022-2024)

import os, re, json, time, textwrap
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pdfplumber
from tqdm import tqdm
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
import faiss
import tiktoken

# =========================
# 0) CONFIG
# =========================
USER_AGENT = "YourName-YourProject/1.0 (your_email@example.com)"  # <-- PUT YOUR REAL EMAIL
DATA_DIR = Path("sec_10k").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES = {
    "GOOGL": {"cik": "1652044"},
    "MSFT":  {"cik": "789019"},
    "NVDA":  {"cik": "1045810"},
}
YEARS = [2022, 2023, 2024]

# =========================
# 1) SEC DOWNLOAD
# =========================
def pad_cik(cik: str) -> str:
    return str(cik).zfill(10)

def sec_headers():
    return {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

def list_10k_for_year(cik: str, year: int):
    """Pick best 10-K entry for the report year."""
    url = f"https://data.sec.gov/submissions/CIK{pad_cik(cik)}.json"
    r = requests.get(url, headers=sec_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    forms = data.get("filings", {}).get("recent", {})

    entries = []
    for i, form in enumerate(forms.get("form", [])):
        if form != "10-K":
            continue
        filing_date = forms["filingDate"][i]
        acc_no = forms["accessionNumber"][i]
        primary_doc = forms["primaryDocument"][i]
        report_date = (forms.get("reportDate") or [None]*len(forms["form"]))[i]

        # derive report year
        if report_date and len(report_date) >= 4:
            ry = int(report_date[:4])
        else:
            m = re.search(r"(20\d{2})", primary_doc or "")
            ry = int(m.group(1)) if m else int(filing_date[:4])

        if ry == year:
            entries.append({
                "filing_date": filing_date,
                "accession_number": acc_no,
                "primary_doc": primary_doc
            })

    def is_amendment(docname: str):
        dn = (docname or "").lower()
        return dn.endswith(("10ka.htm","10ka.html","10k_a.htm","10k_a.html"))

    entries.sort(key=lambda x: (is_amendment(x["primary_doc"]), x["filing_date"]))
    return entries[:1]

def download_10k(cik: str, acc_no: str, primary_doc: str, out_dir: Path) -> Path:
    acc_nodash = acc_no.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}"
    doc_url = f"{base}/{primary_doc}"
    headers = {"User-Agent": USER_AGENT}
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / primary_doc
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    rr = requests.get(doc_url, headers=headers, timeout=60)
    rr.raise_for_status()
    out_path.write_bytes(rr.content)
    return out_path

def acquire_filings():
    saved = []
    for ticker, meta in COMPANIES.items():
        cik = meta["cik"]
        for y in YEARS:
            try:
                entries = list_10k_for_year(cik, y)
                if not entries:
                    print(f"[WARN] No 10-K found for {ticker} {y}")
                    continue
                e = entries[0]
                out_dir = DATA_DIR / ticker / str(y)
                fpath = download_10k(cik, e["accession_number"], e["primary_doc"], out_dir)
                saved.append((ticker, y, fpath))
                print(f"[OK] {ticker} {y} -> {fpath.name}")
                time.sleep(0.4)  # polite to SEC
            except Exception as ex:
                print(f"[ERR] {ticker} {y}: {ex}")
    return saved

# =========================
# 2) PARSE & TEXT EXTRACT
# =========================
def html_to_text(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "lxml")
    for tag in soup(["script", "style"]): tag.extract()
    text = soup.get_text(separator="\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln])

def pdf_to_text(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)

def load_document_text(path: Path) -> str:
    b = path.read_bytes()
    name = path.name.lower()
    if name.endswith((".htm", ".html")):
        return html_to_text(b)
    if name.endswith(".pdf"):
        return pdf_to_text(path)
    try:
        return html_to_text(b)
    except:
        return b.decode("utf-8", errors="ignore")

def build_corpus():
    records = []
    for ticker in COMPANIES.keys():
        for y in YEARS:
            d = DATA_DIR / ticker / str(y)
            if not d.exists(): continue
            files = list(d.glob("*"))
            if not files: continue
            doc_path = files[0]
            txt_path = d / "fulltext.txt"
            if not txt_path.exists():
                text = load_document_text(doc_path)
                txt_path.write_text(text, encoding="utf-8")
            records.append({"ticker": ticker, "year": y, "text_path": str(txt_path)})
            print(f"[OK] Text extracted: {ticker} {y} -> {txt_path.name}")
    return records

# =========================
# 3) CHUNKING
# =========================
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(s: str) -> int:
    return len(enc.encode(s))

def chunk_text(text: str, min_toks=200, max_toks=1000, target_toks=800) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf, buf_tokens = [], [], 0
    for p in paras:
        tks = count_tokens(p)
        if tks > max_toks:
            words = p.split()
            step = max(50, int(len(words) * (target_toks / tks)))
            for i in range(0, len(words), step):
                piece = " ".join(words[i:i+step])
                if piece.strip(): chunks.append(piece)
            continue
        if buf_tokens + tks <= target_toks:
            buf.append(p); buf_tokens += tks
        else:
            if buf: chunks.append("\n".join(buf))
            buf, buf_tokens = [p], tks
    if buf: chunks.append("\n".join(buf))
    return chunks

def build_all_chunks(DOCS):
    ALL = []
    for rec in DOCS:
        text = Path(rec["text_path"]).read_text(encoding="utf-8")
        chs = chunk_text(text)
        out_dir = Path(rec["text_path"]).parent
        with open(out_dir/"chunks.jsonl", "w", encoding="utf-8") as f:
            for i, c in enumerate(chs):
                f.write(json.dumps({"i": i, "text": c}) + "\n")
        ALL.append({**rec, "chunks": chs})
        print(f"[OK] {rec['ticker']} {rec['year']} chunks: {len(chs)}")
    return ALL

# =========================
# 4) EMBEDDINGS + FAISS
# =========================
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
emb_model = SentenceTransformer(EMB_MODEL_NAME)

def build_index_for_doc(ticker: str, year: int, chunks: List[str]):
    vecs = emb_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False,
                            batch_size=64, normalize_embeddings=True)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    meta = [{"i": i, "ticker": ticker, "year": year} for i in range(len(chunks))]
    return index, vecs, meta

INDEXES: Dict[Tuple[str,int], Dict] = {}

def build_indexes(ALL_CHUNKS):
    for entry in ALL_CHUNKS:
        key = (entry["ticker"], entry["year"])
        idx, vecs, meta = build_index_for_doc(entry["ticker"], entry["year"], entry["chunks"])
        INDEXES[key] = {"index": idx, "chunks": entry["chunks"], "meta": meta}
        print(f"[OK] Indexed {key} with {len(entry['chunks'])} chunks")

def retrieve(ticker: str, year: int, query: str, k: int = 12):
    key = (ticker, year)
    if key not in INDEXES: return []
    idx = INDEXES[key]["index"]; chs = INDEXES[key]["chunks"]
    qvec = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = idx.search(qvec, k)
    results = []
    for rank, cid in enumerate(ids[0]):
        if cid == -1: continue
        results.append({"rank": rank, "score": float(scores[0][rank]), "chunk_id": int(cid), "text": chs[cid]})
    return results

# =========================
# 5) AGENT & EXTRACTION (PATCHED)
# =========================
METRIC_KEYWORDS = {
    # Revenues
    "total revenue": [
        "total revenue", "revenues", "net sales", "consolidated revenue",
        "total net revenue", "net revenue"
    ],
    # Margins
    "gross margin": [
        "gross margin", "gross margin percentage", "gross profit margin",
        "gross margin rate", "gross profit as a percentage of revenue"
    ],
    "operating margin": [
        "operating margin", "operating income margin",
        "income from operations as a percentage of revenue",
        "operating income as a percentage of revenue"
    ],
    # Segments
    "cloud revenue": [
        "cloud revenue", "google cloud revenue", "microsoft cloud revenue",
        "intelligent cloud", "azure", "gcp", "gcp revenue", "google cloud"
    ],
    "data center revenue": [
        "data center revenue", "datacenter revenue", "data centre revenue",
        "data center", "datacenter"
    ],
    "advertising": [
        "advertising revenue", "ads revenue", "ad revenues", "advertising",
        "google advertising", "google ads", "search and other", "YouTube ads"
    ],
    # R&D / AI
    "r&d": ["research and development", "r&d", "research & development"],
    "ai investment": [
        "artificial intelligence", "ai", "ai infrastructure", "ai investment",
        "ai risk", "risk factors", "ai strategy"
    ]
}

def normalize_company(name: Optional[str]) -> str:
    if not name: return ""
    name = name.lower().strip()
    if name.endswith("'s") or name.endswith("’s"): name = name[:-2]
    if "microsoft" in name or "msft" in name: return "MSFT"
    if "google" in name or "alphabet" in name or "googl" in name: return "GOOGL"
    if "nvidia" in name or "nvda" in name: return "NVDA"
    return name.upper()

def decompose_query(q: str) -> Dict:
    q_low = q.lower().replace("’", "'")

    # Basic metric
    m_basic = re.search(
        r"what (?:was|is).*?(microsoft(?:'s)?|google|alphabet|nvidia).*?"
        r"(total revenue|operating margin|gross margin|cloud revenue|data center revenue|advertising).*?"
        r"(20\d{2})", q_low
    )
    if m_basic:
        co = normalize_company(m_basic.group(1)); metric = m_basic.group(2); year = int(m_basic.group(3))
        return {"type": "basic", "sub": [(co, year, f"{co} {metric} {year}")], "metric": metric}

    # YoY comparison
    m_yoy = re.search(
        r"(?:how .*?(?:grow|change).*?)(nvidia(?:'s)?|microsoft(?:'s)?|google|alphabet).*?"
        r"(data center|cloud|total revenue|operating margin|gross margin).*?"
        r"(20\d{2}).*?(20\d{2})", q_low
    )
    if m_yoy:
        co = normalize_company(m_yoy.group(1)); metric = m_yoy.group(2)
        y1, y2 = sorted([int(m_yoy.group(3)), int(m_yoy.group(4))])
        subs = [(co, y1, f"{co} {metric} {y1}"), (co, y2, f"{co} {metric} {y2}")]
        return {"type": "yoy", "sub": subs, "metric": metric, "years": (y1, y2)}

    # Cross-company
    m_cross = re.search(r"which company.*?(operating margin|gross margin|total revenue).*?(20\d{2})", q_low)
    if m_cross:
        metric = m_cross.group(1); year = int(m_cross.group(2))
        subs = [(c, year, f"{c} {metric} {year}") for c in ["MSFT","GOOGL","NVDA"]]
        return {"type":"cross", "sub": subs, "metric": metric, "year": year}

    # Cloud share (% of total)
    m_pct = re.search(
        r"(?:what.*?(?:percent|percentage).*?)(google|alphabet|microsoft|nvidia).*?(cloud).*?(20\d{2})",
        q_low
    )
    if m_pct:
        co = normalize_company(m_pct.group(1)); year = int(m_pct.group(3))
        return {"type":"segment_share", "sub":[(co, year, f"{co} cloud revenue as % of total {year}")], "metric":"cloud share"}

    # Advertising share (% of total) — NEW
    m_adv = re.search(
        r"(?:what.*?(?:percent|percentage).*?)(google|alphabet).*?(advertis|ads).*?(20\d{2})",
        q_low
    )
    if m_adv:
        co = normalize_company(m_adv.group(1))
        year = int(m_adv.group(3))
        return {
            "type": "segment_share_adv",
            "sub": [
                (co, year, f"{co} advertising revenue {year}"),
                (co, year, f"{co} total revenue {year}")
            ],
            "metric": "advertising share",
            "year": year
        }

    # AI strategy compare
    if ("ai" in q_low) and ("compare" in q_low):
        m_year = re.search(r"(20\d{2})", q_low)
        year = int(m_year.group(1)) if m_year else YEARS[-1]
        subs = [(c, year, f"{c} AI investments risks {year}") for c in ["MSFT","GOOGL","NVDA"]]
        return {"type":"ai_compare", "sub": subs, "metric":"ai investments", "year": year}

    # Fallback
    subs = [(c, y, q) for c in ["MSFT","GOOGL","NVDA"] for y in YEARS]
    return {"type":"fallback", "sub": subs, "metric": "unknown"}

def keywords_for_metric(metric_or_subq: str) -> List[str]:
    low = metric_or_subq.lower()
    for k, v in METRIC_KEYWORDS.items():
        if k in low:
            return v
    # Heuristic fallbacks
    if "gross" in low and "margin" in low:
        return METRIC_KEYWORDS["gross margin"]
    if "operating" in low and "margin" in low:
        return METRIC_KEYWORDS["operating margin"]
    if "cloud" in low:
        return METRIC_KEYWORDS["cloud revenue"]
    if "data center" in low or "datacenter" in low:
        return METRIC_KEYWORDS["data center revenue"]
    if "advertis" in low or "ads" in low:
        return METRIC_KEYWORDS["advertising"]
    if "revenue" in low:
        return METRIC_KEYWORDS["total revenue"]
    return [low]

def extract_numbers_near_keywords(text: str, keywords: List[str]) -> List[str]:
    window = 240  # widened
    found = []
    low = text.lower()
    for kw in keywords:
        for m in re.finditer(re.escape(kw.lower()), low):
            start = max(0, m.start() - window); end = min(len(text), m.end() + window)
            snippet = text[start:end]
            for pat in [r"\$[\s]*[\d\.,]+(?:\s*(?:billion|million))?",
                        r"[\d\.,]+%",
                        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
                        r"\b\d+(?:\.\d+)?\b"]:
                for nm in re.findall(pat, snippet, flags=re.IGNORECASE):
                    found.append((kw, nm.strip(), snippet))
    # dedupe
    unique, seen = [], set()
    for kw, nm, sn in found:
        key = (kw, nm)
        if key not in seen:
            seen.add(key); unique.append((kw, nm, sn))
    return unique

def pick_best_numeric(matches: List[Tuple[str,str,str]]) -> Optional[Tuple[str,str,str]]:
    for kw, nm, sn in matches:
        if "%" in nm: return (kw, nm, sn)
    for kw, nm, sn in matches:
        if nm.startswith("$"): return (kw, nm, sn)
    return matches[0] if matches else None

def retrieve_multi(ticker: str, year: int, subq: str, k_each: int = 8, max_total: int = 30):
    """
    Build multiple related queries (metric synonyms + section hints) and union their top-k.
    """
    key = (ticker, year)
    if key not in INDEXES:
        return []

    kws = keywords_for_metric(subq)
    section_hints = [
        "Item 7", "Management's Discussion and Analysis", "MD&A",
        "Item 8", "Financial Statements", "Consolidated Statements",
        "Item 1A", "Risk Factors"
    ]

    variants = [subq]
    for kw in kws:
        variants.append(f"{kw} {year}")
        variants.append(f"{kw} {ticker} {year}")
    for kw in kws:
        for hint in section_hints:
            variants.append(f"{kw} {hint} {year}")

    idx = INDEXES[key]["index"]
    chs = INDEXES[key]["chunks"]

    seen = set()
    results = []
    for q in variants[:18]:   # cap variants for speed
        qvec = emb_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        scores, ids = idx.search(qvec, k_each)
        for rank, cid in enumerate(ids[0]):
            if cid == -1: continue
            if cid in seen:  continue
            seen.add(cid)
            results.append({
                "rank": rank, "score": float(scores[0][rank]),
                "chunk_id": int(cid), "text": chs[cid], "query": q
            })
            if len(results) >= max_total:
                break
        if len(results) >= max_total:
            break

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def answer_subquery(ticker: str, year: int, subq: str) -> Dict:
    hits = retrieve_multi(ticker, year, subq, k_each=8, max_total=30)
    kws = keywords_for_metric(subq)

    best = None
    chosen_hit = None

    # Pass 1: keyword-scoped numeric extraction on top-N
    for h in hits[:20]:
        matches = extract_numbers_near_keywords(h["text"], kws)
        cand = pick_best_numeric(matches)
        if cand:
            best = cand
            chosen_hit = h
            break

    # Pass 2: generic numeric scan if keyword-bound search missed
    if not best:
        for h in hits[:20]:
            generic_matches = extract_numbers_near_keywords(h["text"], ["revenue", "margin", "percent", "percentage"])
            cand = pick_best_numeric(generic_matches)
            if cand:
                best = cand
                chosen_hit = h
                break

    res = {"company": ticker, "year": year, "query": subq}
    if best and chosen_hit:
        res["value"] = best[1]
        res["excerpt"] = textwrap.shorten(best[2], width=300, placeholder="…")
        res["chunk_id"] = chosen_hit["chunk_id"]
        res["score"] = chosen_hit["score"]
    else:
        res["value"] = None
    return res

def synthesize(query: str, plan: Dict) -> Dict:
    sub_results = [answer_subquery(co, y, sq) for (co, y, sq) in plan["sub"]]
    answer_text = "Could not determine from filings."

    if plan["type"] == "basic":
        r = sub_results[0]
        if r["value"]:
            answer_text = f"{r['company']} {plan.get('metric','')} in {r['year']} appears to be {r['value']}."
    elif plan["type"] == "yoy":
        vals = [r for r in sub_results if r["value"]]
        if len(vals) >= 2:
            vals = sorted(vals, key=lambda x: x["year"])
            v1, v2 = vals[0]["value"], vals[1]["value"]
            answer_text = f"{vals[1]['company']} {plan['metric']} changed from {v1} in {vals[0]['year']} to {v2} in {vals[1]['year']}."
    elif plan["type"] == "cross":
        parsed = []
        for r in sub_results:
            if not r["value"]: continue
            vtxt = r["value"].replace(",", "")
            num = None
            if vtxt.endswith("%"):
                try: num = float(vtxt.strip("%"))
                except: pass
            elif vtxt.startswith("$"):
                m = re.search(r"\$([\d\.]+)\s*(billion|million)?", vtxt, re.I)
                if m:
                    base = float(m.group(1)); scale = (m.group(2) or "").lower()
                    num = base * 1000 if scale == "billion" else base
            if num is not None:
                parsed.append((r["company"], num, r))
        if parsed:
            parsed.sort(key=lambda x: x[1], reverse=True)
            winner = parsed[0][2]
            answer_text = f"{winner['company']} appears highest for {plan['metric']} in {plan['year']} with {winner['value']}."
    elif plan["type"] == "segment_share":
        r = sub_results[0]
        if r["value"]:
            answer_text = f"{r['company']} cloud share of revenue in {r['year']} appears to be {r['value']}."
    elif plan["type"] == "segment_share_adv":
        # Expect two sub-results: [advertising revenue, total revenue]
        if len(sub_results) >= 2 and sub_results[0].get("value") and sub_results[1].get("value"):
            def parse_dollar(v):
                v = v.lower().replace(",", "").strip()
                m = re.search(r"\$([\d\.]+)\s*(billion|million)?", v)
                if m:
                    base = float(m.group(1)); scale = (m.group(2) or "").lower()
                    if scale == "billion": return base * 1_000  # normalize to millions
                    if scale == "million": return base
                    return base
                m2 = re.search(r"\$([\d\.]+)", v)
                return float(m2.group(1)) if m2 else None
            adv = parse_dollar(sub_results[0]["value"])
            tot = parse_dollar(sub_results[1]["value"])
            if adv and tot and tot != 0:
                pct = (adv / tot) * 100
                answer_text = f"{sub_results[0]['company']} advertising share of revenue in {plan['year']} ≈ {pct:.1f}%."
            else:
                answer_text = "Could not compute share from extracted values."
        else:
            answer_text = "Could not determine from filings."
    elif plan["type"] == "ai_compare":
        bullets = []
        for r in sub_results:
            val = r["value"] or "see excerpt"
            bullets.append(f"{r['company']}: {val}")
        answer_text = " | ".join(bullets)

    sources = []
    for r in sub_results:
        s = {"company": r["company"], "year": r["year"]}
        if r.get("excerpt"): s["excerpt"] = r["excerpt"]
        if r.get("chunk_id") is not None: s["chunk_id"] = r["chunk_id"]
        sources.append(s)
    return {"query": query, "answer": answer_text, "sub_queries": [sq for (_,_,sq) in plan["sub"]], "sources": sources}

def ask(query: str) -> Dict:
    plan = decompose_query(query)
    return synthesize(query, plan)

# =========================
# 6) RUNNERS / DEMO
# =========================
REQUIRED_QUERIES = [
    "What was Microsoft's total revenue in 2023?",
    "How did NVIDIA’s data center revenue grow from 2022 to 2023?",
    "Which company had the highest operating margin in 2023?",
    "What percentage of Google's revenue came from cloud in 2023?",
    "Compare AI investments mentioned by all three companies in their 2024 10-Ks"
]

SAMPLE_TEST_QUERIES = [
    "What was NVIDIA's total revenue in fiscal year 2024?",
    "What percentage of Google's 2023 revenue came from advertising?",
    "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
    "Which of the three companies had the highest gross margin in 2023?",
    "Compare the R&D spending as a percentage of revenue across all three companies in 2023",
    "How did each company's operating margin change from 2022 to 2024?",
    "What are the main AI risks mentioned by each company and how do they differ?"
]

def pretty_print(outputs):
    for i, o in enumerate(outputs, 1):
        print(f"\nQ{i}. {o['query']}")
        print("Ans:", o['answer'])
        if o.get("sources"):
            s0 = o["sources"][0]
            ex = (s0.get("excerpt","") or "")[:220].replace("\n"," ")
            print("Source:", f"{s0.get('company','')} {s0.get('year','')}", "| Excerpt:", ex, "...")

def run_required_queries():
    print("\n[RUN] Required 5 queries")
    outputs = [ask(q) for q in REQUIRED_QUERIES]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "sample_outputs.json","w",encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    pretty_print(outputs)
    print("\nSaved ->", DATA_DIR / "sample_outputs.json")
    return outputs

def run_sample_test_queries():
    print("\n[RUN] Sample Test Queries")
    outputs = [ask(q) for q in SAMPLE_TEST_QUERIES]
    with open(DATA_DIR / "test_outputs.json","w",encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    pretty_print(outputs)
    print("\nSaved ->", DATA_DIR / "test_outputs.json")
    return outputs

def demo_agent_decomposition():
    complex_q = "Compare the R&D spending as a percentage of revenue across all three companies in 2023"
    plan = decompose_query(complex_q)
    print("\n[DEMO] Agent Decomposition")
    print("Query:", complex_q)
    print("Type :", plan.get("type"))
    print("Sub-queries:")
    for s in plan.get("sub", [])[:8]:
        print("  -", s)
    if len(plan.get("sub", [])) > 8:
        print("  ...")

# =========================
# 7) PIPELINE
# =========================
def run_pipeline():
    print("[STEP] Acquire SEC filings…")
    acquire_filings()
    print("[STEP] Parse & build corpus…")
    DOCS = build_corpus()
    print("[STEP] Chunk all docs…")
    ALL_CHUNKS = build_all_chunks(DOCS)
    print("[STEP] Build FAISS indexes…")
    build_indexes(ALL_CHUNKS)

    # Demo/Outputs
    run_required_queries()
    run_sample_test_queries()
    demo_agent_decomposition()

if __name__ == "__main__":
    run_pipeline()
