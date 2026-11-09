"""PharmaSignal core logic for the Streamlit app.

Loads the artefacts written by Drug_Recall_Intelligence_Final.ipynb (outputs/ folder) and re-uses the
notebook's tender-screening and assistant logic. Nothing is retrained here: run the notebook first.
"""
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
RAW_FILE = ROOT / "drug_recalls.csv"
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5-5")
RANDOM_STATE = 42
REQUIRED = [OUT / "recalls.db", OUT / "snapshot_provenance.json", OUT / "vector_db" / "documents.json",
            OUT / "tableau" / "monthly_series.csv"]

# ---------------------------------------------------------------- notebook constants (kept identical)
LEGAL_SUFFIXES = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company",
                  "lp", "llp", "plc", "pvt", "private", "gmbh", "sa", "ag", "bv"}
GENERIC_WORDS = {"pharmaceutical", "pharmaceuticals", "pharma", "pharmacy", "health", "healthcare", "medical", "laboratories",
                 "labs", "the", "international", "american", "global", "national", "generic", "generics", "united", "general",
                 "which", "what", "have", "has", "kenya", "kenyan", "fda", "nairobi", "mombasa", "best", "first", "quality",
                 "drug", "drugs", "should", "show", "tell", "are", "how", "is", "did", "does", "can", "list"}
PRODUCT_WORDS = {"hand", "sanitizer", "sanitizers", "eye", "drops", "tablets", "capsules", "injection", "cream", "ointment",
                 "solution", "spray", "supplier", "suppliers", "supply"}
NITROSAMINE_PRONE = r"sartan|ranitidine|nizatidine|metformin|rifampin|rifapentine|varenicline|sitagliptin|propranolol|duloxetine"
COMPANY_NAME_PATTERN = r"(?:[A-Z][\w&.-]*\s+)+(?:Pharma\w*|Laborator\w*|Labs?|Ltd|Limited|Inc|LLC|Healthcare|Industries|PLC|Generics)\b"
OUT_OF_SCOPE = r"\b(price|prices|cost|costs|stock level|availability|sales|market share)\b"
CATEGORY_WORDS = {"contamina": "CONTAMINATION_STERILITY", "steril": "CONTAMINATION_STERILITY", "impurit": "IMPURITIES",
                  "nitrosamine": "IMPURITIES", "ndma": "IMPURITIES", "potency": "POTENCY_STRENGTH", "dissolution": "DISSOLUTION",
                  "cgmp": "CGMP_DEVIATION", "label": "LABELING_PACKAGING", "stability": "STABILITY", "particulate": "FOREIGN_MATTER",
                  "foreign": "FOREIGN_MATTER", "undeclared": "UNAPPROVED_UNDECLARED"}
NOTE_INTENTS = {"note-procurement-checks": r"procure|tender|award|supplier|supply contract|purchas|buy|kemsa|meds",
                "note-limitations": r"limitation|kenya|relevan|reliab",
                "note-methods": r"how (do|does|did) (you|the system|it) (detect|work|classif)|method"}
SYSTEM_PROMPT = '''You are a pharmaceutical recall intelligence assistant for regulators and procurement teams in Kenya
(PPB, KEMSA, MEDS, hospital pharmacies). Follow these rules strictly:
1. Use ONLY the STRUCTURED FACTS and RETRIEVED EVIDENCE provided. Never invent firms, recall numbers, dates or numbers.
2. Structure: "Facts from the data" (cite recall numbers or document IDs in brackets), then "What this may mean"
   (clearly interpretation), then "Limitations and checks to make".
3. If the evidence does not answer the question, say so plainly.
4. An anomaly alert is a signal for investigation, NOT a confirmed quality failure.
5. Never recommend approving or rejecting a supplier and never call a product or firm unsafe. Give evidence and checks; humans decide.
6. The recalling firm may not be the manufacturer; do not infer sites or countries from names.
7. This is US FDA data up to the snapshot date; say so when relevance to Kenya is limited.
Write in plain language for pharmacists, under 300 words.'''


def standardise_firm(name):
    words = re.sub(r"[^a-z0-9 ]", " ", str(name).lower().replace("&", " and ")).split()
    words = [w for w in words if w not in LEGAL_SUFFIXES]
    return " ".join(words) if words else str(name).lower()


def load_env_file(path=ROOT / ".env"):
    """Put KEY=value lines from .env into os.environ (without overriding variables that are already set)."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def missing_outputs():
    return [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]


# ---------------------------------------------------------------- data bundle
@dataclass
class PharmaData:
    con: sqlite3.Connection
    provenance: dict
    recalls: pd.DataFrame        # one row per recall record, with failure category
    fe: pd.DataFrame             # firm evidence (one row per standardised firm)
    alerts: pd.DataFrame
    monthly: pd.DataFrame
    backtest: pd.DataFrame
    method_summary: pd.DataFrame
    evaluation: pd.DataFrame
    last_month: pd.Period
    recent_start: pd.Period
    drug_names: list
    known_firms: list
    # retrieval
    doc_ids: list
    documents: list
    metadatas: list
    embed: object                # callable: list of texts -> normalised vectors
    vectors: np.ndarray
    max_distance: float
    embedding_method: str

    @property
    def source(self):
        return f"openFDA snapshot {self.provenance['effective_snapshot_date']}"

    def run_sql(self, query, params=()):
        return pd.read_sql_query(query, self.con, params=params)


def load_data():
    con = sqlite3.connect(OUT / "recalls.db", check_same_thread=False)
    provenance = json.loads((OUT / "snapshot_provenance.json").read_text())
    recalls = pd.read_sql_query("SELECT r.*, c.analysis_category, c.category_confidence FROM recalls r "
                                "JOIN recall_categories c USING(recall_number)", con)
    recalls["report_date_dt"] = pd.to_datetime(recalls.report_date, errors="coerce")
    recalls["year"] = recalls.report_date_dt.dt.year
    fe = pd.read_sql_query("SELECT * FROM firm_evidence", con).fillna({"flags": "", "recurring_failure": "", "openfda_manufacturer": ""})
    alerts = pd.read_sql_query("SELECT * FROM alerts", con)
    t = OUT / "tableau"
    monthly = pd.read_csv(t / "monthly_series.csv")
    read = lambda p: pd.read_csv(p) if p.exists() else pd.DataFrame()
    last_month = pd.Period(monthly.month.max(), "M")

    # drug names for question routing come from the openFDA generic-name field of the raw snapshot
    generic = pd.read_csv(RAW_FILE, usecols=["openfda.generic_name"], low_memory=False)["openfda.generic_name"] \
        if RAW_FILE.exists() else pd.Series(dtype=str)
    drug_names = sorted({w for s in generic.dropna().str.lower().str.findall(r"[a-z]{6,}") for w in s}
                        - {"sodium", "hydrochloride", "acetate", "tablets", "capsules", "injection", "extended", "release",
                           "potassium", "calcium", "sulfate", "citrate", "phosphate", "chloride", "solution", "acetaminophen"}
                        | {"acetaminophen"})

    store = json.loads((OUT / "vector_db" / "documents.json").read_text())
    embed, vectors, max_distance, method = build_retriever(store["documents"])

    return PharmaData(con=con, provenance=provenance, recalls=recalls, fe=fe, alerts=alerts, monthly=monthly,
                      backtest=read(t / "backtest.csv"), method_summary=read(t / "method_summary.csv"),
                      evaluation=read(OUT / "project_evaluation.csv"), last_month=last_month,
                      recent_start=last_month - 11, drug_names=drug_names,
                      known_firms=sorted(fe.firm_std, key=len, reverse=True),
                      doc_ids=store["ids"], documents=store["documents"], metadatas=store["metadatas"],
                      embed=embed, vectors=vectors, max_distance=max_distance, embedding_method=method)


def build_retriever(documents):
    """Same search method as the notebook: semantic embeddings (all-MiniLM-L6-v2) when sentence-transformers is
    installed and the notebook saved vectors with it, otherwise the LSA fallback (TF-IDF + SVD) refitted here."""
    saved_path = OUT / "vector_db" / "vectors.npy"
    saved = np.load(saved_path) if saved_path.exists() else None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embed = lambda texts: model.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        if saved is not None and len(saved) == len(documents) and saved.shape[1] == len(embed(["dimension check"])[0]):
            vectors = saved.astype("float32")
        else:
            vectors = np.asarray(embed(documents), dtype="float32")
        return embed, vectors, 0.60, "all-MiniLM-L6-v2 (semantic embeddings)"
    except Exception:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        svd = TruncatedSVD(256, random_state=RANDOM_STATE).fit(tfidf.fit_transform(documents))
        embed = lambda texts: normalize(svd.transform(tfidf.transform(list(texts))))
        return embed, np.asarray(embed(documents), dtype="float32"), 0.80, "LSA fallback (TF-IDF + SVD)"


# ---------------------------------------------------------------- tender screening (from notebook section 6)
def match_bidder(d: PharmaData, name):
    std = standardise_firm(name)
    fe_idx = d.fe.set_index("firm_std")
    if std in fe_idx.index:
        return std, "exact name match"
    first = std.split()[0] if std else ""
    if first and first not in GENERIC_WORDS:
        cands = [f for f in fe_idx.index if f.split()[0] == first]
        if cands:
            return fe_idx.loc[cands, "total_events"].idxmax(), "matched on brand word - confirm it is the same company"
    return None, "not found"


def related_entities(d: PharmaData, firm):
    first = firm.split()[0]
    if first in GENERIC_WORDS:
        return []
    return [f for f in d.fe.firm_std if f.split()[0] == first and f != firm]


def checks_for(firm_row, product, product_recalls):
    checks = ["Confirm the actual manufacturing site of the tendered product (the recalling firm may be a repackager or distributor).",
              "Verify Pharmacy and Poisons Board (PPB) registration of the product and of that manufacturing site.",
              "Request the current GMP certificate / latest inspection outcome for the site (PPB, WHO PQ or a stringent regulator).",
              "Request certificates of analysis for the batches to be supplied."]
    failures = (firm_row.top_failures if firm_row is not None else "") + " " + \
        " ".join(product_recalls.analysis_category if len(product_recalls) else [])
    if re.search(NITROSAMINE_PRONE, product, re.I) or "IMPURITIES" in failures:
        checks.append("Request a nitrosamine risk assessment and NDMA/NDEA test results for the API and finished product.")
    if "CONTAMINATION_STERILITY" in failures:
        checks.append("Request sterility-assurance evidence (media fills, environmental monitoring) for sterile products.")
    if "DISSOLUTION" in failures or "STABILITY" in failures:
        checks.append("Request dissolution profiles and real-time stability data covering the full shelf life (Zone IVb conditions for Kenya).")
    if "LABELING_PACKAGING" in failures:
        checks.append("Inspect labelling/packaging samples and ask how line clearance prevents mix-ups.")
    if firm_row is not None and firm_row.class_I_events_12m > 0:
        checks.append("Ask for the root-cause investigation and CAPA for the recent Class I recall(s).")
    return checks


def tender_screen(d: PharmaData, bidders, product):
    """Returns (summary table, {bidder: markdown brief})."""
    rows, briefs = [], {}
    fe_idx = d.fe.set_index("firm_std")
    df = d.recalls
    for bidder in bidders:
        firm, how = match_bidder(d, bidder)
        r = fe_idx.loc[firm] if firm else None
        prod = df[(df.firm_std == firm) & df.product_description.str.contains(product, case=False, na=False, regex=False)] \
            if firm else df.iloc[0:0]
        prod_events = prod.drop_duplicates("event_id").sort_values("report_date_dt", ascending=False)
        lines = [f"# Tender evidence brief - {bidder} - {product}",
                 f"*Source: {d.provenance['source']}, snapshot {d.provenance['effective_snapshot_date']}. "
                 "Decision support only - not a supplier rating.*", ""]
        if firm is None:
            lines += ["**No US FDA recall record found under this name.** This is NOT evidence of good quality: the firm may not "
                      "supply the US market or may trade under another name.", ""]
        else:
            rel = related_entities(d, firm)
            lines += [f"**Matched recalling firm:** `{firm}` ({how}). Recorded as: {r.recorded_names}.",
                      f"Address country of recalling firm: {r.address_countries} (not necessarily the manufacturing site). "
                      f"Manufacturer named by openFDA: {r.openfda_manufacturer or 'not provided'}.",
                      f"Similar-named entities (NOT merged): {', '.join(rel[:5]) if rel else 'none'}.", "",
                      "## Recall evidence",
                      f"- Total recall events: {int(r.total_events)} (Class I: {int(r.class_I_events)})",
                      f"- Last 12 months ({d.recent_start} to {d.last_month}): {int(r.events_12m)} events, "
                      f"{int(r.class_I_events_12m)} Class I",
                      f"- Main failure types: {r.top_failures}",
                      f"- Evidence flags: {r['flags'] or 'none'}",
                      f"- {product.title()} recalls by this firm: {len(prod_events)}"]
            lines += [f"  - {p.recall_number} | {p.report_date_dt.date()} | {p.classification} | {p.analysis_category} | "
                      f"{p.reason_clean[:110]}" for p in prod_events.head(6).itertuples()]
        lines += ["", "## Checks to request before award"] + [f"- [ ] {c}" for c in checks_for(r, product, prod_events)]
        briefs[bidder] = "\n".join(lines)
        rows.append({"bidder": bidder, "matched_firm": firm or "-", "match": how,
                     "events_total": int(r.total_events) if firm else 0, "events_12m": int(r.events_12m) if firm else 0,
                     "class_I_12m": int(r.class_I_events_12m) if firm else 0, f"{product}_recalls": len(prod_events),
                     "flags": (r["flags"] or "none") if firm else "no FDA record"})
    return pd.DataFrame(rows), briefs


# ---------------------------------------------------------------- assistant (from notebook section 8)
def is_descriptive_name(firm):
    return all(w in GENERIC_WORDS | PRODUCT_WORDS for w in firm.split())


def find_firm(d: PharmaData, question):
    q = f" {standardise_firm(question)} "
    for firm in d.known_firms:
        if len(firm) >= 5 and f" {firm} " in q and not is_descriptive_name(firm):
            return firm
    for word in {re.sub(r"[^a-z]", "", w.lower()) for w in question.split()[1:] if w[:1].isupper()}:
        if len(word) >= 4 and word not in GENERIC_WORDS:
            cands = [f for f in d.known_firms if f.split()[0] == word]
            if cands:
                return d.fe.set_index("firm_std").loc[cands, "total_events"].idxmax()
    return None


def route(d: PharmaData, question):
    q = question.lower()
    firm = find_firm(d, question)
    product = next((w for w in d.drug_names if re.search(rf"\b{w}\b", q)), None)
    category = next((c for w, c in CATEGORY_WORDS.items() if w in q), None)
    facts, warn = [], []
    if firm:
        facts.append((f"Firm evidence (SQL): {firm}", d.run_sql(
            "SELECT firm_std, recorded_names, address_countries, total_events, class_I_events, events_12m, class_I_events_12m, "
            "alert_months_12m, top_failures, flags, latest_recall_numbers FROM firm_evidence WHERE firm_std = ?", (firm,))))
        facts.append((f"Class I recall events by year for {firm} (SQL)", d.run_sql(
            "SELECT substr(report_date,1,4) AS year, COUNT(DISTINCT event_id) AS class_I_events FROM recalls "
            "WHERE firm_std = ? AND classification = 'Class I' GROUP BY year", (firm,))))
        rel = related_entities(d, firm)
        if rel:
            facts.append(("Similar-named firm entities (listed separately, NOT merged)",
                          d.fe[d.fe.firm_std.isin(rel)][["firm_std", "total_events", "events_12m"]]))
    if product:
        facts.append((f"Recall events involving '{product}' by failure category (SQL)", d.run_sql(
            "SELECT c.analysis_category, COUNT(DISTINCT r.event_id) AS events, MIN(r.report_date) AS first, MAX(r.report_date) AS latest "
            "FROM recalls r JOIN recall_categories c USING(recall_number) WHERE lower(r.product_description) LIKE ? "
            "GROUP BY c.analysis_category ORDER BY events DESC", (f"%{product}%",))))
    if re.search(r"\b(most|highest|top|largest|rank)\b", q) and not firm and not re.search(r"dosage|product group", q):
        facts.append((f"Top recalling firms by distinct events, {d.recent_start} to {d.last_month} (SQL)", d.run_sql(
            "SELECT firm_std, COUNT(DISTINCT event_id) AS events, COUNT(DISTINCT CASE WHEN classification='Class I' THEN event_id END) "
            "AS class_I_events FROM recalls WHERE substr(report_date,1,7) BETWEEN ? AND ? GROUP BY firm_std "
            "ORDER BY events DESC LIMIT 10", (str(d.recent_start), str(d.last_month)))))
    if re.search(r"unusual|increase|surge|anomal|warrant|investigat|signal|spike|alert", q):
        sql, p = ("SELECT series_type, series_name, month, n_events, ROUND(hist_mean,1) AS baseline, n_methods, why_flagged "
                  "FROM alerts WHERE "), []
        if category:
            sql += "series_type='category' AND series_name=? "; p.append(category)
        elif firm:
            sql += "series_type='firm' AND series_name=? "; p.append(firm)
        else:
            sql += "month > ? "; p.append(str(d.last_month - 12))
        facts.append(("Anomaly alerts (SQL)", d.run_sql(sql + "ORDER BY n_methods DESC, month DESC LIMIT 10", tuple(p))))
    if re.search(r"dosage|product group", q):
        facts.append(("Recall events by product group and failure category (SQL)", d.run_sql(
            "SELECT r.product_group, c.analysis_category, COUNT(DISTINCT r.event_id) AS events FROM recalls r "
            "JOIN recall_categories c USING(recall_number) GROUP BY 1,2 ORDER BY events DESC LIMIT 12")))
    if re.search(COMPANY_NAME_PATTERN, question) and not firm:
        warn.append("The company named in the question does NOT appear among US FDA recalling firms in this snapshot. "
                    "No FDA record is not evidence of good quality.")
    if re.search(OUT_OF_SCOPE, q):
        warn.append("This dataset holds FDA recall records only - no prices, stock levels or sales data.")
    return {"firm": firm, "product": product, "category": category, "facts": facts, "warnings": warn}


def retrieve(d: PharmaData, question, firm=None, k=6):
    q = np.asarray(d.embed([question]), dtype="float32")[0]
    distance = 1 - d.vectors @ q
    if firm:
        keep = np.array([m.get("firm_std") == firm for m in d.metadatas])
    else:
        keep = np.array([m.get("doc_type") != "firm_evidence" for m in d.metadatas])
    distance[~keep] = np.inf
    best = [i for i in np.argsort(distance)[:k] if np.isfinite(distance[i]) and distance[i] <= d.max_distance]
    hits = pd.DataFrame({"doc_id": [d.doc_ids[i] for i in best], "distance": [round(float(distance[i]), 3) for i in best],
                         "source": [d.metadatas[i]["source"] for i in best], "text": [d.documents[i] for i in best]})
    notes = [n for n, pat in NOTE_INTENTS.items() if re.search(pat, question, re.I) and n in d.doc_ids]
    if notes:
        pos = [d.doc_ids.index(n) for n in notes]
        extra = pd.DataFrame({"doc_id": notes, "distance": 0.0, "source": [d.metadatas[p]["source"] for p in pos],
                              "text": [d.documents[p] for p in pos]})
        hits = pd.concat([extra, hits[~hits.doc_id.isin(notes)]], ignore_index=True)
    return hits


def check_answer(answer, context):
    ids_in_answer = set(re.findall(r"\b[A-Z]-\d{3,4}-\d{4}\b", answer))
    bad_ids = sorted(i for i in ids_in_answer if i not in context)
    context_numbers = set(re.findall(r"\d+(?:\.\d+)?", context))
    numbers = set(re.findall(r"(?<![\w.-])\d+(?:\.\d+)?(?![\w-])", re.sub(r"\b[A-Z]-\d{3,4}-\d{4}\b", " ", answer)))
    bad_numbers = sorted(n for n in numbers if float(n) > 12 and n not in context_numbers)
    return bad_ids, bad_numbers


def make_llm():
    """Anthropic client if an API key is available (env var or .env), else None."""
    load_env_file()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    return anthropic.Anthropic()


def _llm_text(llm, messages):
    resp = llm.beta.messages.create(model=LLM_MODEL, max_tokens=800, system=SYSTEM_PROMPT, messages=messages,
                                    betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined to answer")
    return "".join(b.text for b in resp.content if b.type == "text")


def ask(d: PharmaData, question, llm=None):
    r = route(d, question)
    ev = retrieve(d, question, r["firm"])
    facts_text = "\n\n".join(f"[{t}]\n{tab.to_string(index=False) if len(tab) else '(no rows)'}" for t, tab in r["facts"])
    context = (f"QUESTION: {question}\nSNAPSHOT: {d.source}; recent window {d.recent_start} to {d.last_month}\n"
               f"SYSTEM NOTES: {' '.join(r['warnings']) or 'none'}\n\nSTRUCTURED FACTS:\n{facts_text or '(none)'}\n\n"
               "RETRIEVED EVIDENCE:\n" + "\n".join(f"[{e.doc_id}] ({e.source}) {e.text}" for e in ev.itertuples()))
    answer, status, verified = None, "evidence only (Claude not used)", None
    if llm is not None:
        try:
            msgs = [{"role": "user", "content": context}]
            answer = _llm_text(llm, msgs)
            bad = check_answer(answer, context)
            if bad[0] or bad[1]:                       # one self-correction round, as in the notebook
                msgs += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": f"These items are not in the evidence: {bad}. Rewrite using only the evidence."}]
                answer = _llm_text(llm, msgs)
                bad = check_answer(answer, context)
            verified = not (bad[0] or bad[1])
            status = "Claude answer - checker: all numbers and IDs verified" if verified else f"Claude answer - UNVERIFIED items: {bad}"
        except Exception as e:
            answer, status = None, f"Claude call failed ({type(e).__name__}) - showing evidence only"
    return {**r, "evidence": ev, "answer": answer, "status": status, "verified": verified, "context": context}
