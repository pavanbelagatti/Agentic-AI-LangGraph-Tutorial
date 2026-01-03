from __future__ import annotations

import os
import json
from typing import TypedDict, List, Dict, Any
from urllib.parse import urlparse
from datetime import datetime

import numpy as np
from dateutil.parser import parse as dtparse
from sklearn.cluster import KMeans
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch


# =========================================================
# Load .env BEFORE initializing OpenAI clients
# (Streamlit may run with a different working directory)
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # /.../ai-trends-2026-agent
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(ENV_PATH)


class TrendState(TypedDict, total=False):
    topic: str
    iterations_left: int
    queries: List[str]
    sources: List[Dict[str, Any]]
    claims: List[Dict[str, Any]]
    clusters: List[Dict[str, Any]]
    report: str
    debug_claim_extraction: List[Dict[str, Any]]


# -------------------------------
# Model + Tools
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
emb = OpenAIEmbeddings(model="text-embedding-3-small")

search_tool = TavilySearch(
    max_results=8,
    search_depth="advanced",
)

# -------------------------------
# Scoring helpers
# -------------------------------
REPUTATION_BOOST = {
    "gartner.com": 0.35,
    "forrester.com": 0.35,
    "mckinsey.com": 0.30,
    "bain.com": 0.25,
    "bcg.com": 0.25,
    "mit.edu": 0.20,
    "stanford.edu": 0.20,
    "microsoft.com": 0.20,
    "google.com": 0.20,
    "ibm.com": 0.20,
    "openai.com": 0.20,
    "nvidia.com": 0.20,
    "axios.com": 0.18,
    "ft.com": 0.22,
    "wsj.com": 0.22,
    "economist.com": 0.22,
    "theverge.com": 0.15,
    "wired.com": 0.15,
}


def publisher_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower().replace("www.", "")
    return host


def parse_date_maybe(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return dtparse(s, fuzzy=True)
    except Exception:
        return None


def freshness_score(dt: datetime | None) -> float:
    if not dt:
        return 0.0
    y = dt.year
    if y >= 2026:
        return 0.20
    if y == 2025:
        return 0.12
    if y == 2024:
        return 0.05
    return 0.0


def trust_score(publisher: str, snippet: str, published: datetime | None) -> float:
    score = 0.35
    score += REPUTATION_BOOST.get(publisher, 0.0)

    txt = (snippet or "").lower()
    if any(w in txt for w in ["survey", "report", "data", "analysis", "research", "benchmark"]):
        score += 0.10
    if any(w in txt for w in ["according to", "estimate", "forecast", "projection"]):
        score += 0.05

    hype_hits = sum(txt.count(w) for w in ["shocking", "insane", "guaranteed", "secret", "unbelievable"])
    score -= min(0.15, hype_hits * 0.03)

    score += freshness_score(published)
    return float(np.clip(score, 0.0, 1.0))


# =========================================================
# Robust JSON extraction helper
# =========================================================
def _safe_json_extract(text: str):
    """
    Extract JSON from an LLM response even if it includes extra text/code fences.
    Returns python object or None.
    """
    text = (text or "").strip()

    # strip ```json fences if present
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    # Try direct load
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON object region
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None

    return None


# -------------------------------
# Agent Nodes
# -------------------------------
def plan_search(state: TrendState) -> TrendState:
    prompt = f"""
You are researching: {state["topic"]}

Generate 3 focused web search queries to find high-quality AI predictions for 2026.
Prefer: analyst firms, major AI/cloud vendors, reputable tech journalism.
Avoid: low-quality listicles, SEO spam.

Return ONLY a JSON array of strings.
"""
    resp = llm.invoke(prompt).content.strip()

    queries: List[str] = []
    try:
        queries = json.loads(resp)
    except Exception:
        queries = [ln.strip("- ").strip() for ln in resp.splitlines() if ln.strip()]

    state.setdefault("queries", [])
    for q in queries:
        if q and q not in state["queries"]:
            state["queries"].append(q)
    return state


def search_web(state: TrendState) -> TrendState:
    state.setdefault("sources", [])
    latest = state["queries"][-3:] if state.get("queries") else []
    for q in latest:
        results = search_tool.invoke(q)
        items = results.get("results", []) if isinstance(results, dict) else (results or [])
        for r in items:
            url = r.get("url")
            if not url:
                continue
            state["sources"].append({
                "title": r.get("title", ""),
                "url": url,
                "content": (r.get("content") or r.get("snippet") or "")[:1200],
                "published_date": r.get("published_date") or r.get("date") or None,
                "query": q
            })
    return state


def dedupe_sources(state: TrendState) -> TrendState:
    seen = set()
    uniq = []
    for s in state.get("sources", []):
        u = s.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        uniq.append(s)
    state["sources"] = uniq[:30]
    return state


def extract_claims(state: TrendState) -> TrendState:
    claims: List[Dict[str, Any]] = []
    debug_samples: List[Dict[str, Any]] = []

    for s in state.get("sources", []):
        url = s["url"]
        pub = publisher_from_url(url)
        published = parse_date_maybe(s.get("published_date"))
        trust = trust_score(pub, s.get("content", ""), published)

        prompt = f"""
You are extracting forward-looking AI predictions/trends that will likely matter by 2026
(even if the source talks about 2025-2027 or "next 1-2 years").

Return STRICT JSON object with this shape ONLY:
{{
  "items": [
    {{"claim": "one clear sentence", "category": "short label"}},
    ...
  ]
}}

Rules:
- Extract 3 to 6 items if possible.
- If the snippet has no real predictions, return {{ "items": [] }}.
- No extra keys. No markdown. No commentary.

TITLE: {s.get("title","")}
URL: {url}
SNIPPET:
{s.get("content","")}
"""
        resp = llm.invoke(prompt).content
        data = _safe_json_extract(resp)

        if data is None or not isinstance(data, dict) or "items" not in data:
            if len(debug_samples) < 3:
                debug_samples.append({
                    "url": url,
                    "title": s.get("title", ""),
                    "raw_response": (resp or "")[:800]
                })
            continue

        items = data.get("items", [])
        if not isinstance(items, list):
            continue

        for c in items[:8]:
            if not isinstance(c, dict):
                continue
            text = (c.get("claim") or "").strip()
            if not text:
                continue

            claims.append({
                "claim": text,
                "category": (c.get("category") or "General").strip(),
                "url": url,
                "title": s.get("title", ""),
                "publisher": pub,
                "publish_date": published.isoformat() if published else None,
                "trust": trust
            })

    state["claims"] = claims
    state["debug_claim_extraction"] = debug_samples
    return state


# =========================================================
# UPDATED: cluster titles now trend-like (not article titles)
# - representative_claim: closest claim to centroid
# - title: strict noun-phrase trend label (3-6 words, no year, no hype)
# - summary: 1 practical sentence describing trend
# =========================================================
def cluster_and_score(state: TrendState) -> TrendState:
    if not state.get("claims"):
        state["clusters"] = []
        return state

    texts = [c["claim"] for c in state["claims"]]
    X = np.array(emb.embed_documents(texts), dtype=np.float32)

    n = len(texts)
    k = min(10, max(4, n // 7))

    km = KMeans(n_clusters=k, n_init="auto", random_state=7)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    # Build raw cluster buckets
    clusters: Dict[int, Dict[str, Any]] = {}
    for label, c in zip(labels, state["claims"]):
        clusters.setdefault(label, {"claims": [], "sources": set(), "publish_dates": [], "trusts": []})
        clusters[label]["claims"].append(c)
        clusters[label]["sources"].add(c["url"])
        if c.get("publish_date"):
            clusters[label]["publish_dates"].append(c["publish_date"])
        clusters[label]["trusts"].append(float(c.get("trust", 0.0)))

    ranked: List[Dict[str, Any]] = []
    for label, obj in clusters.items():
        n_claims = len(obj["claims"])
        n_sources = len(obj["sources"])
        avg_trust = float(np.mean(obj["trusts"])) if obj["trusts"] else 0.0

        freshest = None
        if obj["publish_dates"]:
            try:
                freshest = max(dtparse(d) for d in obj["publish_dates"])
            except Exception:
                freshest = None
        fresh = freshness_score(freshest)

        consensus = min(1.0, n_sources / 5.0)
        volume = min(1.0, n_claims / 10.0)

        trend_score = (
            0.50 * consensus +
            0.30 * avg_trust +
            0.15 * volume +
            0.05 * fresh
        )

        # --- Representative claim: closest claim to centroid (trend-like)
        cluster_claim_idxs = [i for i, lab in enumerate(labels) if lab == label]
        if cluster_claim_idxs:
            centroid = centers[label]
            dists: List[tuple[float, int]] = []
            for i_idx in cluster_claim_idxs:
                v = X[i_idx]
                dists.append((float(np.linalg.norm(v - centroid)), i_idx))
            dists.sort(key=lambda x: x[0])
            rep_idx = dists[0][1]
            representative_claim = state["claims"][rep_idx]["claim"]
        else:
            representative_claim = obj["claims"][0]["claim"]

        ranked.append({
            "cluster_id": label,
            "trend_score": float(np.clip(trend_score, 0.0, 1.0)),
            "n_sources": n_sources,
            "n_claims": n_claims,
            "avg_trust": avg_trust,
            "freshness_bonus": fresh,
            "claims": obj["claims"],
            "sources": sorted(list(obj["sources"]))[:10],
            "representative_claim": representative_claim,
            "title": "",
            "summary": "",
        })

    ranked.sort(key=lambda x: x["trend_score"], reverse=True)

    # Label + summary with strict formatting rules
    for cl in ranked:
        sample = "\n".join([f"- {c['claim']}" for c in cl["claims"][:8]])

        label_prompt = f"""
You are naming an industry trend (NOT an article title).

Rules:
- 3 to 6 words
- noun phrase only (no "will", no verbs like "marks", no full sentences)
- no year numbers (no 2026)
- no colon ":" and no "The Year of"
- no hype words (revolution, transformation, game-changer)
- should sound like a dashboard tag

Claims:
{sample}

Return ONLY the label text.
"""
        title = llm.invoke(label_prompt).content.strip().strip('"').strip()

        # fallback if model still returns junk
        if ":" in title or any(x in title.lower() for x in ["the year of", "revolution", "transformation", "2026", "2025", "2027"]):
            title = " • ".join(cl["representative_claim"].split()[:6]).replace(" •", "")

        summary_prompt = f"""
Write ONE practical sentence describing the trend implied by these claims.
Rules: no year, no hype, be concrete.

Claims:
{sample}

Return only the sentence.
"""
        summary = llm.invoke(summary_prompt).content.strip()

        cl["title"] = title
        cl["summary"] = summary

    state["clusters"] = ranked
    return state


def write_report(state: TrendState) -> TrendState:
    clusters = state.get("clusters", [])[:10]
    packed = []
    for i, cl in enumerate(clusters, start=1):
        packed.append(
            f"""
Trend #{i}: {cl.get('title','(untitled)')}
Summary: {cl.get('summary','')}
Score: {cl['trend_score']:.2f} | Sources: {cl['n_sources']} | Claims: {cl['n_claims']} | AvgTrust: {cl['avg_trust']:.2f}

Representative claim:
- {cl.get("representative_claim","")}

Key claims:
{chr(10).join('- ' + c['claim'] for c in cl['claims'][:5])}

Sources:
{chr(10).join('- ' + u for u in cl['sources'][:6])}
""".strip()
        )

    prompt = f"""
Write a crisp briefing: "Best AI Predictions for 2026 (Clustered + Scored)".

Rules:
- Use the trend labels provided.
- For each trend: what it is, why it matters, signals to watch, what to do next.
- Include 2-4 source URLs from the cluster.
- Do not invent sources. Only use URLs given.

Data:
{chr(10).join(packed)}
"""
    state["report"] = llm.invoke(prompt).content.strip()
    return state


def should_continue(state: TrendState) -> str:
    if state.get("iterations_left", 0) <= 0:
        return "stop"
    if len(state.get("sources", [])) >= 24:
        return "stop"
    return "continue"


def decrement(state: TrendState) -> TrendState:
    state["iterations_left"] = state.get("iterations_left", 0) - 1
    return state


def build_graph():
    builder = StateGraph(TrendState)

    builder.add_node("plan", plan_search)
    builder.add_node("search", search_web)
    builder.add_node("dedupe", dedupe_sources)
    builder.add_node("decrement", decrement)
    builder.add_node("claims", extract_claims)
    builder.add_node("cluster_score", cluster_and_score)
    builder.add_node("report", write_report)

    builder.set_entry_point("plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "dedupe")
    builder.add_edge("dedupe", "decrement")

    builder.add_conditional_edges(
        "decrement",
        should_continue,
        {"continue": "plan", "stop": "claims"},
    )

    builder.add_edge("claims", "cluster_score")
    builder.add_edge("cluster_score", "report")
    builder.add_edge("report", END)

    return builder.compile()


def run_trends(topic: str, iterations_left: int = 2) -> TrendState:
    graph = build_graph()
    init: TrendState = {
        "topic": topic,
        "iterations_left": iterations_left,
        "queries": [],
        "sources": [],
        "claims": [],
        "clusters": [],
        "report": "",
        "debug_claim_extraction": [],
    }
    return graph.invoke(init)