import os
import sys
import warnings

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agent import run_trends

# -----------------------------
# Optional: silence noisy warnings from langchain_tavily
# -----------------------------
warnings.filterwarnings("ignore", message='Field name "output_schema"')
warnings.filterwarnings("ignore", message='Field name "stream"')

# -----------------------------
# Load .env (app-level). Agent.py also loads it, but this helps for UI checks.
# -----------------------------
load_dotenv()

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="AI Trends 2026 — Agentic Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AI Trends 2026 — Agentic Research Dashboard")
st.caption("LangGraph + Tavily + OpenAI • clustering + scoring • runs locally on your machine")

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    topic = st.text_input(
        "Topic",
        value="AI predictions and trends for the year 2026",
    )

    iterations = st.slider(
        "Agent iterations (deeper = more sources/cost)",
        min_value=1,
        max_value=4,
        value=2,
    )

    top_k = st.slider(
        "Show top trends",
        min_value=5,
        max_value=15,
        value=10,
    )

    run_btn = st.button("Run agent", type="primary")

    st.divider()
    st.subheader("Environment")

    st.write("Python executable:")
    st.code(sys.executable)

    st.subheader("Keys check")
    st.write("OPENAI_API_KEY:", "✅" if os.getenv("OPENAI_API_KEY") else "❌")
    st.write("TAVILY_API_KEY:", "✅" if os.getenv("TAVILY_API_KEY") else "❌")

# -----------------------------
# Session state
# -----------------------------
if "result" not in st.session_state:
    st.session_state.result = None

# -----------------------------
# Run agent on click
# -----------------------------
if run_btn:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing. Please add it to your .env file.")
    elif not os.getenv("TAVILY_API_KEY"):
        st.error("TAVILY_API_KEY is missing. Please add it to your .env file.")
    else:
        with st.spinner("Running agent… searching, extracting claims, clustering, scoring…"):
            st.session_state.result = run_trends(topic=topic, iterations_left=iterations)

result = st.session_state.result

if not result:
    st.info("Click **Run agent** in the sidebar to generate clustered + scored trends.")
    st.stop()

# -----------------------------
# Pull data safely
# -----------------------------
clusters_all = result.get("clusters", []) or []
clusters = clusters_all[:top_k]
sources = result.get("sources", []) or []
claims = result.get("claims", []) or []

# -----------------------------
# KPIs
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sources collected", len(sources))
c2.metric("Claims extracted", len(claims))
c3.metric("Clusters found", len(clusters_all))
c4.metric("Top trends shown", len(clusters))

st.divider()

# -----------------------------
# If no clusters, show diagnostics and stop early
# -----------------------------
if len(clusters_all) == 0:
    st.warning("No trends/clusters found yet.")
    st.write("This usually means **the agent hasn't run** or **0 claims were extracted**.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### What to do")
        st.markdown(
            "- Click **Run agent** in the sidebar\n"
            "- Increase **iterations** to 3\n"
            "- Ensure both keys show ✅ in the sidebar\n"
            "- Confirm Tavily returns results (not blocked)\n"
        )

    with colB:
        st.markdown("### Quick diagnostics")
        st.write("Sources (first 5):")
        st.write(sources[:5])
        st.write("Claims (first 5):")
        st.write(claims[:5])

    st.stop()

# -----------------------------
# Ranked Trends Table
# -----------------------------
st.subheader("📌 Ranked trends (clustered + scored)")

trend_rows = []
for i, cl in enumerate(clusters, start=1):
    trend_rows.append(
        {
            "Rank": i,
            "Trend": cl.get("title", ""),
            "Score": round(float(cl.get("trend_score", 0.0)), 3),
            "Sources": int(cl.get("n_sources", 0)),
            "Claims": int(cl.get("n_claims", 0)),
            "Avg trust": round(float(cl.get("avg_trust", 0.0)), 3),
        }
    )

df_trends = pd.DataFrame(trend_rows)
if not df_trends.empty and "Score" in df_trends.columns:
    df_trends = df_trends.sort_values("Score", ascending=False)

st.dataframe(df_trends, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# Trend Explorer
# -----------------------------
st.subheader("🔎 Trend explorer")

trend_names = [f"#{i+1} — {cl.get('title','')}" for i, cl in enumerate(clusters)]
selected = st.selectbox("Select a trend", trend_names, index=0)

idx = trend_names.index(selected)
chosen = clusters[idx]

left, right = st.columns([1, 1])

with left:
    st.markdown(f"### {chosen.get('title','')}")
    st.write(
        f"**Score:** {chosen.get('trend_score', 0.0):.2f}  •  "
        f"**Sources:** {chosen.get('n_sources', 0)}  •  "
        f"**Claims:** {chosen.get('n_claims', 0)}  •  "
        f"**Avg trust:** {chosen.get('avg_trust', 0.0):.2f}"
    )

    st.markdown("#### Key claims")
    for c in chosen.get("claims", [])[:10]:
        st.write("•", c.get("claim", ""))

with right:
    st.markdown("#### Supporting sources")
    for u in chosen.get("sources", [])[:10]:
        st.link_button(u, u)

st.divider()

# -----------------------------
# Full report
# -----------------------------
with st.expander("📝 Generated briefing (final report)", expanded=True):
    st.write(result.get("report", ""))

st.divider()

# -----------------------------
# Raw tables (optional)
# -----------------------------
with st.expander("🧾 Raw data (sources + claims)"):
    st.markdown("##### Sources")
    st.dataframe(pd.DataFrame(sources), use_container_width=True)

    st.markdown("##### Claims")
    st.dataframe(pd.DataFrame(claims), use_container_width=True)