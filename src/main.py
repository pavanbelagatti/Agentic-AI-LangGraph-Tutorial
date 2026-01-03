from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

# --------------------------------
# 1. Agent State
# --------------------------------
class TrendState(TypedDict):
    topic: str
    iterations_left: int
    notes: List[str]
    queries: List[str]
    sources: List[Dict[str, Any]]
    report: str

# --------------------------------
# 2. Tools & Model
# --------------------------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
)

search_tool = TavilySearch(
    max_results=6,
    search_depth="advanced",
)

# --------------------------------
# 3. Nodes
# --------------------------------
def plan_search(state: TrendState) -> TrendState:
    prompt = f"""
You are researching: {state['topic']}

Based on what you know so far:
{state['notes']}

Generate 3 focused search queries to find high-quality AI predictions for 2026.
Prefer analyst firms, major AI companies, and reputable tech journalism.
Return only the queries as bullet points.
"""
    response = llm.invoke(prompt).content

    queries = [q.strip("- ").strip() for q in response.splitlines() if q.strip()]
    state["queries"].extend(queries)

    return state


def search_web(state: TrendState) -> TrendState:
    for query in state["queries"][-3:]:
        results = search_tool.invoke(query)

        for r in results.get("results", []):
            state["sources"].append({
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content", "")[:500],
            })

        state["notes"].append(f"Searched: {query}")

    return state


def extract_trends(state: TrendState) -> TrendState:
    context = "\n".join(
        f"- {s['title']} ({s['url']}): {s['content']}"
        for s in state["sources"][:20]
    )

    prompt = f"""
From the sources below, extract clear AI predictions or trends for 2026.

Rules:
- Only include forward-looking claims
- Group similar ideas
- Ignore marketing fluff

Sources:
{context}
"""
    summary = llm.invoke(prompt).content
    state["notes"].append(summary)
    return state


def should_continue(state: TrendState) -> str:
    if state["iterations_left"] <= 0:
        return "stop"
    return "continue"


def decrement(state: TrendState) -> TrendState:
    state["iterations_left"] -= 1
    return state


def write_report(state: TrendState) -> TrendState:
    prompt = f"""
Using the extracted insights below, write a ranked report:

Title: Best AI Predictions for 2026

Include:
- Top 8–12 trends
- Why each matters
- Supporting sources (URLs)
- A short "What to do next" section

Insights:
{state['notes']}
"""
    state["report"] = llm.invoke(prompt).content
    return state

# --------------------------------
# 4. LangGraph Definition
# --------------------------------
builder = StateGraph(TrendState)

builder.add_node("plan", plan_search)
builder.add_node("search", search_web)
builder.add_node("extract", extract_trends)
builder.add_node("decrement", decrement)
builder.add_node("report", write_report)

builder.set_entry_point("plan")

builder.add_edge("plan", "search")
builder.add_edge("search", "extract")
builder.add_edge("extract", "decrement")

builder.add_conditional_edges(
    "decrement",
    should_continue,
    {
        "continue": "plan",
        "stop": "report",
    },
)

builder.add_edge("report", END)

graph = builder.compile()

# --------------------------------
# 5. Run
# --------------------------------
if __name__ == "__main__":
    initial_state: TrendState = {
        "topic": "AI predictions and trends for the year 2026",
        "iterations_left": 2,
        "notes": [],
        "queries": [],
        "sources": [],
        "report": "",
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 80)
    print(result["report"])
    print("=" * 80)