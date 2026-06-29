import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="Knowledge Support Agent", page_icon="KS", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding: 1.2rem 2rem 2rem; max-width: 1280px; }
    [data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 14px;
        background: #ffffff;
    }
    .status-strip {
        border: 1px solid #d0d5dd;
        border-radius: 8px;
        padding: 12px 14px;
        background: #f8fafc;
        color: #344054;
        font-size: 0.92rem;
    }
    .section-title { font-weight: 700; font-size: 1.02rem; margin-bottom: .35rem; }
    .muted { color: #667085; font-size: .86rem; }
    div[data-testid="stChatMessage"] { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any] | None = None, timeout: int = 90) -> Any:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def latest_first_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        user_message = messages[index]
        assistant_message = messages[index + 1] if index + 1 < len(messages) else None
        pairs.append({"user": user_message, "assistant": assistant_message})
        index += 2
    return list(reversed(pairs))


if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None


st.title("Knowledge Support Agent")
st.caption("LangGraph workflow, Chroma RAG, guardrails, structured routing, tickets, trace, memory, eval")

try:
    health = api_get("/health")
    api_ready = True
except Exception as exc:  # noqa: BLE001
    health = {"error": str(exc)}
    api_ready = False

if not api_ready:
    st.error("API is not connected.")
    st.code(f"uvicorn app.main:app --reload\n\n{health['error']}")
    st.stop()

st.markdown(
    f"""
    <div class="status-strip">
    API connected · orchestrator: <b>{health['orchestrator']}</b> · vector store: <b>{health['vector_store']}</b> ·
    embedding: <b>{health['embedding_provider']}</b> · router: <b>{health['router_mode']}</b> · model: <b>{health['chat_model']}</b>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Knowledge items", health["knowledge_items"])
metric_cols[1].metric("Eval cases", health.get("eval_cases", 0))
metric_cols[2].metric("LLM", "On" if health["llm_enabled"] else "Fallback")
metric_cols[3].metric("Session", st.session_state.session_id or "New")

left, right = st.columns([0.62, 0.38], gap="large")

with left:
    st.markdown('<div class="section-title">Support Chat</div>', unsafe_allow_html=True)
    examples = [
        "免费版和专业版有什么区别？",
        "这个怎么弄？",
        "支付成功但是会员没有开通，订单号是 A123",
        "帮我判断劳动合同纠纷该怎么起诉",
        "我要找人工客服处理投诉",
    ]
    selected = st.selectbox("Demo question", [""] + examples)
    query = st.text_area("User query", value=selected, height=88, placeholder="输入一个客服问题")

    action_cols = st.columns([0.24, 0.24, 0.52])
    send = action_cols[0].button("Send", type="primary", use_container_width=True)
    reset = action_cols[1].button("New session", use_container_width=True)
    action_cols[2].caption("Newest turns are shown at the top. High-risk requests create tickets or decline safely.")

    if reset:
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.rerun()

    if send and query.strip():
        payload = {
            "query": query.strip(),
            "user_id": "demo_user",
            "session_id": st.session_state.session_id,
        }
        with st.spinner("Retrieving, routing, and generating response..."):
            response = api_post("/chat", payload)
        st.session_state.session_id = response["session_id"]
        st.session_state.chat_history.append({"role": "user", "content": query.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"], "raw": response})

    for pair in latest_first_pairs(st.session_state.chat_history):
        for message in [pair["user"], pair["assistant"]]:
            if not message:
                continue
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "raw" in message:
                    raw = message["raw"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.caption(f"action: {raw['action']}")
                    c2.caption(f"intent: {raw.get('intent', 'unknown')}")
                    c3.caption(f"source: {raw.get('routing_source', 'unknown')}")
                    c4.caption(f"confidence: {raw['confidence']}")
                    st.caption(f"trace: {raw['trace_id']}")
                    if raw.get("citations"):
                        st.dataframe(raw["citations"], hide_index=True, use_container_width=True)
        st.divider()

with right:
    tab_trace, tab_eval, tab_tickets = st.tabs(["Trace", "Eval", "Tickets"])

    with tab_trace:
        if st.session_state.session_id:
            trace = api_get(f"/sessions/{st.session_state.session_id}")
            st.markdown('<div class="section-title">Memory</div>', unsafe_allow_html=True)
            st.info(trace["session"]["memory_summary"] or "No memory summary yet.")
            if trace["traces"]:
                latest = trace["traces"][0]
                st.markdown('<div class="section-title">Decision</div>', unsafe_allow_html=True)
                st.json(
                    {
                        "action": latest["action"],
                        "confidence": latest["confidence"],
                        "latency_ms": round(latest["latency_ms"], 2),
                        "routing": latest["data"].get("routing", {}),
                        "routing_source": latest["data"].get("routing_source"),
                        "graph_steps": latest["data"].get("graph_steps", []),
                        "errors": latest["data"].get("errors", []),
                    },
                    expanded=True,
                )
                st.markdown('<div class="section-title">Retrieval</div>', unsafe_allow_html=True)
                st.dataframe(latest["data"]["retrieval"], hide_index=True, use_container_width=True)
        else:
            st.info("Send a question to see graph steps, retrieval hits, routing, and trace data.")

    with tab_eval:
        max_eval_cases = int(health.get("eval_cases", 1))
        limit = st.slider("Eval cases", min_value=1, max_value=max_eval_cases, value=min(20, max_eval_cases), step=1)
        if st.button("Run legacy eval", use_container_width=True):
            with st.spinner("Running evaluation..."):
                st.session_state.last_eval = api_post(f"/eval/run?limit={int(limit)}", timeout=180)
        if st.session_state.last_eval:
            result = st.session_state.last_eval
            cols = st.columns(3)
            cols[0].metric("Action accuracy", f"{result['action_accuracy']:.1%}")
            cols[1].metric("Category hit", f"{result['category_hit_rate']:.1%}")
            cols[2].metric("Refusal precision", f"{result['refusal_precision']:.1%}")
            st.caption(f"Average latency: {result['average_latency_ms']} ms")
            st.dataframe(result["cases"], hide_index=True, use_container_width=True)

    with tab_tickets:
        tickets = api_get("/tickets?limit=20")
        if tickets:
            st.dataframe(tickets, hide_index=True, use_container_width=True)
        else:
            st.caption("No tickets yet.")
