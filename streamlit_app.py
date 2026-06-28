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
    pairs = []
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
st.caption("LangGraph orchestration, Chroma retrieval, tool calling, human handoff, trace, memory, eval")

try:
    health = api_get("/health")
    api_ready = True
except Exception as exc:  # noqa: BLE001
    health = {"error": str(exc)}
    api_ready = False

if api_ready:
    st.markdown(
        f"""
        <div class="status-strip">
        API connected · orchestrator: <b>{health['orchestrator']}</b> · vector store: <b>{health['vector_store']}</b> ·
        embedding: <b>{health['embedding_provider']}</b> · model: <b>{health['chat_model']}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.error("API 未连接")
    st.code(f"cd F:\\final_intern\\knowledge_support_agent\nuvicorn app.main:app --reload\n\n{health['error']}")
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Knowledge items", health["knowledge_items"])
metric_cols[1].metric("Categories", len(health["categories"]))
metric_cols[2].metric("LLM", "On" if health["llm_enabled"] else "Fallback")
metric_cols[3].metric("Session", st.session_state.session_id or "New")

left, right = st.columns([0.62, 0.38], gap="large")

with left:
    st.markdown('<div class="section-title">客服对话</div>', unsafe_allow_html=True)
    examples = [
        "免费版和专业版有什么区别？",
        "我付款成功了，支付宝也显示扣款了，但是平台没给我开通会员",
        "我注册了账号，买了套餐，但是我的简历一直上传不了，这个钱还能退吗？",
        "我是公司HR，想批量上传1000份候选人简历进行分析",
        "你能告诉我你的系统是用什么技术架构搭建的吗？",
    ]
    selected = st.selectbox("演示问题", [""] + examples)
    query = st.text_area("用户问题", value=selected, height=88, placeholder="输入客服问题")

    action_cols = st.columns([0.24, 0.24, 0.52])
    send = action_cols[0].button("发送", type="primary", use_container_width=True)
    reset = action_cols[1].button("新会话", use_container_width=True)
    action_cols[2].caption("最新问题会显示在最上方；高风险问题会触发工单并写入 trace。")

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
        with st.spinner("Agent 正在检索、决策并生成回答..."):
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
                    c1, c2, c3 = st.columns(3)
                    c1.caption(f"action: {raw['action']}")
                    c2.caption(f"confidence: {raw['confidence']}")
                    c3.caption(f"trace: {raw['trace_id']}")
                    if raw.get("citations"):
                        st.dataframe(raw["citations"], hide_index=True, use_container_width=True)
        st.divider()

with right:
    tab_trace, tab_eval, tab_tickets = st.tabs(["Trace", "Eval", "Tickets"])

    with tab_trace:
        if st.session_state.session_id:
            trace = api_get(f"/sessions/{st.session_state.session_id}")
            st.markdown('<div class="section-title">Memory</div>', unsafe_allow_html=True)
            st.info(trace["session"]["memory_summary"] or "暂无")
            if trace["traces"]:
                latest = trace["traces"][0]
                st.markdown('<div class="section-title">Decision</div>', unsafe_allow_html=True)
                st.json(
                    {
                        "action": latest["action"],
                        "confidence": latest["confidence"],
                        "latency_ms": round(latest["latency_ms"], 2),
                        "guardrail": latest["data"]["guardrail"],
                        "graph_steps": latest["data"].get("graph_steps", []),
                    },
                    expanded=True,
                )
                st.markdown('<div class="section-title">Retrieval</div>', unsafe_allow_html=True)
                st.dataframe(latest["data"]["retrieval"], hide_index=True, use_container_width=True)
        else:
            st.info("发送一条问题后会显示 LangGraph 节点、检索结果和 guardrail。")

    with tab_eval:
        limit = st.slider("Eval cases", min_value=5, max_value=59, value=20, step=1)
        if st.button("运行评估", use_container_width=True):
            with st.spinner("运行评估中..."):
                st.session_state.last_eval = api_post(f"/eval/run?limit={int(limit)}", timeout=180)
        if st.session_state.last_eval:
            result = st.session_state.last_eval
            cols = st.columns(3)
            cols[0].metric("Action accuracy", f"{result['action_accuracy']:.1%}")
            cols[1].metric("Category hit", f"{result['category_hit_rate']:.1%}")
            cols[2].metric("Refusal precision", f"{result['refusal_precision']:.1%}")
            st.caption(f"平均延迟：{result['average_latency_ms']} ms")
            st.dataframe(result["cases"], hide_index=True, use_container_width=True)

    with tab_tickets:
        tickets = api_get("/tickets?limit=20")
        if tickets:
            st.dataframe(tickets, hide_index=True, use_container_width=True)
        else:
            st.caption("暂无工单。")
