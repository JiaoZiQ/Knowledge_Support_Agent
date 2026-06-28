import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="Knowledge Support Agent", page_icon="KS", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; max-width: 1180px; }
    .small-muted { color: #667085; font-size: 0.86rem; }
    .metric-row { border: 1px solid #e4e7ec; border-radius: 8px; padding: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any] | None = None) -> Any:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload or {}, timeout=60)
    response.raise_for_status()
    return response.json()


if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_trace_id" not in st.session_state:
    st.session_state.last_trace_id = None


st.title("Knowledge Support Agent")
st.caption("RAG + Tool Calling + Human Handoff + Trace + Eval")

with st.sidebar:
    st.subheader("运行状态")
    try:
        health = api_get("/health")
        st.success("API connected")
        st.metric("知识条目", health["knowledge_items"])
        st.json(health["categories"], expanded=False)
    except Exception as exc:  # noqa: BLE001
        st.error("API 未连接")
        st.code(f"uvicorn app.main:app --reload\n\n{exc}")

    st.divider()
    st.subheader("评估")
    limit = st.number_input("Eval cases", min_value=5, max_value=59, value=15, step=5)
    if st.button("运行评估", use_container_width=True):
        try:
            result = api_post(f"/eval/run?limit={int(limit)}")
            st.metric("Action accuracy", f"{result['action_accuracy']:.0%}")
            st.metric("Category hit rate", f"{result['category_hit_rate']:.0%}")
            st.metric("Refusal precision", f"{result['refusal_precision']:.0%}")
            st.caption(f"平均延迟：{result['average_latency_ms']} ms")
            st.dataframe(result["cases"], hide_index=True, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"评估失败：{exc}")

left, right = st.columns([0.64, 0.36], gap="large")

with left:
    st.subheader("客服对话")
    examples = [
        "免费版和专业版有什么区别？",
        "我的 PDF 简历上传成功了，但是解析失败怎么办？",
        "我被重复扣费了，可以直接退款吗？",
        "你能保证我拿到 offer 吗？",
        "我想问劳动合同纠纷的法律问题。",
    ]
    selected = st.selectbox("示例问题", [""] + examples)
    query = st.text_area("用户问题", value=selected, height=94, placeholder="输入一个客服问题")

    col_a, col_b = st.columns([0.25, 0.75])
    with col_a:
        send = st.button("发送", type="primary", use_container_width=True)
    with col_b:
        if st.button("新会话", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.chat_history = []
            st.session_state.last_trace_id = None
            st.rerun()

    if send and query.strip():
        payload = {
            "query": query.strip(),
            "user_id": "demo_user",
            "session_id": st.session_state.session_id,
        }
        try:
            response = api_post("/chat", payload)
            st.session_state.session_id = response["session_id"]
            st.session_state.last_trace_id = response["trace_id"]
            st.session_state.chat_history.append({"role": "user", "content": query.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": response["answer"], "raw": response})
        except Exception as exc:  # noqa: BLE001
            st.error(f"请求失败：{exc}")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "raw" in message:
                raw = message["raw"]
                st.caption(
                    f"action={raw['action']} | confidence={raw['confidence']} | trace={raw['trace_id']}"
                )
                if raw.get("citations"):
                    st.dataframe(raw["citations"], hide_index=True, use_container_width=True)

with right:
    st.subheader("Trace")
    if st.session_state.session_id:
        try:
            trace = api_get(f"/sessions/{st.session_state.session_id}")
            st.caption(f"Session: {st.session_state.session_id}")
            st.write("Memory")
            st.info(trace["session"]["memory_summary"] or "暂无")
            if trace["traces"]:
                latest = trace["traces"][0]
                st.write("Latest decision")
                st.json(
                    {
                        "action": latest["action"],
                        "confidence": latest["confidence"],
                        "latency_ms": latest["latency_ms"],
                        "guardrail": latest["data"]["guardrail"],
                    },
                    expanded=True,
                )
                st.write("Retrieval")
                st.dataframe(latest["data"]["retrieval"], hide_index=True, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Trace 加载失败：{exc}")
    else:
        st.info("发送一条问题后会显示会话 trace。")

    st.divider()
    st.subheader("工单")
    try:
        tickets = api_get("/tickets?limit=10")
        if tickets:
            st.dataframe(tickets, hide_index=True, use_container_width=True)
        else:
            st.caption("暂无工单。")
    except Exception as exc:  # noqa: BLE001
        st.error(f"工单加载失败：{exc}")
