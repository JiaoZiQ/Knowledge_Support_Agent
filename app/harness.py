import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.llm import AnswerGenerator
from app.schemas import ChatRequest, ChatResponse, SearchHit
from app.storage import AppStorage
from app.tools import ToolRegistry, serialize_hits


class HarnessState(TypedDict, total=False):
    request: ChatRequest
    started: float
    session_id: str
    profile: dict[str, Any]
    hits: list[SearchHit]
    action: str
    confidence: float
    ticket_id: str | None
    memory: str
    answer: str
    memory_summary: str
    trace_id: str
    latency_ms: float
    graph_steps: list[str]


class AgentHarness:
    def __init__(
        self,
        settings: Settings,
        storage: AppStorage,
        tools: ToolRegistry,
        answer_generator: AnswerGenerator,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.tools = tools
        self.answer_generator = answer_generator
        self.graph = self._build_graph()

    def chat(self, request: ChatRequest) -> ChatResponse:
        state = self.graph.invoke({"request": request, "started": time.perf_counter(), "graph_steps": []})
        hits = state.get("hits", [])
        citations = [
            {
                "id": hit.item.id,
                "category": hit.item.category,
                "title": hit.item.title,
                "score": hit.score,
            }
            for hit in hits[:2]
        ]
        return ChatResponse(
            session_id=state["session_id"],
            answer=state["answer"],
            action=state["action"],
            citations=citations,
            ticket_id=state.get("ticket_id"),
            trace_id=state["trace_id"],
            confidence=round(state.get("confidence", 0.0), 4),
            memory_summary=state.get("memory_summary"),
        )

    def _build_graph(self):
        graph = StateGraph(HarnessState)
        graph.add_node("prepare_session", self._prepare_session)
        graph.add_node("load_profile", self._load_profile)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("decide", self._decide)
        graph.add_node("create_ticket", self._create_ticket)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("persist_trace", self._persist_trace)

        graph.set_entry_point("prepare_session")
        graph.add_edge("prepare_session", "load_profile")
        graph.add_edge("load_profile", "retrieve")
        graph.add_edge("retrieve", "decide")
        graph.add_conditional_edges(
            "decide",
            self._route_after_decision,
            {"ticket": "create_ticket", "answer": "generate_answer"},
        )
        graph.add_edge("create_ticket", "generate_answer")
        graph.add_edge("generate_answer", "persist_trace")
        graph.add_edge("persist_trace", END)
        return graph.compile()

    def _prepare_session(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        session_id = self.storage.ensure_session(request.user_id, request.session_id)
        self.storage.add_message(session_id, "user", request.query)
        return {**state, "session_id": session_id, "graph_steps": [*state.get("graph_steps", []), "prepare_session"]}

    def _load_profile(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        profile = self.tools.call("get_user_profile", user_id=request.user_id).output
        return {**state, "profile": profile, "graph_steps": [*state.get("graph_steps", []), "load_profile"]}

    def _retrieve(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        search_result = self.tools.call(
            "search_knowledge_base",
            query=request.query,
            top_k=self.settings.retrieval_top_k,
        )
        hits: list[SearchHit] = search_result.output
        confidence = hits[0].score if hits else 0.0
        return {
            **state,
            "hits": hits,
            "confidence": confidence,
            "graph_steps": [*state.get("graph_steps", []), "retrieve"],
        }

    def _decide(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        hits = state.get("hits", [])
        action = self._decide_action(request.query, hits)
        return {**state, "action": action, "graph_steps": [*state.get("graph_steps", []), "decide"]}

    def _create_ticket(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        hits = state.get("hits", [])
        ticket_type = self._ticket_type(request.query, hits)
        priority = "urgent" if hits and hits[0].item.risk_level == "high" else "normal"
        ticket_result = self.tools.call(
            "create_ticket",
            user_id=request.user_id,
            session_id=state["session_id"],
            ticket_type=ticket_type,
            description=request.query,
            priority=priority,
            metadata={"top_hits": serialize_hits(hits[:3]), "profile": state.get("profile", {})},
        )
        return {
            **state,
            "ticket_id": ticket_result.output["ticket_id"],
            "graph_steps": [*state.get("graph_steps", []), "create_ticket"],
        }

    def _generate_answer(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        memory = self.storage.get_memory(state["session_id"])
        answer = self.answer_generator.generate(request.query, state.get("hits", []), state["action"], memory)
        if state.get("ticket_id"):
            answer += f"\n\n工单已创建：{state['ticket_id']}。"
        self.storage.add_message(state["session_id"], "assistant", answer)
        memory_summary = self.storage.update_memory(state["session_id"], request.query, state["action"])
        return {
            **state,
            "memory": memory,
            "answer": answer,
            "memory_summary": memory_summary,
            "graph_steps": [*state.get("graph_steps", []), "generate_answer"],
        }

    def _persist_trace(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        latency_ms = (time.perf_counter() - state["started"]) * 1000
        trace_data: dict[str, Any] = {
            "profile": state.get("profile", {}),
            "retrieval": serialize_hits(state.get("hits", [])),
            "ticket_id": state.get("ticket_id"),
            "guardrail": self._guardrail_note(state["action"], state.get("hits", [])),
            "graph_steps": state.get("graph_steps", []),
            "orchestrator": "langgraph",
        }
        trace_id = self.storage.add_trace(
            session_id=state["session_id"],
            user_query=request.query,
            action=state["action"],
            confidence=state.get("confidence", 0.0),
            data=trace_data,
            latency_ms=latency_ms,
        )
        return {
            **state,
            "trace_id": trace_id,
            "latency_ms": latency_ms,
            "graph_steps": [*state.get("graph_steps", []), "persist_trace"],
        }

    @staticmethod
    def _route_after_decision(state: HarnessState) -> str:
        return "ticket" if state["action"] in {"create_ticket", "answer_then_escalate"} else "answer"

    def _decide_action(self, query: str, hits: list[SearchHit]) -> str:
        query_lower = query.lower()
        if self._needs_clarifying_question(query):
            return "ask_clarifying_question"
        if any(word in query for word in ["微信客服", "电话客服"]) and "工单" in query:
            return "answer"
        if "冷静期" in query and any(word in query for word in ["恢复", "取消注销"]):
            return "answer"
        if any(word in query for word in ["人工", "真人", "客服", "投诉"]):
            return "create_ticket"
        if any(word in query for word in ["重复扣费", "退款", "退钱", "退费", "补偿"]) or (
            "退" in query and any(word in query for word in ["钱", "套餐", "会员"])
        ):
            return "create_ticket"
        if any(word in query for word in ["支付", "付款"]) and any(
            word in query for word in ["没有开通", "未开通", "没给我开通", "不能用"]
        ):
            return "answer_then_escalate"
        if "解析失败" in query and any(word in query for word in ["多次", "一直", "连续", "好几次"]):
            return "answer_then_escalate"
        if "50份" in query:
            return "answer"
        if any(word in query for word in ["保证", "成功率", "录用概率", "拿到offer", "找到工作", "腾讯", "阿里"]):
            return "answer_with_disclaimer"
        if any(word in query for word in ["他人", "朋友"]) and any(word in query for word in ["隐私", "简历", "个人信息"]):
            return "answer_with_warning"
        if any(word in query for word in ["技术架构", "系统架构", "内部实现"]):
            return "escalate_if_unknown"
        if any(word in query for word in ["GPT", "自研"]):
            return "create_ticket"
        if any(word in query_lower for word in ["legal", "medical", "financial"]):
            return "decline"
        if any(word in query for word in ["法律", "医疗", "投资", "理财", "心理健康"]):
            return "decline"
        if not hits or hits[0].score < self.settings.clarify_threshold:
            return "ask_clarifying_question"
        top = hits[0]
        if top.score < self.settings.answer_threshold:
            if any(word in query for word in ["退款", "扣费", "删除", "封禁", "申诉"]):
                return "create_ticket"
            return "ask_clarifying_question"
        if top.item.recommended_action == "answer_then_escalate":
            escalation_terms = ["多次", "一直", "仍然", "还是", "超过", "无法", "失败", "没有开通"]
            if not any(term in query for term in escalation_terms):
                return "answer"
        return top.item.recommended_action

    @staticmethod
    def _needs_clarifying_question(query: str) -> bool:
        stripped = query.strip()
        vague_exact = {
            "我想退款",
            "我付款了但是不能用",
            "我的简历解析有问题",
            "AI优化没有效果",
            "我账号有问题，帮我处理一下",
            "我刚才操作完页面没反应了",
            "我想问一下我的工单情况",
        }
        if stripped in vague_exact:
            return True
        if "简历" in query and any(term in query for term in ["传不上去", "上传不了", "上传不上去"]) and not any(
            term in query for term in ["一直", "多次", "好几次", "退款", "退钱"]
        ):
            return True
        if "付款了" in query and "不能用" in query and not any(term in query for term in ["支付成功", "扣款", "支付宝", "微信"]):
            return True
        if "工单" in query and "编号" not in query and any(term in query for term in ["情况", "进度", "状态"]):
            return True
        return False

    @staticmethod
    def _ticket_type(query: str, hits: list[SearchHit]) -> str:
        if any(word in query for word in ["退款", "扣费", "付款", "支付", "套餐", "会员"]):
            return "billing"
        if hits:
            return hits[0].item.category
        return "general"

    @staticmethod
    def _guardrail_note(action: str, hits: list[SearchHit]) -> str:
        if action == "create_ticket":
            return "High-risk or human-requested issue routed to ticket workflow."
        if action == "decline":
            return "Out-of-scope request declined by boundary policy."
        if action in {"clarify", "ask_clarifying_question"}:
            return "Low confidence or incomplete user context; asking a clarifying question."
        if hits and hits[0].item.risk_level == "high":
            return "High-risk knowledge item answered with warning/disclaimer policy."
        return "Answered from retrieved knowledge."
