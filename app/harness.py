import time
from typing import Any

from app.config import Settings
from app.llm import AnswerGenerator
from app.schemas import ChatRequest, ChatResponse, SearchHit
from app.storage import AppStorage
from app.tools import ToolRegistry, serialize_hits


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

    def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        session_id = self.storage.ensure_session(request.user_id, request.session_id)
        self.storage.add_message(session_id, "user", request.query)

        profile = self.tools.call("get_user_profile", user_id=request.user_id).output
        search_result = self.tools.call(
            "search_knowledge_base",
            query=request.query,
            top_k=self.settings.retrieval_top_k,
        )
        hits: list[SearchHit] = search_result.output
        action = self._decide_action(request.query, hits)
        confidence = hits[0].score if hits else 0.0

        ticket_id = None
        if action in {"create_ticket", "answer_then_escalate"}:
            ticket_type = hits[0].item.category if hits else "general"
            priority = "urgent" if hits and hits[0].item.risk_level == "high" else "normal"
            ticket_result = self.tools.call(
                "create_ticket",
                user_id=request.user_id,
                session_id=session_id,
                ticket_type=ticket_type,
                description=request.query,
                priority=priority,
                metadata={"top_hits": serialize_hits(hits[:3]), "profile": profile},
            )
            ticket_id = ticket_result.output["ticket_id"]

        memory = self.storage.get_memory(session_id)
        answer = self.answer_generator.generate(request.query, hits, action, memory)
        if ticket_id:
            answer += f"\n\n工单已创建：{ticket_id}。"

        self.storage.add_message(session_id, "assistant", answer)
        memory_summary = self.storage.update_memory(session_id, request.query, action)
        latency_ms = (time.perf_counter() - started) * 1000
        trace_data: dict[str, Any] = {
            "profile": profile,
            "retrieval": serialize_hits(hits),
            "ticket_id": ticket_id,
            "guardrail": self._guardrail_note(action, hits),
        }
        trace_id = self.storage.add_trace(
            session_id=session_id,
            user_query=request.query,
            action=action,
            confidence=confidence,
            data=trace_data,
            latency_ms=latency_ms,
        )

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
            session_id=session_id,
            answer=answer,
            action=action,
            citations=citations,
            ticket_id=ticket_id,
            trace_id=trace_id,
            confidence=round(confidence, 4),
            memory_summary=memory_summary,
        )

    def _decide_action(self, query: str, hits: list[SearchHit]) -> str:
        query_lower = query.lower()
        if any(word in query for word in ["人工", "真人", "客服", "投诉"]):
            return "create_ticket"
        if any(word in query for word in ["重复扣费", "退款", "退钱", "退费"]):
            return "create_ticket"
        if any(word in query for word in ["支付", "付款"]) and any(word in query for word in ["没有开通", "未开通", "没给我开通", "不能用"]):
            return "answer_then_escalate"
        if "解析失败" in query and any(word in query for word in ["多次", "一直", "连续", "好几次"]):
            return "answer_then_escalate"
        if any(word in query for word in ["保证", "成功率", "录用概率", "拿到offer", "找到工作", "腾讯", "阿里"]):
            return "answer_with_disclaimer"
        if any(word in query for word in ["他人", "朋友"]) and any(word in query for word in ["隐私", "简历", "个人信息"]):
            return "answer_with_warning"
        if any(word in query for word in ["技术架构", "系统架构", "内部实现"]):
            return "create_ticket"
        if not hits or hits[0].score < self.settings.clarify_threshold:
            return "clarify"
        top = hits[0]
        if top.score < self.settings.answer_threshold:
            if any(word in query for word in ["退款", "扣费", "删除", "封禁", "申诉"]):
                return "create_ticket"
            return "clarify"
        if any(word in query_lower for word in ["legal", "medical", "financial"]):
            return "decline"
        if any(word in query for word in ["法律", "医疗", "投资", "理财", "心理健康"]):
            return "decline"
        if top.item.recommended_action == "answer_then_escalate":
            escalation_terms = ["多次", "一直", "仍然", "还是", "超过", "无法", "失败", "没有开通"]
            if not any(term in query for term in escalation_terms):
                return "answer"
        return top.item.recommended_action

    @staticmethod
    def _guardrail_note(action: str, hits: list[SearchHit]) -> str:
        if action == "create_ticket":
            return "High-risk or human-requested issue routed to ticket workflow."
        if action == "decline":
            return "Out-of-scope request declined by boundary policy."
        if action == "clarify":
            return "Low confidence or incomplete user context; asking a clarifying question."
        if hits and hits[0].item.risk_level == "high":
            return "High-risk knowledge item answered with warning/disclaimer policy."
        return "Answered from retrieved knowledge."
