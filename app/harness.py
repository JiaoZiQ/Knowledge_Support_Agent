import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.llm import AnswerGenerator
from app.observability.tracing import build_trace_payload, redact_text
from app.policies.escalation import ConfidencePolicy
from app.policies.guardrails import GuardrailPolicy
from app.policies.intent_router import IntentRouter
from app.schemas import ChatRequest, ChatResponse, SearchHit
from app.schemas.routing import RoutingDecision
from app.storage import AppStorage
from app.tools import ToolRegistry


class HarnessState(TypedDict, total=False):
    request: ChatRequest
    started: float
    session_id: str
    profile: dict[str, Any]
    hits: list[SearchHit]
    guardrail_decision: RoutingDecision | None
    route_decision: RoutingDecision
    final_decision: RoutingDecision
    ticket_id: str | None
    memory: str
    answer: str
    memory_summary: str
    trace_id: str
    latency_ms: float
    graph_steps: list[str]
    errors: list[str]


class AgentHarness:
    def __init__(
        self,
        settings: Settings,
        storage: AppStorage,
        tools: ToolRegistry,
        answer_generator: AnswerGenerator,
        guardrails: GuardrailPolicy,
        intent_router: IntentRouter,
        confidence_policy: ConfidencePolicy,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.tools = tools
        self.answer_generator = answer_generator
        self.guardrails = guardrails
        self.intent_router = intent_router
        self.confidence_policy = confidence_policy
        self.graph = self._build_graph()

    def chat(self, request: ChatRequest) -> ChatResponse:
        state = self.graph.invoke(
            {"request": request, "started": time.perf_counter(), "graph_steps": [], "errors": []}
        )
        hits = state.get("hits", [])
        decision = state["final_decision"]
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
            action=decision.action,
            citations=citations,
            ticket_id=state.get("ticket_id"),
            trace_id=state["trace_id"],
            confidence=round(decision.confidence, 4),
            intent=decision.intent,
            routing_source=decision.source,
            memory_summary=state.get("memory_summary"),
        )

    def _build_graph(self):
        graph = StateGraph(HarnessState)
        graph.add_node("prepare_session", self._prepare_session)
        graph.add_node("load_profile", self._load_profile)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("guardrail_check", self._guardrail_check)
        graph.add_node("route_intent", self._route_intent)
        graph.add_node("confidence_check", self._confidence_check)
        graph.add_node("create_ticket", self._create_ticket)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("persist_trace", self._persist_trace)

        graph.set_entry_point("prepare_session")
        graph.add_edge("prepare_session", "load_profile")
        graph.add_edge("load_profile", "retrieve")
        graph.add_edge("retrieve", "guardrail_check")
        graph.add_edge("guardrail_check", "route_intent")
        graph.add_edge("route_intent", "confidence_check")
        graph.add_conditional_edges(
            "confidence_check",
            self._route_after_confidence,
            {"ticket": "create_ticket", "respond": "generate_response"},
        )
        graph.add_edge("create_ticket", "generate_response")
        graph.add_edge("generate_response", "persist_trace")
        graph.add_edge("persist_trace", END)
        return graph.compile()

    def _add_step(self, state: HarnessState, name: str) -> list[str]:
        return [*state.get("graph_steps", []), name]

    def _prepare_session(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        session_id = self.storage.ensure_session(request.user_id, request.session_id)
        self.storage.add_message(session_id, "user", request.query)
        return {**state, "session_id": session_id, "graph_steps": self._add_step(state, "prepare_session")}

    def _load_profile(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        profile = self.tools.call("get_user_profile", user_id=request.user_id).output
        return {**state, "profile": profile, "graph_steps": self._add_step(state, "load_profile")}

    def _retrieve(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        search_result = self.tools.call(
            "search_knowledge_base",
            query=request.query,
            top_k=self.settings.retrieval_top_k,
        )
        hits: list[SearchHit] = search_result.output
        return {**state, "hits": hits, "graph_steps": self._add_step(state, "retrieve")}

    def _guardrail_check(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        decision = self.guardrails.evaluate(request.query)
        return {
            **state,
            "guardrail_decision": decision,
            "graph_steps": self._add_step(state, "guardrail_check"),
        }

    def _route_intent(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        guardrail_decision = state.get("guardrail_decision")
        decision = guardrail_decision or self.intent_router.route(request.query)
        errors = state.get("errors", [])
        if self.intent_router.last_error:
            errors = [*errors, self.intent_router.last_error]
        return {
            **state,
            "route_decision": decision,
            "errors": errors,
            "graph_steps": self._add_step(state, "route_intent"),
        }

    def _confidence_check(self, state: HarnessState) -> HarnessState:
        decision = self.confidence_policy.apply(state["route_decision"], state.get("hits", []))
        return {
            **state,
            "final_decision": decision,
            "graph_steps": self._add_step(state, "confidence_check"),
        }

    @staticmethod
    def _route_after_confidence(state: HarnessState) -> str:
        action = state["final_decision"].action
        return "ticket" if action in {"create_ticket", "answer_then_escalate"} else "respond"

    def _create_ticket(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        hits = state.get("hits", [])
        decision = state["final_decision"]
        ticket_type = self._ticket_type(decision.intent, hits)
        priority = "urgent" if hits and hits[0].item.risk_level == "high" else "normal"
        ticket_result = self.tools.call(
            "create_ticket",
            user_id=request.user_id,
            session_id=state["session_id"],
            ticket_type=ticket_type,
            description=redact_text(request.query),
            priority=priority,
            metadata={
                "routing": decision.model_dump(),
                "top_hit_ids": [hit.item.id for hit in hits[:3]],
            },
        )
        return {
            **state,
            "ticket_id": ticket_result.output["ticket_id"],
            "graph_steps": self._add_step(state, "create_ticket"),
        }

    def _generate_response(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        decision = state["final_decision"]
        memory = self.storage.get_memory(state["session_id"])
        answer = self.answer_generator.generate(request.query, state.get("hits", []), decision.action, memory)
        if state.get("ticket_id"):
            answer += f"\n\n工单已创建：{state['ticket_id']}。"
        self.storage.add_message(state["session_id"], "assistant", answer)
        memory_summary = self.storage.update_memory(state["session_id"], request.query, decision.action)
        return {
            **state,
            "memory": memory,
            "answer": answer,
            "memory_summary": memory_summary,
            "graph_steps": self._add_step(state, "generate_response"),
        }

    def _persist_trace(self, state: HarnessState) -> HarnessState:
        request = state["request"]
        decision = state["final_decision"]
        latency_ms = (time.perf_counter() - state["started"]) * 1000
        trace_data = build_trace_payload(
            profile=state.get("profile", {}),
            hits=state.get("hits", []),
            decision=decision,
            ticket_id=state.get("ticket_id"),
            graph_steps=state.get("graph_steps", []),
            errors=state.get("errors", []),
        )
        trace_id = self.storage.add_trace(
            session_id=state["session_id"],
            user_query=redact_text(request.query),
            action=decision.action,
            confidence=decision.confidence,
            data=trace_data,
            latency_ms=latency_ms,
        )
        return {
            **state,
            "trace_id": trace_id,
            "latency_ms": latency_ms,
            "graph_steps": self._add_step(state, "persist_trace"),
        }

    @staticmethod
    def _ticket_type(intent: str, hits: list[SearchHit]) -> str:
        if intent in {"billing_issue", "refund_request"}:
            return "billing"
        if intent == "privacy_request":
            return "privacy"
        if intent == "technical_issue":
            return "technical"
        if hits:
            return hits[0].item.category
        return "general"
