from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.routing import Action, Intent, RoutingDecision, RoutingSource


class KnowledgeItem(BaseModel):
    id: str
    category: str
    title: str
    content: str
    keywords: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"]
    recommended_action: Action
    source: str | None = None
    updated_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    requires_human_review: bool = False


class SearchHit(BaseModel):
    item: KnowledgeItem
    score: float
    matched_keywords: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: str = "demo_user"
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    action: Action
    citations: list[dict[str, Any]] = Field(default_factory=list)
    ticket_id: str | None = None
    trace_id: str
    confidence: float
    intent: Intent = "unknown"
    routing_source: RoutingSource = "fallback_rule"
    memory_summary: str | None = None


class TicketRequest(BaseModel):
    user_id: str = "demo_user"
    session_id: str | None = None
    ticket_type: str
    description: str
    priority: Literal["normal", "urgent"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TicketResponse(BaseModel):
    ticket_id: str
    status: str = "open"
    estimated_response: str


class UploadResponse(BaseModel):
    loaded_items: int
    categories: dict[str, int]


class EvalCase(BaseModel):
    id: str
    query: str
    expected_action: Action
    expected_category: str | None = None
    expected_intent: Intent = "unknown"
    expected_should_answer: bool | None = None
    difficulty: str | None = None
    note: str | None = None
    notes: str | None = None


class EvalResult(BaseModel):
    total: int
    action_accuracy: float
    category_hit_rate: float
    refusal_precision: float
    average_latency_ms: float
    cases: list[dict[str, Any]]


__all__ = [
    "Action",
    "Intent",
    "RoutingDecision",
    "RoutingSource",
    "KnowledgeItem",
    "SearchHit",
    "ChatRequest",
    "ChatResponse",
    "TicketRequest",
    "TicketResponse",
    "UploadResponse",
    "EvalCase",
    "EvalResult",
]
