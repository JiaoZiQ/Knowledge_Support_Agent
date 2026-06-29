from typing import Literal

from pydantic import BaseModel, Field


Action = Literal[
    "answer",
    "answer_with_warning",
    "answer_with_disclaimer",
    "answer_then_escalate",
    "create_ticket",
    "decline",
    "clarify",
    "ask_clarifying_question",
    "escalate_if_unknown",
]

Intent = Literal[
    "feature_question",
    "usage_question",
    "billing_issue",
    "refund_request",
    "technical_issue",
    "privacy_request",
    "human_escalation",
    "capability_boundary",
    "unsafe_or_regulated_advice",
    "prompt_injection",
    "ticket_status",
    "unknown",
]

RoutingSource = Literal["guardrail", "llm", "fallback_rule", "confidence_policy"]


class RoutingDecision(BaseModel):
    action: Action
    intent: Intent = "unknown"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    needs_human: bool = False
    reason: str = ""
    source: RoutingSource = "fallback_rule"

    @classmethod
    def clarify(cls, reason: str, confidence: float = 1.0, source: RoutingSource = "confidence_policy") -> "RoutingDecision":
        return cls(
            action="ask_clarifying_question",
            intent="unknown",
            confidence=confidence,
            needs_human=False,
            reason=reason,
            source=source,
        )
