from typing import Any

from app.schemas import SearchHit
from app.schemas.routing import RoutingDecision
from app.tools import serialize_hits


SENSITIVE_MARKERS = ["身份证", "手机号", "电话", "邮箱", "银行卡", "订单号"]


def redact_text(text: str, max_length: int = 180) -> str:
    redacted = text
    for marker in SENSITIVE_MARKERS:
        if marker in redacted:
            redacted = redacted.replace(marker, f"{marker}[REDACTED]")
    return redacted[:max_length]


def redacted_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": profile.get("plan"),
        "preferred_language": profile.get("preferred_language"),
        "open_ticket_count": profile.get("open_ticket_count"),
    }


def build_trace_payload(
    *,
    profile: dict[str, Any],
    hits: list[SearchHit],
    decision: RoutingDecision,
    ticket_id: str | None,
    graph_steps: list[str],
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "profile": redacted_profile(profile),
        "retrieval": serialize_hits(hits),
        "ticket_id": ticket_id,
        "routing": decision.model_dump(),
        "routing_source": decision.source,
        "guardrail": decision.reason,
        "graph_steps": graph_steps,
        "orchestrator": "langgraph",
        "errors": errors or [],
    }
