import json
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from app.config import Settings
from app.policies.policy_config import load_policy_config
from app.schemas.routing import Intent, RoutingDecision


ACTION_BY_INTENT: dict[str, str] = {
    "feature_question": "answer",
    "usage_question": "answer",
    "billing_issue": "answer",
    "refund_request": "create_ticket",
    "technical_issue": "answer",
    "privacy_request": "answer_with_warning",
    "human_escalation": "create_ticket",
    "capability_boundary": "answer_with_warning",
    "unsafe_or_regulated_advice": "decline",
    "prompt_injection": "decline",
    "ticket_status": "ask_clarifying_question",
    "unknown": "ask_clarifying_question",
}


class IntentRouter:
    def __init__(self, settings: Settings, policy_config: dict[str, Any] | None = None) -> None:
        self.settings = settings
        self.policy_config = policy_config or load_policy_config()
        self.last_error: str | None = None
        self.client = None
        if settings.router_mode == "openai" and settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None)

    def route(self, query: str) -> RoutingDecision:
        self.last_error = None
        if self.client:
            decision = self._route_with_llm(query)
            if decision:
                return decision
        return self._route_with_fallback(query)

    def _route_with_llm(self, query: str) -> RoutingDecision | None:
        system_prompt = (
            "Classify a customer-support query for an AI resume platform. "
            "Return JSON only with keys: intent, action, confidence, needs_human, reason. "
            "Allowed actions: answer, clarify, decline, create_ticket. "
            "Allowed intents: feature_question, usage_question, billing_issue, refund_request, "
            "technical_issue, privacy_request, human_escalation, capability_boundary, "
            "unsafe_or_regulated_advice, unknown."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=250,
                timeout=15,
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            action = parsed.get("action", "answer")
            if action == "clarify":
                action = "ask_clarifying_question"
            return RoutingDecision(
                action=action,
                intent=parsed.get("intent", "unknown"),
                confidence=float(parsed.get("confidence", 0.0)),
                needs_human=bool(parsed.get("needs_human", False)),
                reason=parsed.get("reason", "llm structured route"),
                source="llm",
            )
        except Exception as exc:  # OpenAI failures, timeouts, invalid JSON, or schema errors all fall back offline.
            if isinstance(exc, (json.JSONDecodeError, ValidationError)):
                self.last_error = f"structured_router_parse_error: {type(exc).__name__}"
            else:
                self.last_error = f"structured_router_error: {type(exc).__name__}"
            return None

    def _route_with_fallback(self, query: str) -> RoutingDecision:
        lowered = query.lower()
        for pattern in self.policy_config.get("clarify_patterns", []):
            if pattern.lower() in lowered:
                return RoutingDecision(
                    action="ask_clarifying_question",
                    intent="unknown",
                    confidence=0.92,
                    needs_human=False,
                    reason=f"fallback_rule: vague pattern '{pattern}'",
                    source="fallback_rule",
                )

        technical_phrases = ["上传失败", "卡着", "太大", "收不到", "没反应", "转圈", "打不开", "报错", "版式乱"]
        if any(phrase.lower() in lowered for phrase in technical_phrases):
            return RoutingDecision(
                action="answer",
                intent="technical_issue",
                confidence=0.82,
                needs_human=False,
                reason="fallback_rule: technical issue phrase",
                source="fallback_rule",
            )

        intent_keywords = self.policy_config.get("intent_keywords", {})
        best_intent: Intent = "unknown"
        best_count = 0
        for intent, keywords in intent_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in lowered)
            if count > best_count:
                best_intent = intent
                best_count = count

        confidence = min(0.95, 0.45 + best_count * 0.18) if best_count else 0.35
        action = ACTION_BY_INTENT.get(best_intent, "ask_clarifying_question")
        return RoutingDecision(
            action=action,
            intent=best_intent,
            confidence=confidence,
            needs_human=action == "create_ticket",
            reason=f"fallback_rule: matched {best_count} keyword(s)",
            source="fallback_rule",
        )
