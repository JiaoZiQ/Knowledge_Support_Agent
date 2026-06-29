from typing import Any

from app.policies.policy_config import load_policy_config
from app.schemas.routing import RoutingDecision


class GuardrailPolicy:
    def __init__(self, policy_config: dict[str, Any] | None = None) -> None:
        self.policy_config = policy_config or load_policy_config()
        self.rules = self.policy_config.get("guardrails", [])

    def evaluate(self, query: str) -> RoutingDecision | None:
        normalized = query.strip()
        lowered = normalized.lower()
        for rule in self.rules:
            exact_any = rule.get("exact_any", [])
            if exact_any and normalized not in exact_any:
                continue

            keywords_any = rule.get("keywords_any", [])
            if keywords_any and not any(keyword.lower() in lowered for keyword in keywords_any):
                continue

            keywords_all = rule.get("keywords_all", [])
            if keywords_all and not all(keyword.lower() in lowered for keyword in keywords_all):
                continue

            return RoutingDecision(
                action=rule["action"],
                intent=rule.get("intent", "unknown"),
                confidence=float(rule.get("confidence", 1.0)),
                needs_human=bool(rule.get("needs_human", False)),
                reason=rule.get("reason", f"guardrail: {rule.get('name', 'matched')}"),
                source="guardrail",
            )
        return None
