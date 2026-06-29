from app.config import Settings
from app.schemas import SearchHit
from app.schemas.routing import RoutingDecision


class ConfidencePolicy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def apply(self, decision: RoutingDecision, hits: list[SearchHit]) -> RoutingDecision:
        if decision.source == "guardrail":
            return decision

        if not hits:
            return RoutingDecision.clarify("confidence_policy: no retrieval results")

        top_score = hits[0].score
        if top_score < self.settings.retrieval_min_score:
            return RoutingDecision.clarify(
                f"confidence_policy: top retrieval score {top_score:.3f} below {self.settings.retrieval_min_score:.3f}"
            )

        if decision.confidence < self.settings.routing_min_confidence:
            return RoutingDecision.clarify(
                f"confidence_policy: routing confidence {decision.confidence:.3f} below {self.settings.routing_min_confidence:.3f}",
                confidence=decision.confidence,
            )

        return decision
