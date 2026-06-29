import json
import time
from pathlib import Path
from typing import Any

from app.harness import AgentHarness
from app.schemas import ChatRequest, EvalCase, EvalResult


NON_ANSWER_ACTIONS = {"create_ticket", "decline", "clarify", "ask_clarifying_question", "escalate_if_unknown"}


class EvalRunner:
    def __init__(self, harness: AgentHarness, dataset_path: Path) -> None:
        self.harness = harness
        self.dataset_path = dataset_path

    def load_cases(self) -> list[EvalCase]:
        payload = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        raw_cases = payload.get("eval_cases", payload if isinstance(payload, list) else [])
        return [EvalCase.model_validate(item) for item in raw_cases]

    def run(self, limit: int | None = None) -> EvalResult:
        cases = self.load_cases()
        if limit:
            cases = cases[:limit]

        rows: list[dict[str, Any]] = []
        action_matches = 0
        category_hits = 0
        refusal_correct = 0
        refusal_total = 0
        latencies: list[float] = []

        for case in cases:
            started = time.perf_counter()
            response = self.harness.chat(
                ChatRequest(query=case.query, user_id="eval_user", session_id=f"eval_{case.id}")
            )
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)

            predicted_category = response.citations[0]["category"] if response.citations else "none"
            action_match = response.action == case.expected_action
            category_match = predicted_category == case.expected_category
            should_not_answer = not case.expected_should_answer
            if should_not_answer:
                refusal_total += 1
                if response.action in NON_ANSWER_ACTIONS:
                    refusal_correct += 1

            action_matches += int(action_match)
            category_hits += int(category_match)
            rows.append(
                {
                    "id": case.id,
                    "query": case.query,
                    "expected_action": case.expected_action,
                    "predicted_action": response.action,
                    "expected_category": case.expected_category,
                    "predicted_category": predicted_category,
                    "difficulty": case.difficulty,
                    "action_match": action_match,
                    "category_match": category_match,
                    "confidence": response.confidence,
                    "latency_ms": round(latency_ms, 2),
                }
            )

        total = max(len(cases), 1)
        return EvalResult(
            total=len(cases),
            action_accuracy=round(action_matches / total, 4),
            category_hit_rate=round(category_hits / total, 4),
            refusal_precision=round(refusal_correct / max(refusal_total, 1), 4),
            average_latency_ms=round(sum(latencies) / max(len(latencies), 1), 2),
            cases=rows,
        )
