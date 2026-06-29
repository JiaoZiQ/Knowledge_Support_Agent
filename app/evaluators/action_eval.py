import json
import time
from pathlib import Path
from typing import Any

from app.harness import AgentHarness
from app.schemas import ChatRequest


NON_ANSWER_ACTIONS = {"create_ticket", "decline", "clarify", "ask_clarifying_question", "escalate_if_unknown"}
ESCALATION_ACTIONS = {"create_ticket", "answer_then_escalate"}
CLARIFY_ACTIONS = {"clarify", "ask_clarifying_question"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ActionEvaluator:
    def __init__(self, harness: AgentHarness) -> None:
        self.harness = harness

    def evaluate_dataset(self, name: str, path: Path) -> dict[str, Any]:
        cases = load_jsonl(path)
        rows: list[dict[str, Any]] = []
        action_matches = 0
        intent_matches = 0
        refusal_expected = 0
        refusal_correct = 0
        escalation_expected = 0
        escalation_correct = 0
        clarify_expected = 0
        clarify_correct = 0
        latencies: list[float] = []

        for case in cases:
            started = time.perf_counter()
            response = self.harness.chat(
                ChatRequest(query=case["query"], user_id="eval_user", session_id=f"eval_{name}_{case['id']}")
            )
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)

            expected_action = case["expected_action"]
            expected_intent = case.get("expected_intent", "unknown")
            action_match = response.action == expected_action
            intent_match = response.intent == expected_intent
            action_matches += int(action_match)
            intent_matches += int(intent_match)

            if expected_action in NON_ANSWER_ACTIONS:
                refusal_expected += 1
                refusal_correct += int(response.action in NON_ANSWER_ACTIONS)
            if expected_action in ESCALATION_ACTIONS:
                escalation_expected += 1
                escalation_correct += int(response.action in ESCALATION_ACTIONS)
            if expected_action in CLARIFY_ACTIONS:
                clarify_expected += 1
                clarify_correct += int(response.action in CLARIFY_ACTIONS)

            rows.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected_action": expected_action,
                    "predicted_action": response.action,
                    "expected_intent": expected_intent,
                    "predicted_intent": response.intent,
                    "routing_source": response.routing_source,
                    "confidence": response.confidence,
                    "action_match": action_match,
                    "intent_match": intent_match,
                    "latency_ms": round(latency_ms, 2),
                    "notes": case.get("notes", ""),
                }
            )

        total = max(len(cases), 1)
        errors = [row for row in rows if not row["action_match"] or not row["intent_match"]]
        return {
            "dataset": name,
            "total": len(cases),
            "action_accuracy": round(action_matches / total, 4),
            "intent_accuracy": round(intent_matches / total, 4),
            "refusal_precision": round(refusal_correct / max(refusal_expected, 1), 4),
            "escalation_recall": round(escalation_correct / max(escalation_expected, 1), 4),
            "clarify_accuracy": round(clarify_correct / max(clarify_expected, 1), 4),
            "average_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
            "errors": errors,
            "cases": rows,
        }

    def evaluate_many(self, datasets: dict[str, Path]) -> dict[str, Any]:
        dataset_results = {name: self.evaluate_dataset(name, path) for name, path in datasets.items()}
        all_cases = [case for result in dataset_results.values() for case in result["cases"]]
        total = max(len(all_cases), 1)
        action_matches = sum(1 for case in all_cases if case["action_match"])
        intent_matches = sum(1 for case in all_cases if case["intent_match"])
        errors = [case for case in all_cases if not case["action_match"] or not case["intent_match"]]
        return {
            "overall": {
                "total": len(all_cases),
                "action_accuracy": round(action_matches / total, 4),
                "intent_accuracy": round(intent_matches / total, 4),
                "errors": errors,
            },
            "datasets": dataset_results,
        }


def write_action_report(result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Action Evaluation Report",
        "",
        "## Overall",
        "",
        f"- Total cases: {result['overall']['total']}",
        f"- Action accuracy: {result['overall']['action_accuracy']:.2%}",
        f"- Intent accuracy: {result['overall']['intent_accuracy']:.2%}",
        f"- Error samples: {len(result['overall']['errors'])}",
        "",
        "## By Dataset",
        "",
    ]
    for name, dataset in result["datasets"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Total: {dataset['total']}",
                f"- Action accuracy: {dataset['action_accuracy']:.2%}",
                f"- Intent accuracy: {dataset['intent_accuracy']:.2%}",
                f"- Refusal precision: {dataset['refusal_precision']:.2%}",
                f"- Escalation recall: {dataset['escalation_recall']:.2%}",
                f"- Clarify accuracy: {dataset['clarify_accuracy']:.2%}",
                f"- Avg latency: {dataset['average_latency_ms']} ms",
                f"- Errors: {len(dataset['errors'])}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
