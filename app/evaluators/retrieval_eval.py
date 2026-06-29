import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.vector_store import ChromaVectorStore


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class RetrievalEvaluator:
    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self.vector_store = vector_store

    def evaluate(self, path: Path, top_k: int = 3) -> dict[str, Any]:
        cases = load_jsonl(path)
        rows: list[dict[str, Any]] = []
        recall_1 = 0
        recall_3 = 0
        reciprocal_ranks: list[float] = []
        no_hit = 0
        by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "recall_1": 0, "recall_3": 0})

        for case in cases:
            expected_ids = set(case["expected_ids"])
            hits = self.vector_store.search(case["query"], top_k=top_k)
            hit_ids = [hit.item.id for hit in hits]
            rank = next((index + 1 for index, item_id in enumerate(hit_ids) if item_id in expected_ids), None)
            hit_at_1 = rank == 1
            hit_at_3 = rank is not None and rank <= 3
            recall_1 += int(hit_at_1)
            recall_3 += int(hit_at_3)
            reciprocal_ranks.append(1 / rank if rank else 0)
            no_hit += int(rank is None)

            category = case.get("expected_category", "unknown")
            by_category[category]["total"] += 1
            by_category[category]["recall_1"] += int(hit_at_1)
            by_category[category]["recall_3"] += int(hit_at_3)
            rows.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected_ids": list(expected_ids),
                    "hit_ids": hit_ids,
                    "rank": rank,
                    "recall_1": hit_at_1,
                    "recall_3": hit_at_3,
                    "expected_category": category,
                }
            )

        total = max(len(cases), 1)
        grouped = {
            category: {
                "total": values["total"],
                "recall_1": round(values["recall_1"] / max(values["total"], 1), 4),
                "recall_3": round(values["recall_3"] / max(values["total"], 1), 4),
            }
            for category, values in sorted(by_category.items())
        }
        return {
            "total": len(cases),
            "recall_at_1": round(recall_1 / total, 4),
            "recall_at_3": round(recall_3 / total, 4),
            "mrr": round(sum(reciprocal_ranks) / total, 4),
            "no_hit_rate": round(no_hit / total, 4),
            "by_category": grouped,
            "cases": rows,
        }


def write_retrieval_report(result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"- Total cases: {result['total']}",
        f"- Recall@1: {result['recall_at_1']:.2%}",
        f"- Recall@3: {result['recall_at_3']:.2%}",
        f"- MRR: {result['mrr']:.4f}",
        f"- No-hit rate: {result['no_hit_rate']:.2%}",
        "",
        "## By Category",
        "",
    ]
    for category, values in result["by_category"].items():
        lines.append(
            f"- {category}: total={values['total']}, recall@1={values['recall_1']:.2%}, recall@3={values['recall_3']:.2%}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
