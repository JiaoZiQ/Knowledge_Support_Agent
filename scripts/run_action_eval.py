import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.evaluators.action_eval import ActionEvaluator, write_action_report
from app.main import create_services


EVAL_DIR = ROOT / "data" / "eval"
OUTPUT_DIR = ROOT / "artifacts" / "eval"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime_dir = ROOT / "artifacts" / "runtime" / f"action_eval_{uuid.uuid4().hex[:8]}"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        use_openai_llm=False,
        router_mode="offline",
        embedding_provider="hash",
        database_path=runtime_dir / "action_eval.db",
        chroma_path=runtime_dir / "chroma",
        knowledge_base_path=ROOT / "data" / "knowledge_base" / "raw" / "knowledge_base.json",
        eval_dataset_path=ROOT / "data" / "eval" / "eval_dataset.json",
    )
    services = create_services(settings)
    evaluator = ActionEvaluator(services["harness"])
    result = evaluator.evaluate_many(
        {
            "standard": EVAL_DIR / "standard_cases.jsonl",
            "paraphrase": EVAL_DIR / "paraphrase_cases.jsonl",
            "adversarial": EVAL_DIR / "adversarial_cases.jsonl",
        }
        )

    (OUTPUT_DIR / "action_eval_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_action_report(result, OUTPUT_DIR / "action_eval_report.md")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
