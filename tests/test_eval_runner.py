from pathlib import Path

from app.config import Settings
from app.main import create_services


ROOT = Path(__file__).resolve().parents[1]


def test_eval_runner_returns_metrics(tmp_path):
    settings = Settings(
        database_path=tmp_path / "test.db",
        knowledge_base_path=ROOT / "data" / "knowledge_base" / "raw" / "knowledge_base.json",
        eval_dataset_path=ROOT / "data" / "eval" / "eval_dataset.json",
        use_openai_llm=False,
    )
    runner = create_services(settings)["eval_runner"]
    result = runner.run(limit=5)
    assert result.total == 5
    assert 0 <= result.action_accuracy <= 1
    assert 0 <= result.category_hit_rate <= 1
    assert result.cases
