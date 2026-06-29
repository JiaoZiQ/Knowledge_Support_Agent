from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.main import app, create_services


@pytest.fixture()
def test_settings(tmp_path):
    return Settings(
        use_openai_llm=False,
        router_mode="offline",
        embedding_provider="hash",
        database_path=tmp_path / "test.db",
        chroma_path=tmp_path / "chroma",
        knowledge_base_path=ROOT / "data" / "knowledge_base" / "raw" / "knowledge_base.json",
        eval_dataset_path=ROOT / "data" / "eval" / "eval_dataset.json",
        retrieval_min_score=0.35,
        routing_min_confidence=0.60,
    )


@pytest.fixture()
def services(test_settings):
    return create_services(test_settings)


@pytest.fixture()
def harness(services):
    return services["harness"]


@pytest.fixture()
def api_client(services, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "services", services)
    return TestClient(app)
