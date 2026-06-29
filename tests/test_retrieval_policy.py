from app.config import Settings
from app.main import create_services
from app.schemas import ChatRequest


def test_low_retrieval_score_returns_clarify(tmp_path, test_settings):
    strict_settings = test_settings.model_copy(
        update={
            "database_path": tmp_path / "strict.db",
            "chroma_path": tmp_path / "strict_chroma",
            "retrieval_min_score": 999.0,
        }
    )
    harness = create_services(strict_settings)["harness"]
    response = harness.chat(ChatRequest(query="普通功能怎么使用？", user_id="u1"))
    assert response.action == "ask_clarifying_question"
    assert response.routing_source == "confidence_policy"
