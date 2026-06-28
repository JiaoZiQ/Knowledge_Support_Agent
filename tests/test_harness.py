from pathlib import Path

from app.config import Settings
from app.main import create_services
from app.schemas import ChatRequest


ROOT = Path(__file__).resolve().parents[1]


def make_harness(tmp_path):
    settings = Settings(
        database_path=tmp_path / "test.db",
        chroma_path=tmp_path / "chroma",
        knowledge_base_path=ROOT / "data" / "knowledge_base" / "raw" / "knowledge_base.json",
        eval_dataset_path=ROOT / "data" / "eval" / "eval_dataset.json",
        use_openai_llm=False,
    )
    return create_services(settings)["harness"]


def test_answer_with_citation(tmp_path):
    harness = make_harness(tmp_path)
    response = harness.chat(ChatRequest(query="免费版和专业版有什么区别？", user_id="u1"))
    assert response.action == "answer"
    assert response.citations
    assert response.citations[0]["category"] == "billing"
    assert "来源" in response.answer


def test_refund_creates_ticket(tmp_path):
    harness = make_harness(tmp_path)
    response = harness.chat(ChatRequest(query="我被重复扣费了，可以直接退款吗？", user_id="u1"))
    assert response.action == "create_ticket"
    assert response.ticket_id
    assert "工单已创建" in response.answer


def test_boundary_decline(tmp_path):
    harness = make_harness(tmp_path)
    response = harness.chat(ChatRequest(query="我想问一下劳动合同纠纷的法律问题", user_id="u1"))
    assert response.action == "decline"
    assert not response.ticket_id
