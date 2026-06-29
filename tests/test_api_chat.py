def test_health_returns_ok(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["orchestrator"] == "langgraph"
    assert payload["vector_store"] == "chroma"


def test_chat_schema_and_no_key_required(api_client):
    response = api_client.post("/chat", json={"query": "免费版和专业版有什么区别？", "user_id": "api_user"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"]
    assert payload["trace_id"]
    assert payload["action"] in {
        "answer",
        "answer_with_warning",
        "answer_with_disclaimer",
        "answer_then_escalate",
        "create_ticket",
        "decline",
        "clarify",
        "ask_clarifying_question",
        "escalate_if_unknown",
    }
    assert "intent" in payload
    assert "routing_source" in payload


def test_human_escalation_creates_ticket(api_client):
    response = api_client.post("/chat", json={"query": "我要找人工客服投诉", "user_id": "api_user"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["action"] == "create_ticket"
    assert payload["ticket_id"]
