from app.schemas import ChatRequest


def test_trace_written_with_routing_data(harness, services):
    response = harness.chat(ChatRequest(query="我被重复扣费了", user_id="trace_user"))
    assert response.ticket_id
    trace = services["storage"].get_session_trace(response.session_id)
    assert trace["traces"]
    latest = trace["traces"][0]
    data = latest["data"]
    assert data["routing_source"] == "guardrail"
    assert data["routing"]["intent"] == "billing_issue"
    assert data["ticket_id"] == response.ticket_id
    assert "graph_steps" in data
    assert "openai_api_key" not in str(data).lower()
