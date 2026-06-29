from app.policies.intent_router import IntentRouter


def test_router_fallback_works_without_api_key(test_settings):
    settings = test_settings.model_copy(update={"router_mode": "offline", "openai_api_key": None})
    decision = IntentRouter(settings).route("免费版和专业版有什么区别？")
    assert decision.source == "fallback_rule"
    assert decision.action == "answer"
    assert decision.confidence > 0


def test_router_fallback_clarifies_vague_query(test_settings):
    decision = IntentRouter(test_settings).route("我的简历传不上去怎么办？")
    assert decision.action == "ask_clarifying_question"
