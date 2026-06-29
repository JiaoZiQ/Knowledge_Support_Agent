from openai import OpenAI

from app.schemas import SearchHit


class AnswerGenerator:
    def __init__(
        self,
        use_openai: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.use_openai = use_openai
        self.model = model
        self.client = None
        if use_openai and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def generate(self, query: str, hits: list[SearchHit], action: str, memory: str = "") -> str:
        if self.client:
            generated = self._generate_with_model(query, hits, action, memory)
            if generated:
                return generated
        return self._generate_template(query, hits, action, memory)

    def _generate_with_model(self, query: str, hits: list[SearchHit], action: str, memory: str = "") -> str | None:
        context = "\n\n".join(
            f"[{hit.item.id}] {hit.item.title}\n"
            f"category={hit.item.category}; risk={hit.item.risk_level}; action={hit.item.recommended_action}\n"
            f"{hit.item.content}"
            for hit in hits[:3]
        )
        if not context:
            context = "没有可靠知识库命中。"

        system_prompt = (
            "你是 AI 简历优化平台的客服 Agent。必须基于知识库回答，不要编造。"
            "如果 action 是 create_ticket，说明需要人工核实；如果 action 是 ask_clarifying_question，先追问必要信息；如果 action 是 decline，礼貌拒绝；"
            "如果 action 是 answer_with_disclaimer 或 answer_with_warning，要清楚给出免责或风险提醒。"
            "回答要简洁、中文、适合客服场景，最后保留来源 ID。"
        )
        user_prompt = (
            f"用户问题：{query}\n"
            f"当前动作：{action}\n"
            f"会话记忆：{memory or '无'}\n"
            f"知识库上下文：\n{context}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            content = response.choices[0].message.content or ""
            return content.strip() or None
        except Exception:
            return None

    def _generate_template(self, query: str, hits: list[SearchHit], action: str, memory: str = "") -> str:
        if not hits:
            return "我暂时没有在知识库中找到可靠依据。请补充更多背景信息，或选择转人工客服。"

        primary = hits[0].item
        prefix = {
            "answer_with_warning": "需要提醒你：",
            "answer_with_disclaimer": "免责声明：",
            "answer_then_escalate": "我先根据知识库说明处理，同时建议继续转人工确认：",
            "create_ticket": "这个问题需要人工客服进一步核实，我会为你创建工单。根据知识库说明：",
            "decline": "抱歉，这类问题不在平台可提供支持的范围内。",
            "clarify": "为了更准确地处理，请你再补充一点信息。",
            "ask_clarifying_question": "为了更准确地处理，请你再补充一点信息。",
            "escalate_if_unknown": "这个问题当前知识库无法确认，建议转人工进一步核实。",
        }.get(action, "")

        if action == "decline":
            return f"{prefix}{primary.content}"
        if action in {"clarify", "ask_clarifying_question"}:
            return f"{prefix}请提供错误提示、订单号、文件格式、工单编号或你正在操作的页面。"

        answer = f"{prefix}{primary.content}"
        if memory:
            answer += f"\n\n我也会参考当前会话记录：{memory[:120]}"
        answer += f"\n\n来源：{primary.title}（{primary.id}）"
        return answer
