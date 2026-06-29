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
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url or None) if use_openai and api_key else None

    def generate(self, query: str, hits: list[SearchHit], action: str, memory: str = "") -> str:
        if self.client:
            generated = self._generate_with_model(query, hits, action, memory)
            if generated:
                return generated
        return self._generate_template(hits, action, memory)

    def _generate_with_model(self, query: str, hits: list[SearchHit], action: str, memory: str = "") -> str | None:
        context = "\n\n".join(
            f"[{hit.item.id}] {hit.item.title}\n"
            f"category={hit.item.category}; risk={hit.item.risk_level}; action={hit.item.recommended_action}\n"
            f"{hit.item.content}"
            for hit in hits[:3]
        ) or "No reliable knowledge-base hit."
        system_prompt = (
            "You are a customer-support agent for an AI resume platform. "
            "Answer in Chinese, use only the supplied knowledge-base context, do not invent facts, "
            "and keep the tone concise and service-oriented. If the action is create_ticket, explain "
            "that human review is required. If the action is ask_clarifying_question, ask for the "
            "minimum missing information. If the action is decline, politely refuse. Always include source IDs."
        )
        user_prompt = (
            f"User query: {query}\n"
            f"Action: {action}\n"
            f"Session memory summary: {memory or 'none'}\n"
            f"Knowledge context:\n{context}"
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

    def _generate_template(self, hits: list[SearchHit], action: str, memory: str = "") -> str:
        if action in {"clarify", "ask_clarifying_question"}:
            return "为了更准确地处理，请补充关键信息，例如错误提示、订单号、文件格式、工单编号或你正在操作的页面。"
        if not hits:
            return "我暂时没有在知识库中找到可靠依据。请补充更多背景信息，或选择转人工客服。"

        primary = hits[0].item
        prefix = {
            "answer_with_warning": "需要提醒你：",
            "answer_with_disclaimer": "免责声明：",
            "answer_then_escalate": "我先根据知识库说明处理，同时建议继续转人工确认：",
            "create_ticket": "这个问题需要人工客服进一步核实，我会为你创建工单。根据知识库说明：",
            "decline": "抱歉，这类问题不在平台可提供支持的范围内。",
            "escalate_if_unknown": "这个问题当前知识库无法确认，建议转人工进一步核实。",
        }.get(action, "")
        answer = f"{prefix}{primary.content}"
        if memory:
            answer += f"\n\n我会参考当前会话摘要：{memory[:120]}"
        answer += f"\n\n来源：{primary.title}（{primary.id}）"
        return answer
