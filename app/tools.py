from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.schemas import SearchHit
from app.storage import AppStorage
from app.vector_store import LocalVectorStore


@dataclass
class ToolResult:
    name: str
    output: Any


class ToolRegistry:
    def __init__(self, storage: AppStorage, vector_store: LocalVectorStore) -> None:
        self.storage = storage
        self.vector_store = vector_store
        self.tools: dict[str, Callable[..., ToolResult]] = {
            "search_knowledge_base": self.search_knowledge_base,
            "create_ticket": self.create_ticket,
            "get_user_profile": self.get_user_profile,
            "summarize_session": self.summarize_session,
        }

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        return self.tools[name](**kwargs)

    def search_knowledge_base(self, query: str, top_k: int = 4) -> ToolResult:
        hits = self.vector_store.search(query, top_k=top_k)
        return ToolResult(name="search_knowledge_base", output=hits)

    def create_ticket(
        self,
        user_id: str,
        session_id: str,
        ticket_type: str,
        description: str,
        priority: str = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        ticket_id = self.storage.create_ticket(
            user_id=user_id,
            session_id=session_id,
            ticket_type=ticket_type,
            description=description,
            priority=priority,
            metadata=metadata or {},
        )
        return ToolResult(name="create_ticket", output={"ticket_id": ticket_id})

    def get_user_profile(self, user_id: str) -> ToolResult:
        profile = {
            "user_id": user_id,
            "plan": "free",
            "preferred_language": "zh-CN",
            "open_ticket_count": 0,
        }
        return ToolResult(name="get_user_profile", output=profile)

    def summarize_session(self, session_id: str) -> ToolResult:
        return ToolResult(name="summarize_session", output=self.storage.get_memory(session_id))


def serialize_hits(hits: list[SearchHit]) -> list[dict[str, Any]]:
    return [
        {
            "id": hit.item.id,
            "category": hit.item.category,
            "title": hit.item.title,
            "score": hit.score,
            "risk_level": hit.item.risk_level,
            "recommended_action": hit.item.recommended_action,
            "source": hit.item.source,
            "requires_human_review": hit.item.requires_human_review,
            "matched_keywords": hit.matched_keywords,
        }
        for hit in hits
    ]
