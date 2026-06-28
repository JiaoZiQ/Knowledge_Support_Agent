import json
from collections import Counter
from pathlib import Path

from app.schemas import KnowledgeItem


class KnowledgeBase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: list[KnowledgeItem] = []

    def load(self) -> list[KnowledgeItem]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        raw_items = payload.get("knowledge", payload if isinstance(payload, list) else [])
        self.items = [KnowledgeItem.model_validate(item) for item in raw_items]
        return self.items

    def summary(self) -> dict[str, int]:
        return dict(Counter(item.category for item in self.items))
