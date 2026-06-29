from typing import Protocol

import chromadb

from app.embeddings import HashEmbeddingModel, tokenize
from app.policies.policy_config import get_category_terms
from app.schemas import KnowledgeItem, SearchHit


class EmbeddingModel(Protocol):
    def embed(self, text: str): ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class ChromaVectorStore:
    def __init__(
        self,
        persist_path: str,
        collection_name: str,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.items_by_id: dict[str, KnowledgeItem] = {}
        self.category_terms = get_category_terms()

    def build(self, items: list[KnowledgeItem]) -> None:
        self.items_by_id = {item.id: item for item in items}
        existing = self.collection.get(include=[])
        existing_ids = set(existing.get("ids", []))
        if existing_ids:
            self.collection.delete(ids=list(existing_ids))

        documents = [self._searchable_text(item) for item in items]
        embeddings = self.embedding_model.embed_batch(documents)
        self.collection.add(
            ids=[item.id for item in items],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "category": item.category,
                    "title": item.title,
                    "risk_level": item.risk_level,
                    "recommended_action": item.recommended_action,
                    "keywords": ",".join(item.keywords),
                    "source": item.source or "",
                    "updated_at": item.updated_at or "",
                    "tags": ",".join(item.tags),
                    "requires_human_review": item.requires_human_review,
                }
                for item in items
            ],
        )

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        query_embedding = self.embedding_model.embed(query).astype(float).tolist()
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(top_k * 4, top_k), max(len(self.items_by_id), 1)),
            include=["distances", "metadatas"],
        )
        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        intent_category = self._intent_category(query)

        hits: list[SearchHit] = []
        for item_id, distance in zip(ids, distances):
            item = self.items_by_id.get(item_id)
            if not item:
                continue
            base_score = max(0.0, 1.0 - float(distance))
            matched_keywords = [
                keyword
                for keyword in item.keywords
                if keyword.lower() in query.lower() or query.lower() in keyword.lower()
            ]
            score = base_score
            score += self._category_boost(query, item.category)
            score += 0.55 if intent_category == item.category else 0.0
            score += min(0.18, 0.045 * len(matched_keywords))
            score += self._title_boost(query, item.title)
            hits.append(
                SearchHit(
                    item=item,
                    score=round(score, 4),
                    matched_keywords=matched_keywords,
                )
            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    @staticmethod
    def _searchable_text(item: KnowledgeItem) -> str:
        return " ".join([item.category, item.title, item.content, *item.keywords, *item.tags])

    def _category_boost(self, query: str, category: str) -> float:
        query_lower = query.lower()
        matches = [term for term in self.category_terms.get(category, []) if term.lower() in query_lower]
        return min(0.3, 0.075 * len(matches))

    @staticmethod
    def _title_boost(query: str, title: str) -> float:
        query_tokens = set(tokenize(query))
        title_lower = title.lower()
        return 0.08 if any(token in title_lower for token in query_tokens if len(token) > 1) else 0.0

    def _intent_category(self, query: str) -> str | None:
        query_lower = query.lower()
        best_category = None
        best_count = 0
        for category, terms in self.category_terms.items():
            count = sum(1 for term in terms if term.lower() in query_lower)
            if count > best_count:
                best_category = category
                best_count = count
        return best_category


LocalVectorStore = ChromaVectorStore
