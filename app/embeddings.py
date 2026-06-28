import hashlib
import math
import re
from collections import Counter

import numpy as np
from openai import OpenAI


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    compact = "".join(t for t in tokens if len(t) == 1 and "\u4e00" <= t <= "\u9fff")
    grams = [compact[i : i + 2] for i in range(max(0, len(compact) - 1))]
    return tokens + grams


class HashEmbeddingModel:
    """Small deterministic embedding model for local demos and tests."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1 if digest[4] % 2 == 0 else -1
            vector[index] += sign * (1.0 + math.log(count))
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text).astype(float).tolist() for text in texts]


class OpenAIEmbeddingModel:
    def __init__(
        self,
        api_key: str,
        base_url: str | None,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        return np.array(self.embed_batch([text])[0], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)
