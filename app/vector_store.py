from typing import Protocol

import chromadb

from app.embeddings import HashEmbeddingModel, tokenize
from app.schemas import KnowledgeItem, SearchHit


CATEGORY_TERMS: dict[str, list[str]] = {
    "product": ["平台", "产品", "适合", "用户", "求职", "跳槽", "职场"],
    "account": ["账号", "账户", "登录", "注册", "密码", "邮箱", "手机", "锁"],
    "resume": ["简历", "上传", "解析", "pdf", "docx", "txt", "扫描", "文件", "15mb", "大小"],
    "ai_optimization": ["优化", "jd", "岗位", "项目经历", "实习经历", "技能", "中英文"],
    "interview": ["面试", "面试题", "追问", "岗位题"],
    "billing": ["免费", "付费", "套餐", "价格", "退款", "扣费", "支付", "续费", "发票", "订单", "开通", "会员"],
    "privacy": ["隐私", "数据", "删除", "训练", "导出", "保存", "个人信息"],
    "error": ["失败", "报错", "错误", "转圈", "卡住", "验证码", "邮件", "收不到", "没反应"],
    "ticket": ["人工", "工单", "客服", "催办", "申诉", "投诉"],
    "boundary": ["法律", "医疗", "投资", "理财", "offer", "保证", "成功率", "录用", "腾讯", "阿里", "找到工作", "朋友"],
}


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
        metadatas = raw.get("metadatas", [[]])[0]

        hits: list[SearchHit] = []
        for item_id, distance, metadata in zip(ids, distances, metadatas):
            item = self.items_by_id.get(item_id)
            if not item:
                continue
            base_score = max(0.0, 1.0 - float(distance))
            score = base_score + self._category_boost(query, item.category)
            intent_category = self._intent_category(query)
            if intent_category == item.category:
                score += 0.55
            matched_keywords = [
                keyword
                for keyword in item.keywords
                if keyword.lower() in query.lower() or query.lower() in keyword.lower()
            ]
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
        return " ".join([item.category, item.title, item.content, *item.keywords])

    @staticmethod
    def _category_boost(query: str, category: str) -> float:
        query_lower = query.lower()
        matches = [term for term in CATEGORY_TERMS.get(category, []) if term in query_lower]
        return min(0.3, 0.075 * len(matches))

    @staticmethod
    def _title_boost(query: str, title: str) -> float:
        query_tokens = set(tokenize(query))
        title_lower = title.lower()
        return 0.08 if any(token in title_lower for token in query_tokens if len(token) > 1) else 0.0

    @staticmethod
    def _intent_category(query: str) -> str | None:
        query_lower = query.lower()
        if any(term in query for term in ["技术架构", "系统架构", "内部实现", "GPT", "自研", "法律", "医疗", "投资", "理财"]):
            return "boundary"
        if any(term in query for term in ["公司HR", "候选人", "1000份", "批量上传"]):
            return "boundary"
        if any(term in query for term in ["HR", "筛选系统", "全新的简历", "代写", "50份", "批量"]):
            return "product"
        if any(term in query for term in ["投了", "没有面试", "没面试"]):
            return "boundary"
        if any(term in query for term in ["付款成功", "支付成功"]) and any(term in query for term in ["没给我开通", "没有开通", "未开通"]):
            return "error"
        if any(term in query for term in ["退款", "扣费", "付款", "支付", "续费", "发票", "套餐", "会员", "补偿"]):
            return "billing"
        if any(term in query for term in ["上传失败", "一直失败", "没反应", "转圈", "卡住", "验证码", "收不到"]):
            return "error"
        if any(term in query for term in ["身份证", "隐私", "个人信息", "数据删除", "用于训练"]):
            return "privacy"
        if any(term in query for term in ["注销", "账号", "登录", "密码", "邮箱", "手机号", "冷静期"]):
            return "account"
        if "识别" in query and any(term in query for term in ["项目经历", "实习经历", "教育经历"]):
            return "resume"
        if any(term in query for term in ["扫描", "pdf", "docx", "文件大小", "15mb", "解析失败"]):
            return "resume"
        if any(term in query_lower for term in ["offer", "jd"]) or any(term in query for term in ["AI会帮我修改", "修改简历", "优化", "项目经历", "技能关键词"]):
            return "ai_optimization"
        if "面试" in query:
            return "interview"
        return None


LocalVectorStore = ChromaVectorStore
