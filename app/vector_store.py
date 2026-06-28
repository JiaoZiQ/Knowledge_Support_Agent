from dataclasses import dataclass

from app.embeddings import HashEmbeddingModel, cosine_similarity, tokenize
from app.schemas import KnowledgeItem, SearchHit


CATEGORY_TERMS: dict[str, list[str]] = {
    "product": ["平台", "产品", "适合", "用户", "求职", "跳槽", "职场"],
    "account": ["账号", "账户", "登录", "注册", "密码", "邮箱", "手机", "锁"],
    "resume": ["简历", "上传", "解析", "pdf", "docx", "txt", "扫描", "文件", "15mb", "大小"],
    "ai_optimization": ["优化", "jd", "岗位", "项目经历", "实习经历", "技能", "中英文"],
    "interview": ["面试", "面试题", "追问", "岗位题"],
    "billing": ["免费", "付费", "套餐", "价格", "退款", "扣费", "支付", "续费", "发票", "订单", "开通"],
    "privacy": ["隐私", "数据", "删除", "训练", "导出", "保存", "个人信息"],
    "error": ["失败", "报错", "错误", "转圈", "卡住", "验证码", "邮件", "收不到", "没反应"],
    "ticket": ["人工", "工单", "客服", "催办", "申诉", "投诉"],
    "boundary": ["法律", "医疗", "投资", "理财", "offer", "保证", "成功率", "录用", "腾讯", "阿里", "找到工作"],
}


@dataclass
class IndexedItem:
    item: KnowledgeItem
    embedding: object
    tokens: set[str]


class LocalVectorStore:
    def __init__(self, embedding_model: HashEmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.index: list[IndexedItem] = []

    def build(self, items: list[KnowledgeItem]) -> None:
        self.index = []
        for item in items:
            searchable_text = self._searchable_text(item)
            self.index.append(
                IndexedItem(
                    item=item,
                    embedding=self.embedding_model.embed(searchable_text),
                    tokens=set(tokenize(searchable_text)),
                )
            )

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        query_embedding = self.embedding_model.embed(query)
        query_tokens = set(tokenize(query))
        hits: list[SearchHit] = []

        for indexed in self.index:
            semantic_score = cosine_similarity(query_embedding, indexed.embedding)
            keyword_matches = [
                keyword
                for keyword in indexed.item.keywords
                if keyword.lower() in query.lower() or query.lower() in keyword.lower()
            ]
            token_overlap = len(query_tokens & indexed.tokens) / max(len(query_tokens), 1)
            keyword_boost = min(0.22, 0.055 * len(keyword_matches))
            category_boost = self._category_boost(query, indexed.item.category)
            title_boost = 0.08 if any(token in indexed.item.title.lower() for token in query_tokens if len(token) > 1) else 0.0
            score = semantic_score * 0.56 + token_overlap * 0.24 + keyword_boost + category_boost + title_boost
            hits.append(
                SearchHit(
                    item=indexed.item,
                    score=round(float(score), 4),
                    matched_keywords=keyword_matches,
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
        return min(0.24, 0.06 * len(matches))
