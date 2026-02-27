import jieba
from rank_bm25 import BM25Okapi
from typing import List, Tuple


class BM25Searcher:
    """BM25 全文搜索"""

    def __init__(self):
        self.corpus: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25 = None
        self.doc_ids: List[str] = []

    def build_index(self, documents: List[dict]):
        """构建 BM25 索引

        documents: [{"id": "doc_id", "text": "content"}, ...]
        """
        self.corpus = [doc["text"] for doc in documents]
        self.doc_ids = [doc["id"] for doc in documents]

        # 中文分词
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]

        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        return len(self.corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """搜索

        返回: [(doc_id, score), ...]
        """
        if not self.bm25:
            return []

        # 分词查询
        tokenized_query = list(jieba.cut(query))

        # 获取 BM25 分数
        scores = self.bm25.get_scores(tokenized_query)

        # 获取 top_k 结果
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = [(self.doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]

        return results

    def clear(self):
        """清空索引"""
        self.corpus = []
        self.tokenized_corpus = []
        self.doc_ids = []
        self.bm25 = None
