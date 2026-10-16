"""混合检索：BM25（关键词）+ 向量（语义），用 RRF 融合排名。

为什么用 RRF（Reciprocal Rank Fusion）而不是加权分数：
  BM25 分数（0~12+）与余弦相似度（-1~1）**量纲完全不同**，直接加权需要调参且不稳。
  RRF 只看「排名」，天然免疫量纲差异，是工业界融合多路召回的常用做法。

  公式：score(d) = Σ_r  weight_r / (k + rank_r(d))     k 通常取 60
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from bm25 import BM25
from retriever import Document, VectorIndex


class HybridRetriever:
    """BM25 + 向量 混合检索。

    - 只用 BM25：当向量不可用（或没装模型）时依然能工作
    - 只用向量：把 bm25_weight 设为 0
    - 默认两路都用，RRF 融合
    """

    def __init__(
        self,
        docs: List[Document],
        vector_index: Optional[VectorIndex] = None,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        rrf_k: int = 60,
    ):
        self.docs = docs
        self.vector_index = vector_index
        self.bm25 = BM25([d.text for d in docs])
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

    def search(self, query: str, query_vector=None, top_k: int = 5, pool: int = 20) -> List[Dict[str, Any]]:
        """返回 [{score, doc, bm25_rank, vector_rank, bm25_score, vector_score}]"""
        n = len(self.docs)
        pool = max(min(pool, n), top_k)

        # --- 第一路：BM25 ---
        bm_scores = self.bm25.scores(query)
        bm_rank = {}
        for pos, i in enumerate(sorted(range(n), key=lambda i: -bm_scores[i])[:pool]):
            bm_rank[i] = pos + 1

        # --- 第二路：向量 ---
        vec_rank = {}
        vec_scores = {}
        if self.vector_index is not None and query_vector is not None:
            hits = self.vector_index.search(query_vector, top_k=pool)
            for pos, h in enumerate(hits, 1):
                # 用 doc.id 反查在当前 docs 列表里的位置
                idx = self._id_to_pos.get(h['doc'].id)
                if idx is None:
                    continue
                vec_rank[idx] = pos
                vec_scores[idx] = h['score']

        # --- RRF 融合 ---
        fused: Dict[int, float] = {}
        for i, r in bm_rank.items():
            fused[i] = fused.get(i, 0.0) + self.bm25_weight / (self.rrf_k + r)
        for i, r in vec_rank.items():
            fused[i] = fused.get(i, 0.0) + self.vector_weight / (self.rrf_k + r)

        out = []
        for i, sc in sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]:
            out.append({
                'score': float(sc),
                'doc': self.docs[i],
                'bm25_rank': bm_rank.get(i),
                'vector_rank': vec_rank.get(i),
                'bm25_score': float(bm_scores[i]),
                'vector_score': vec_scores.get(i),
            })
        return out

    @property
    def _id_to_pos(self) -> Dict[str, int]:
        if not hasattr(self, '__id_pos'):
            self.__id_pos = {d.id: i for i, d in enumerate(self.docs)}
        return self.__id_pos


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from retriever import load_corpus

    docs = load_corpus()
    hy = HybridRetriever(docs)   # 只有 BM25 的版本（不需要模型）
    print(f'混合检索自测（无向量路，仅 BM25）：{len(docs)} 篇\n')

    for q in ['要求 YOLO 或计算机视觉的岗位有哪些？', '哪些岗位提到了 Python？', '长沙的 AI 应用开发岗位']:
        print(f'查询: {q}')
        for rank, h in enumerate(hy.search(q, top_k=4), 1):
            print(f'  {rank}. rrf={h["score"]:.5f} bm25排名={h["bm25_rank"]:<3} {h["doc"].title[:46]}')
        print()
    print('✅ 混合检索框架可用（向量路在 Web 服务里启用）')
