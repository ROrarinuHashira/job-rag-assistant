"""BM25 关键词检索（自实现，只用 numpy + 标准库）。

为什么需要它：
  岗位检索里大量查询是**技能关键词**（Python / YOLO / RAG / LangChain），
  这类查询靠"字面精确匹配"比靠"语义相似度"准得多。
  而基座语言模型（Qwen2-1.5B）做 embedding 时区分度很差，所以关键词路必须补上。

中文分词策略（不依赖 jieba）：
  - 连续的 CJK 字符切成 bigram（"计算机视觉" → 计算/算机/机视/视觉）
  - 连续的 ASCII 单词保留原形并小写（"Python" → python）
  这个策略对"技能关键词命中"这个目标足够，且零依赖。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_CJK = r'\u4e00-\u9fff\u3400-\u4dbf'
_TOKEN_RE = re.compile(rf'[A-Za-z0-9_+#.]+|[{_CJK}]+')


def tokenize(text: str) -> List[str]:
    """英文按词、中文按 bigram。"""
    tokens: List[str] = []
    for m in _TOKEN_RE.finditer(str(text)):
        s = m.group(0)
        if re.fullmatch(rf'[{_CJK}]+', s):
            if len(s) == 1:
                tokens.append(s)
            else:
                tokens.extend(s[i:i + 2] for i in range(len(s) - 1))
        else:
            w = s.lower().strip('.')
            if len(w) >= 1:
                tokens.append(w)
    return tokens


class BM25:
    """标准 BM25（k1/b 可调），纯 Python 实现。"""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs_tokens: List[List[str]] = [tokenize(d) for d in corpus]
        self.n_docs = len(self.docs_tokens)
        self.doc_len = [len(t) for t in self.docs_tokens]
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        # 文档频率
        df: Counter = Counter()
        term_freqs: List[Counter] = []
        for toks in self.docs_tokens:
            tf = Counter(toks)
            term_freqs.append(tf)
            for term in tf:
                df[term] += 1
        self.term_freqs = term_freqs

        # idf（带平滑，避免负值）
        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def scores(self, query: str) -> List[float]:
        q_terms = tokenize(query)
        out = [0.0] * self.n_docs
        if not q_terms:
            return out
        for term in q_terms:
            idf = self.idf.get(term)
            if not idf:
                continue
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(term)
                if not f:
                    continue
                dl = self.doc_len[i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                out[i] += idf * (f * (self.k1 + 1)) / (denom or 1)
        return out

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        sc = self.scores(query)
        idx = sorted(range(len(sc)), key=lambda i: -sc[i])[:top_k]
        return [(i, sc[i]) for i in idx if sc[i] > 0]


if __name__ == '__main__':
    # 自测：用真实语料，验证关键词命中
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from retriever import load_corpus

    docs = load_corpus()
    bm = BM25([d.text for d in docs])
    print(f'语料 {len(docs)} 篇，词表大小 {len(bm.idf)}\n')

    TESTS = [
        ('哪些岗位要求 RAG 经验？', ['rag', 'langchain', '检索', '知识库']),
        ('要求 YOLO 或计算机视觉的岗位有哪些？', ['yolo', '视觉', '图像', '检测']),
        ('哪些岗位提到了 Python？', ['python']),
        ('长沙有哪些 AI 应用开发的岗位？', ['ai应用', 'ai 应用', '大模型', '算法']),
        ('有没有大模型微调相关的岗位？', ['微调', '后训练', '预训练']),
    ]
    total = 0
    for q, hints in TESTS:
        res = bm.search(q, top_k=5)
        good = 0
        print(f'\n查询: {q}')
        for rank, (i, sc) in enumerate(res, 1):
            d = docs[i]
            blob = (d.title + ' ' + d.text).lower()
            hit = any(x in blob for x in hints)
            if hit:
                good += 1
            print(f'  {"✅" if hit else "  "} {rank}. [{sc:6.2f}] {d.title[:44]}')
        total += good
        print(f'  → {good}/{len(res)} 相关')
    print(f'\nBM25 总计: {total}/25 命中')
