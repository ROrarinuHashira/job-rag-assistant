"""检索模块：向量索引 + 相似度检索（纯 numpy 实现，无需 FAISS）。

为什么不直接依赖 FAISS：
  - 语料只有几十到几千条，numpy 点积足够快（毫秒级）
  - 少一个二进制依赖，换机器更容易跑起来
  - 需要时可以平滑替换（接口一致）
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')


@dataclass
class Document:
    id: str
    title: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def company(self) -> str:
        return str(self.meta.get('company', ''))

    @property
    def salary(self) -> str:
        return str(self.meta.get('salary', ''))

    @property
    def location(self) -> str:
        return str(self.meta.get('location', ''))


class VectorIndex:
    """文档 + 向量 + 检索。向量为一维归一化后，内积即余弦相似度。"""

    def __init__(self, docs: List[Document], vectors: np.ndarray):
        if len(docs) != len(vectors):
            raise ValueError(f'文档数({len(docs)})与向量数({len(vectors)})不一致')
        self.docs = docs
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError('向量必须是二维数组')
        # 关键：构造时就按行归一化。
        # 否则 `vectors @ q` 算的是内积而不是余弦相似度 —— 模长大的向量会占优，
        # 排序会静默出错（这个坑自测时踩到过）。
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        self.vectors = arr / np.clip(norms, 1e-12, None)
        self.dim = int(self.vectors.shape[1]) if self.vectors.size else 0

    # ---------- 构造 / 持久化 ----------
    @classmethod
    def build(cls, docs: List[Document], embedder) -> 'VectorIndex':
        vectors = embedder.encode([d.text for d in docs])
        return cls(docs, vectors)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            docs=json.dumps(
                [{'id': d.id, 'title': d.title, 'text': d.text, 'meta': d.meta} for d in self.docs],
                ensure_ascii=False,
            ),
        )

    @classmethod
    def load(cls, path: str) -> 'VectorIndex':
        z = np.load(path, allow_pickle=False)
        raw = json.loads(str(z['docs']))
        docs = [Document(id=r['id'], title=r['title'], text=r['text'], meta=r.get('meta', {})) for r in raw]
        return cls(docs, z['vectors'])

    # ---------- 检索 ----------
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores = self.vectors @ q                       # 余弦相似度
        k = int(min(top_k, len(self.docs)))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        out = []
        for i in idx:
            out.append({'score': float(scores[i]), 'doc': self.docs[int(i)]})
        return out


def load_corpus(path: Optional[str] = None) -> List[Document]:
    """读取 build_corpus.py 生成的 jobs.jsonl。"""
    path = path or os.path.join(DATA, 'jobs.jsonl')
    docs: List[Document] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            docs.append(Document(
                id=r['id'],
                title=r['title'],
                text=r['text'],
                meta={k: v for k, v in r.items() if k not in ('id', 'title', 'text')},
            ))
    return docs


if __name__ == '__main__':
    # 用随机向量自测检索逻辑（不需要模型）
    rng = np.random.default_rng(0)
    docs = [Document(id=str(i), title=f'文档{i}', text=f'内容{i}') for i in range(10)]
    vecs = rng.normal(size=(10, 8)).astype(np.float32)
    idx = VectorIndex(docs, vecs)
    q = vecs[3] + rng.normal(scale=0.01, size=8).astype(np.float32)
    hits = idx.search(q, top_k=3)
    print('检索自测（查询最接近文档3）：')
    for h in hits:
        print(f"  {h['score']:.4f}  {h['doc'].title}")
    assert hits[0]['doc'].id == '3', '排序错误'
    print('✅ 检索逻辑正确')
