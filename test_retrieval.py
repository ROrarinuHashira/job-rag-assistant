"""用真实查询测检索质量（检索是 RAG 的命门，比生成更关键）。"""
import sys
import numpy as np
sys.path.insert(0, r'C:\Users\Rorarinu\Documents\deepseek-harness\default-workspace\rag')

from embedder import Embedder
from retriever import load_corpus, VectorIndex

print('加载模型 + 语料...')
emb = Embedder()
docs = load_corpus()
print(f'  后端={emb.backend} 维度={emb.dim} 文档数={len(docs)}')

print('向量化语料（可能几十秒）...')
idx = VectorIndex.build(docs, emb)
print('  完成，向量形状:', idx.vectors.shape)

QUERIES = [
    ('哪些岗位要求 RAG 经验？', ['RAG', 'LangChain', '大模型']),
    ('长沙有哪些 AI 应用开发的岗位？', ['长沙', 'AI应用']),
    ('有没有面向 2027 届的算法实习？', ['2027', '实习', '算法']),
    ('要求 YOLO 或计算机视觉的岗位有哪些？', ['YOLO', '视觉', '图像']),
    ('哪些岗位提到了 Python？', ['Python']),
]

for q, hints in QUERIES:
    print('\n' + '=' * 74)
    print(f'查询: {q}     (期望命中关键词: {", ".join(hints)})')
    qv = emb.encode_one(q)
    hits = idx.search(qv, top_k=5)
    for i, h in enumerate(hits, 1):
        d = h['doc']
        # 检查这条文档是否真的包含期望关键词
        blob = (d.title + d.text).lower()
        mark = '✅' if any(x.lower() in blob for x in hints) else '  '
        print(f'  {mark} {i}. [{h["score"]:.3f}] {d.title[:40]:42s} {d.salary[:12]:14s} {d.location[:16]}')

# 统计：前5命中里，包含期望关键词的比例
print('\n' + '=' * 74)
print('检索质量统计：')
for q, hints in QUERIES:
    qv = emb.encode_one(q)
    hits = idx.search(qv, top_k=5)
    good = sum(1 for h in hits if any(x.lower() in (h['doc'].title + h['doc'].text).lower() for x in hints))
    print(f'  {good}/5 相关  ← {q[:40]}')
