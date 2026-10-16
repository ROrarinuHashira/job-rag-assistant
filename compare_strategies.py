"""对比不同检索策略的效果，找出最优方案。

策略：
  A. 全部文档 + 原始 embedding
  B. 只有 JD 的文档 + 原始 embedding
  C. 全部文档 + 去掉主成分（去各向异性）
  D. 只有 JD 的文档 + 去主成分
"""
import sys
import numpy as np

sys.path.insert(0, r'C:\Users\Rorarinu\Documents\deepseek-harness\default-workspace\rag')
from embedder import Embedder
from retriever import load_corpus, VectorIndex

# 每个查询 + 期望命中的关键词（用于自动评分）
QUERIES = [
    ('哪些岗位要求 RAG 经验？', ['rag', 'langchain', '检索', '知识库']),
    ('长沙有哪些 AI 应用开发的岗位？', ['ai应用', 'ai 应用', '大模型', '算法']),
    ('有没有面向 2027 届的算法实习？', ['27届', '2027', '实习', '应届', '在校']),
    ('要求 YOLO 或计算机视觉的岗位有哪些？', ['yolo', '视觉', '图像', '检测']),
    ('哪些岗位提到了 Python？', ['python']),
    ('有没有大模型微调相关的岗位？', ['微调', '后训练', '预训练', 'finetune']),
]


def score(idx: VectorIndex, emb, q, hints, top_k=5):
    hits = idx.search(emb.encode_one(q), top_k=top_k)
    good = 0
    for h in hits:
        blob = (h['doc'].title + ' ' + h['doc'].text).lower()
        if any(x in blob for x in hints):
            good += 1
    # 同时看分数区分度：top1 - top5 的差
    if len(hits) >= 2:
        spread = hits[0]['score'] - hits[-1]['score']
    else:
        spread = 0.0
    return good, spread, hits


def remove_first_pc(vectors: np.ndarray, n_components: int = 1) -> np.ndarray:
    """去掉前 n 个主成分（缓解各向异性），再重新归一化。"""
    v = vectors - vectors.mean(axis=0, keepdims=True)
    # SVD 取主方向
    _, _, vt = np.linalg.svd(v, full_matrices=False)
    for i in range(n_components):
        pc = vt[i]
        v = v - np.outer(v @ pc, pc)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.clip(norms, 1e-12, None)).astype(np.float32)


def main():
    emb = Embedder()
    docs_all = load_corpus()
    docs_jd = [d for d in docs_all if d.meta.get('jd')]
    print(f'模型={emb.backend} 全部文档={len(docs_all)} 有JD={len(docs_jd)}\n')

    print('向量化...')
    V_all = emb.encode([d.text for d in docs_all])
    V_jd = emb.encode([d.text for d in docs_jd])

    # 去主成分版本（注意：对全量算 PC，再应用到子集，保持一致）
    PC = 3
    V_all_pc = remove_first_pc(V_all, PC)
    # 子集用自己的 PC（因为它是独立的检索空间）
    V_jd_pc = remove_first_pc(V_jd, PC)

    strategies = {
        'A 全部+原始': VectorIndex(docs_all, V_all),
        'B 仅JD+原始': VectorIndex(docs_jd, V_jd),
        f'C 全部+去PC{PC}': VectorIndex(docs_all, V_all_pc),
        f'D 仅JD+去PC{PC}': VectorIndex(docs_jd, V_jd_pc),
    }

    print('=' * 78)
    results = {}
    for name, idx in strategies.items():
        total = 0
        spreads = []
        per_q = []
        for q, hints in QUERIES:
            good, spread, _ = score(idx, emb, q, hints)
            total += good
            spreads.append(spread)
            per_q.append(good)
        results[name] = (total, per_q, float(np.mean(spreads)))
        print(f'{name:18s} 总命中 {total:2d}/{len(QUERIES)*5}  平均区分度 {np.mean(spreads):.3f}  逐条 {per_q}')

    print('\n' + '=' * 78)
    best = max(results.items(), key=lambda kv: (kv[1][0], kv[1][2]))
    print(f'最佳策略: {best[0]}  （命中 {best[1][0]}/{len(QUERIES)*5}，区分度 {best[1][2]:.3f}）')

    print('\n=== 最佳策略的详细结果 ===')
    idx = strategies[best[0]]
    for q, hints in QUERIES:
        good, spread, hits = score(idx, emb, q, hints)
        print(f'\n查询: {q}   ({good}/5 相关, 区分度 {spread:.3f})')
        for i, h in enumerate(hits, 1):
            d = h['doc']
            blob = (d.title + ' ' + d.text).lower()
            mark = '✅' if any(x in blob for x in hints) else '  '
            print(f'  {mark} {i}. [{h["score"]:.3f}] {d.title[:44]}')


if __name__ == '__main__':
    main()
