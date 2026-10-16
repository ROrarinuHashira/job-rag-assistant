"""求职岗位 RAG 问答服务（FastAPI）。

端点：
  GET  /                    简易 Web 界面
  GET  /api/health          健康检查 + 索引/模型状态
  POST /api/index           构建/重建向量索引（首次必须调一次）
  POST /api/query           提问，返回「检索结果 + 模型回答」
  GET  /api/docs            列出语料里的岗位（分页）

设计说明：
  - **模型懒加载**：启动时不加载模型，避免服务起不来；`/api/index` 会触发加载。
  - **BM25 永远可用**：即使向量索引没建、模型没加载，也能用关键词检索回答问题，
    只是没有语义召回（`retrieval_mode` 字段会告诉调用方当前是哪一种）。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)

from bm25 import BM25                      # noqa: E402
from hybrid import HybridRetriever         # noqa: E402
from retriever import Document, VectorIndex, load_corpus   # noqa: E402
from generator import Generator, build_context             # noqa: E402

INDEX_PATH = os.path.join(HERE, 'data', 'index.npz')

app = FastAPI(title='求职岗位 RAG 助手', version='1.0.0')


class AppState:
    """进程内状态。模型/索引都懒加载。"""

    def __init__(self):
        self.docs: List[Document] = load_corpus()
        self.embedder = None
        self.vector_index: Optional[VectorIndex] = None
        self.generator = None
        self.retriever = HybridRetriever(self.docs)   # 先只挂 BM25
        self.load_error: Optional[str] = None
        self.index_built_at: Optional[float] = None
        self.index_build_seconds: Optional[float] = None

    # ---------- 懒加载 ----------
    def ensure_embedder(self):
        if self.embedder is None:
            from embedder import Embedder
            self.embedder = Embedder()
        return self.embedder

    def ensure_generator(self):
        if self.generator is None:
            self.generator = Generator()
        return self.generator

    def build_index(self, force: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        emb = self.ensure_embedder()
        self.vector_index = VectorIndex.build(self.docs, emb)
        self.vector_index.save(INDEX_PATH)
        self.retriever = HybridRetriever(self.docs, vector_index=self.vector_index)
        self.index_built_at = time.time()
        self.index_build_seconds = round(self.index_built_at - t0, 2)
        return {
            'ok': True,
            'backend': emb.backend,
            'dim': emb.dim,
            'docs': len(self.docs),
            'seconds': self.index_build_seconds,
        }

    def try_load_index(self) -> bool:
        if not os.path.exists(INDEX_PATH):
            return False
        try:
            self.vector_index = VectorIndex.load(INDEX_PATH)
            self.retriever = HybridRetriever(self.docs, vector_index=self.vector_index)
            self.index_built_at = os.path.getmtime(INDEX_PATH)
            return True
        except Exception as e:
            self.load_error = f'索引加载失败：{e}'
            return False

    @property
    def retrieval_mode(self) -> str:
        return 'hybrid(bm25+vector)' if self.vector_index is not None else 'bm25-only'


state = AppState()
state.try_load_index()


# ---------------- 数据模型 ----------------
class QueryIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description='用户问题')
    top_k: int = Field(5, ge=1, le=20, description='检索条数')
    temperature: float = Field(0.3, ge=0.0, le=1.5, description='生成温度')
    use_llm: bool = Field(True, description='是否调用本地模型生成回答（false 则只返回检索结果）')


# ---------------- 端点 ----------------
@app.get('/', response_class=HTMLResponse)
def home():
    html_path = os.path.join(HERE, 'static', 'index.html')
    if os.path.exists(html_path):
        with open(html_path, encoding='utf-8') as f:
            return HTMLResponse(f.read())
    return HTMLResponse('<h1>求职岗位 RAG 助手</h1><p>缺少 static/index.html</p>')


@app.get('/api/health')
def health():
    return {
        'ok': True,
        'docs': len(state.docs),
        'docs_with_jd': sum(1 for d in state.docs if d.meta.get('jd')),
        'retrieval_mode': state.retrieval_mode,
        'vector_index': state.vector_index is not None,
        'embedder_loaded': state.embedder is not None,
        'generator_loaded': state.generator is not None,
        'index_built_at': state.index_built_at,
        'load_error': state.load_error,
    }


@app.post('/api/index')
def build_index(force: bool = False):
    try:
        return state.build_index(force=force)
    except Exception as e:
        state.load_error = str(e)
        raise HTTPException(status_code=500, detail=f'建索引失败：{e}')


@app.post('/api/query')
def query(body: QueryIn):
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail='question 不能为空')

    # 1) 检索
    qv = None
    if state.vector_index is not None:
        try:
            qv = state.ensure_embedder().encode_one(q)
        except Exception as e:
            state.load_error = f'向量化失败，已退回 BM25：{e}'
    t_ret = time.time()
    hits = state.retriever.search(q, query_vector=qv, top_k=body.top_k)
    retrieval_ms = round((time.time() - t_ret) * 1000, 1)

    # 2) 生成
    answer = None
    llm_ms = None
    llm_error = None
    if body.use_llm and hits:
        try:
            t0 = time.time()
            answer = state.ensure_generator().answer(q, hits, temperature=body.temperature)
            llm_ms = round((time.time() - t0) * 1000, 1)
        except Exception as e:
            llm_error = str(e)

    return {
        'ok': True,
        'question': q,
        'retrieval_mode': state.retrieval_mode,
        'retrieval_ms': retrieval_ms,
        'llm_ms': llm_ms,
        'llm_error': llm_error,
        'answer': answer,
        'context_preview': build_context(hits)[:400] if hits else '',
        'hits': [
            {
                'rank': i,
                'score': round(h['score'], 6),
                'bm25_rank': h.get('bm25_rank'),
                'vector_rank': h.get('vector_rank'),
                'id': h['doc'].id,
                'title': h['doc'].title,
                'company': h['doc'].company,
                'salary': h['doc'].salary,
                'location': h['doc'].location,
                'has_jd': bool(h['doc'].meta.get('jd')),
                'snippet': h['doc'].text[:200].replace('\n', ' '),
            }
            for i, h in enumerate(hits, 1)
        ],
    }


@app.get('/api/docs')
def list_docs(offset: int = 0, limit: int = 20, q: str = ''):
    docs = state.docs
    if q:
        bm = BM25([d.text for d in docs])
        picked = [i for i, sc in bm.search(q, top_k=len(docs)) if sc > 0]
        docs = [docs[i] for i in picked]
    limit = max(1, min(limit, 100))
    page = docs[offset:offset + limit]
    return {
        'total': len(docs),
        'offset': offset,
        'limit': limit,
        'items': [
            {
                'id': d.id,
                'title': d.title,
                'company': d.company,
                'salary': d.salary,
                'location': d.location,
                'has_jd': bool(d.meta.get('jd')),
            }
            for d in page
        ],
    }


if __name__ == '__main__':
    import uvicorn
    print('启动 RAG 服务：http://127.0.0.1:8020')
    print(f'  语料 {len(state.docs)} 篇，检索模式 {state.retrieval_mode}')
    uvicorn.run(app, host='127.0.0.1', port=8020, log_level='info')
