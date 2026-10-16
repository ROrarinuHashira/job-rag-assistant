"""向量化模块：把文本编码成向量。

设计：优先用本地 Qwen 模型做 mean pooling；如果装了 sentence-transformers，
则优先用它（检索质量通常更好）。
"""
from __future__ import annotations

import os
from typing import Iterable, List

import numpy as np

DEFAULT_MODEL_DIR = r'F:\qwen_1.5b_local'

# Qwen2 的 padding token 与 eos 共用，做 mean pooling 时必须排除 pad
_PAD_TOKEN_FALLBACK = '<|endoftext|>'


class Embedder:
    """文本 → 向量。首次调用时懒加载模型。"""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, backend: str = 'auto', max_length: int = 1024):
        self.model_dir = model_dir
        self.requested_backend = backend
        self.max_length = max_length
        self.backend = None
        self._model = None
        self._tokenizer = None
        self.dim = None
        self._load()

    # ---------- 加载 ----------
    def _load(self):
        if self.requested_backend in ('auto', 'sentence-transformers'):
            if self._try_sentence_transformers():
                return
        if self.requested_backend in ('auto', 'qwen'):
            if self._try_qwen():
                return
        raise RuntimeError(
            '没有可用的编码后端。请确认：\n'
            f'  1) 本地模型存在：{self.model_dir}\n'
            '  2) 或安装 sentence-transformers：pip install sentence-transformers'
        )

    def _try_sentence_transformers(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return False
        try:
            # 若本地目录存在且看起来是模型，就用它；否则退回一个常用小模型
            target = self.model_dir if os.path.isdir(self.model_dir) else 'shibing624/text2vec-base-chinese'
            self._model = SentenceTransformer(target)
            self.backend = 'sentence-transformers'
            self.dim = int(self._model.get_sentence_embedding_dimension())
            return True
        except Exception as e:
            print(f'[embedder] sentence-transformers 加载失败，将尝试 qwen：{e}')
            return False

    def _try_qwen(self) -> bool:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not os.path.isdir(self.model_dir):
            return False

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float32,   # CPU 上 float32 更稳
            trust_remote_code=True,
        )
        self._model.eval()
        self.backend = 'qwen'
        self._torch = torch
        # 取模型隐藏维度
        self.dim = int(getattr(self._model.config, 'hidden_size', 1536))
        return True

    # ---------- 编码 ----------
    def encode(self, texts: Iterable[str], batch_size: int = 8, normalize: bool = True) -> np.ndarray:
        texts = [str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)

        if self.backend == 'sentence-transformers':
            vecs = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=normalize)
            return np.asarray(vecs, dtype=np.float32)

        # ---- Qwen + mean pooling ----
        torch = self._torch
        out: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            with torch.no_grad():
                model_out = self._model(**enc)
            hidden = model_out.last_hidden_state                      # (B, L, H)
            mask = enc['attention_mask'].unsqueeze(-1).float()        # (B, L, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            mean = summed / counts                                    # (B, H)
            arr = mean.cpu().numpy().astype(np.float32)
            out.append(arr)
        vecs = np.vstack(out)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-12, None)
        return vecs

    def encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]


if __name__ == '__main__':
    # 自测
    import sys
    e = Embedder()
    print('后端:', e.backend, ' 维度:', e.dim)
    v = e.encode(['心脏病的机器学习分类', 'YOLO 目标检测的苹果成熟度识别', '今天天气不错'])
    print('形状:', v.shape, ' 已归一化:', np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4))
    sim = v @ v.T
    print('相似度矩阵:')
    for row in sim:
        print('  ' + '  '.join(f'{x:+.3f}' for x in row))
    print(f'前两句相似度 {sim[0,1]:+.3f}，第1句与第3句 {sim[0,2]:+.3f}')
