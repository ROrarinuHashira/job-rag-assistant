"""生成模块：把「检索到的文档 + 用户问题」交给本地 Qwen 模型，生成回答。

设计要点：
  - **Prompt 必须极简 + 给示例**：1.5B 小模型对"多条规则"的遵循能力很差，
    实测写复杂 system prompt 会让它只输出 "[1]" 就停；**给一个示例**才稳定。
    （细节见下面 USER_TEMPLATE 的注释 —— 这是本项目踩过的坑之一）
  - 上下文有长度限制，所以对检索文档做截断
  - 生成结果可能与检索无关时，服务端会把检索结果一并返回，不依赖模型兜底
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List

DEFAULT_MODEL_DIR = r'F:\qwen_1.5b_local'

SYSTEM_PROMPT = '你是一个求职助手，根据岗位信息回答问题。'

# 说明（踩坑记录，四轮迭代 —— 本项目最有价值的调试经验）：
#   一版：多条规则 + 强制标注 [n] → 1.5B **只输出 "[1]" 就停**（规则越多越跑偏）
#   二版：极简指令 → 答对了但**太短**（只回一个岗位名）
#   三版：给完整「问/答」示例 → 模型**原样复读示例**（few-shot 变复读）
#   四版：只给"格式要求"并附带"没有内容就说…" → 模型**一律回答"没有找到相关岗位"**
#        （那句条件句给了它偷懒的出口 —— 小模型会直接选最好答的分支）
#   当前版：去掉所有条件分支，只留一个明确动作："列出相关岗位"。
USER_TEMPLATE = """下面是检索到的岗位信息：
{context}

问题：{question}

请列出与问题相关的岗位，写明岗位名称、公司、地点和关键要求：
"""

# 输出清洗：小模型容易复读「问：…答：」「【回答】」这类结构标记，去掉它们
_ECHO_PREFIXES = ('问：', '答：', '【回答】', '参考回答', '要求：', '示例：')


def clean_answer(text: str) -> str:
    """去掉模型复读出来的结构标记与自问自答。"""
    if not text:
        return ''
    lines = []
    for raw in str(text).split('\n'):
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r'^【回答】\s*', '', line)
        # 丢掉模型自己复读的「问：xxx」行
        if line.startswith('问：'):
            continue
        line = re.sub(r'^答：\s*', '', line)
        # 丢掉与模板要求雷同的说明行
        if any(line.startswith(p) for p in ('要求：', '参考回答', '示例：')):
            continue
        lines.append(line)
    out = '\n'.join(lines).strip()
    # 若清洗后为空（模型只吐了标记），回退到原文
    return out or str(text).strip()


def build_context(hits: List[Dict[str, Any]], max_chars_per_doc: int = 320) -> str:
    """把检索结果拼成给模型的参考资料。

    注意 max_chars_per_doc 故意取得不大：CPU 推理时上下文越长生成越慢
    （700 字时单次要 40~95 秒，320 字能明显改善），而岗位关键信息都在开头。
    """
    parts = []
    for i, h in enumerate(hits, 1):
        d = h['doc']
        head = f"[{i}] {d.title}"
        meta = []
        if d.company:
            meta.append(d.company)
        if d.salary:
            meta.append(d.salary)
        if d.location:
            meta.append(d.location)
        if meta:
            head += '（' + ' / '.join(meta) + '）'
        body = d.text[:max_chars_per_doc]
        parts.append(f'{head}\n{body}')
    return '\n\n'.join(parts)


def build_messages(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context = build_context(hits) if hits else '（没有找到相关岗位）'
    user = USER_TEMPLATE.format(context=context, question=question)
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user},
    ]


class Generator:
    """本地 Qwen 文本生成。懒加载。"""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, max_new_tokens: int = 200):
        # max_new_tokens 取 200：答案本质是「列出岗位」，不需要长文；
        # CPU 上每 token 都要时间，限制长度是这里最有效的提速手段。
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not os.path.isdir(self.model_dir):
            raise RuntimeError(f'本地模型不存在：{self.model_dir}')

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float32,   # CPU 上用 float32
            trust_remote_code=True,
        )
        self._model.eval()

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        self._ensure_loaded()
        torch = self._torch
        # Qwen2 系列自带 chat 模板
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors='pt')
        # 停顿早退：模型开始重复上一步输出时提前停止。
        # 小模型在 CPU 上常出现「问：… 答：…」复读循环，不打断会白烧几十秒。
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _RepeatGuard(StoppingCriteria):
            def __init__(self, prompt_len: int):
                self.prompt_len = prompt_len
                self.seen = None

            def __call__(self, input_ids, scores, **kwargs):
                cur = input_ids[0][self.prompt_len:]
                if self.seen is not None and len(cur) > 24 and cur[-24:].tolist() == self.seen:
                    return True
                self.seen = cur[-24:].tolist() if len(cur) >= 24 else None
                return False

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([_RepeatGuard(inputs['input_ids'].shape[1])]),
            )
        gen = out[0][inputs['input_ids'].shape[1]:]
        raw = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
        return clean_answer(raw)

    def answer(self, question: str, hits: List[Dict[str, Any]], temperature: float = 0.3) -> str:
        return self.generate(build_messages(question, hits), temperature=temperature)


if __name__ == '__main__':
    # Prompt 构造自测（不需要模型）
    class _D:
        def __init__(self, title, text, company='', salary='', location=''):
            self.title, self.text, self.company, self.salary, self.location = title, text, company, salary, location
            self.id = title

    hits = [
        {'score': 0.9, 'doc': _D('AI应用工程师', '要求熟悉 Python 与 RAG 检索。', '华为', '18-35K', '长沙')},
        {'score': 0.8, 'doc': _D('大模型应用开发', '需要 LangChain 与向量数据库经验。', '泛联新安', '15-30K', '长沙')},
    ]
    msgs = build_messages('哪些岗位要求 RAG 经验？', hits)
    print('=== System ===')
    print(msgs[0]['content'][:200])
    print('\n=== User（前 700 字）===')
    print(msgs[1]['content'][:700])
    print('\n✅ Prompt 构造正常')
