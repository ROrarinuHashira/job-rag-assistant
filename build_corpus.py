"""把抓下来的岗位数据合并成 RAG 语料（每条 = 一个「文档」）。

三类来源合并，按优先级去重：
  1. jobs-changsha.json  —— 长沙重点岗位，含完整 JD   ← 信息最全
  2. jobs-detail.json    —— 岗位详情，含完整 JD
  3. jobs-candidates.json—— 岗位列表信息             ← 覆盖面广

输出：data/jobs.jsonl  （每行一个 JSON：{id, title, company, salary, location, tags, text}）

数据来源目录（按顺序尝试，取第一个存在的）：
  1. 命令行第一个参数
  2. 环境变量 JOBS_RAW_DIR
  3. ../career/data          （本机开发时的位置）
  4. ./data                  （把原始 json 放在项目 data/ 下）
如果都找不到，脚本会提示并退出。
"""
import json
import os
import re
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def resolve_source_dir():
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        return sys.argv[1]
    env = os.environ.get('JOBS_RAW_DIR')
    if env and os.path.isdir(env):
        return env
    for cand in (
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'career', 'data'),
        DATA,
    ):
        if os.path.isdir(cand):
            return cand
    return None


CAREER = resolve_source_dir()


def load(name):
    if CAREER is None:
        return []
    path = os.path.join(CAREER, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d if isinstance(d, list) else d.get('jobs', [])

def norm_title(s):
    """标题归一化，用于去重。"""
    return re.sub(r'[\s\-_/|（）()【】\[\]]+', '', str(s or '')).lower()

docs = {}
seen = {}

def add(doc, priority):
    key = (norm_title(doc['title']), norm_title(doc.get('company', '')))
    if key in seen:
        # 已有更完整的信息就不覆盖
        if seen[key] <= priority:
            return
    # 空标题跳过
    if not doc['title'].strip():
        return
    seen[key] = priority
    docs[doc['id']] = doc

# ---- 1) jobs-changsha.json（完整 JD，最高优先级）----
for i, j in enumerate(load('jobs-changsha.json')):
    jd = str(j.get('jd') or '').strip()
    if not jd:
        continue
    add({
        'id': f'cs-{i+1:03d}',
        'title': str(j.get('name') or '').strip(),
        'company': str(j.get('company') or '').strip(),
        'salary': str(j.get('salary') or '').strip(),
        'location': str(j.get('address') or j.get('meta') or '').strip(),
        'tags': str(j.get('tags') or '').strip()[:400],
        'jd': jd,
        'text': '\n'.join(x for x in [
            f"岗位：{j.get('name','')}",
            f"公司：{j.get('company','')}" if j.get('company') else '',
            f"薪资：{j.get('salary','')}" if j.get('salary') else '',
            f"地点：{j.get('address') or j.get('meta') or ''}",
            f"标签：{str(j.get('tags') or '')[:200]}",
            f"职位描述：\n{jd}",
        ] if x.strip()),
        'source': 'changsha',
    }, 3)

# ---- 2) jobs-detail.json（完整 JD）----
for i, j in enumerate(load('jobs-detail.json')):
    jd = str(j.get('JD') or j.get('jd') or '').strip()
    if not jd:
        continue
    lines = j.get('列表行') or []
    add({
        'id': f'dt-{i+1:03d}',
        'title': str(j.get('岗位') or (lines[0] if lines else '')).strip(),
        'company': str(lines[-2] if len(lines) >= 2 else '').strip(),
        'salary': str(j.get('薪资') or '').strip(),
        'location': str(j.get('地址') or '').strip(),
        'tags': str(j.get('标签') or '').strip()[:400],
        'jd': jd,
        'text': '\n'.join(x for x in [
            f"岗位：{j.get('岗位','')}",
            f"薪资：{j.get('薪资','')}" if j.get('薪资') else '',
            f"地点：{j.get('地址','')}" if j.get('地址') else '',
            f"职位描述：\n{jd}",
        ] if x.strip()),
        'source': 'detail',
    }, 2)

# ---- 3) jobs-candidates.json（列表信息）----
for i, j in enumerate(load('jobs-candidates.json')):
    lines = [str(x).strip() for x in (j.get('lines') or []) if str(x).strip()]
    if not lines:
        continue
    title = str(j.get('name') or lines[0]).strip()
    salary = lines[1] if len(lines) > 1 else ''
    loc = next((l for l in lines if re.search(r'[·]|区|市', l)), '')
    company = lines[-2] if len(lines) >= 3 else ''
    add({
        'id': f'cd-{i+1:03d}',
        'title': title,
        'company': company,
        'salary': salary,
        'location': loc,
        'tags': ' / '.join(lines[2:-2]) if len(lines) > 4 else '',
        'jd': '',
        'text': '\n'.join(x for x in [
            f"岗位：{title}",
            f"薪资：{salary}" if salary else '',
            f"地点：{loc}" if loc else '',
            f"公司：{company}" if company else '',
            f"要求标签：{' / '.join(lines[2:-2])}" if len(lines) > 4 else '',
            f"（来源关键词：{j.get('kw','')}）" if j.get('kw') else '',
        ] if x.strip()),
        'source': 'candidate',
    }, 1)

os.makedirs(DATA, exist_ok=True)
out = os.path.join(DATA, 'jobs.jsonl')

if CAREER is None:
    print('找不到岗位原始数据目录。请把 jobs-changsha.json / jobs-detail.json /')
    print('jobs-candidates.json 放到以下任一位置：')
    print('  1) 用参数指定：python build_corpus.py <目录>')
    print('  2) 设置环境变量 JOBS_RAW_DIR=<目录>')
    print('  3) ../career/data  （本机开发时的位置）')
    print('  4) ./data          （把 json 放进项目 data/ 下）')
    print()
    print('没有原始数据时，可以直接用仓库自带的样例语料：')
    print(f'  data/jobs.sample.jsonl  →  复制成 {out} 即可跑通流程')
    sys.exit(1)

items = sorted(docs.values(), key=lambda d: d['id'])
with open(out, 'w', encoding='utf-8') as f:
    for d in items:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

print(f'语料已写入: {out}')
print(f'  文档数: {len(items)}')
from collections import Counter
c = Counter(d['source'] for d in items)
print(f'  来源分布: {dict(c)}')
print(f'  含完整 JD 的: {sum(1 for d in items if d["jd"])} 个')
print(f'  总字符数: {sum(len(d["text"]) for d in items):,}')
print()
print('--- 前 3 条预览 ---')
for d in items[:3]:
    print(f'  [{d["id"]}] {d["title"][:40]} | {d["salary"][:16]} | {d["location"][:20]}')
