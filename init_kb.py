"""Initialize and enrich the local knowledge base."""

import hashlib
import json
from pathlib import Path
from typing import Any

import requests

from KnowledgeBase import KnowledgeBase

BASE_API = "https://plankbevelen.cn/api/article"
BLOG_HOST = "https://plankbevelen.cn"
KB_COLLECTION = "plankbevelen"


def add_seed_docs(kb: KnowledgeBase) -> int:
  added = 0

  seed_docs = [
    (
      "site_plankbevelen_cn",
      """
网站名称：plankbevelen.cn
网站类型：个人技术博客与实践记录站
网站定位：
- 记录真实技术学习路径，而不是只展示结果
- 输出可复现、可落地的工程经验
- 结合 AI Agent 与传统工程实践，强调“能跑起来、能长期维护”
主要受众：
- 前端/全栈工程师
- 对 AI Agent、RAG、自动化流程感兴趣的开发者
- 希望从“会写 Demo”走向“可交付工程”的学习者
内容结构：
- 项目实战复盘（从需求到部署）
- 踩坑记录（环境、依赖、性能、兼容）
- AI 工程化（Prompt、检索、评估、观测）
- 工具链与效率流（脚手架、自动化、模板化）
内容风格：
- 不追热点，不标题党
- 优先讲清“为什么这么做”
- 强调边界条件与失败案例
核心关键词：Python、AI Agent、RAG、LLM、Vue、React、Linux、云服务器、工程化
""".strip(),
      {"source": "https://plankbevelen.cn", "type": "blog", "author": "PlankBevelen"},
    ),
    (
      "site_tool_plankbevelen_cn",
      """
网站名称：tool.plankbevelen.cn
网站类型：在线工具集合
网站定位：简洁、无广告、轻量、高效的免费工具箱。
工具覆盖：开发工具、文本处理、格式转换、计算工具、效率工具、前端调试、数据处理。
""".strip(),
      {"source": "https://tool.plankbevelen.cn", "type": "tool", "author": "PlankBevelen"},
    ),
    (
      "author_intro",
      """
作者名称：PlankBevelen
身份：全栈开发者、AI Agent 研究者、前端工程师、技术创作者
擅长方向：Vue、React、Flutter、WebGL、Python、大模型、AI Agent、Linux、云服务器、工具开发
创作理念：真实、实用、可落地。
工作方式：
- 先定义可验证目标，再做实现
- 先打通最小可用路径，再做优化与抽象
- 先保证稳定性与可维护性，再追求“炫技”
价值观：
- 对结果负责：能上线、可维护、可复盘
- 对细节诚实：不回避不确定性，不掩盖风险
- 对读者友好：给步骤、给边界、给排错线索
""".strip(),
      {"source": "author", "type": "profile"},
    ),
    (
      "author_identity_alias",
      """
身份别名与检索别称：
- 主名称：PlankBevelen
- 常见写法：plankbevelen、Plank、PB
- 站点关联：plankbevelen.cn、tool.plankbevelen.cn
当用户问题涉及“你是谁 / 你在做什么 / 你的风格 / 你的博客 / 你的工具站”时，可优先关联本条信息。
如果问题缺少上下文，先给简明身份说明，再给对应链接或可执行下一步建议。
""".strip(),
      {"source": "author", "type": "identity"},
    ),
    (
      "author_tech_stack",
      """
技术栈画像（按工程职责划分）：
1) 前端与应用层
- Vue、React、TypeScript、Flutter、WebGL
2) 后端与 AI 服务层
- Python、Flask/FastAPI、任务编排、接口封装
3) AI/LLM 工程
- Prompt 设计、RAG 检索、工具调用（Tool Use）、评估与回归
4) 系统与部署
- Linux、云服务器、进程管理、日志与监控
5) 工程质量
- 可复现环境、脚本化初始化、性能与稳定性排查
技术选择原则：
- 优先成熟稳定方案
- 避免过度设计
- 以交付效率与长期维护成本作为主要判断标准
""".strip(),
      {"source": "author", "type": "stack"},
    ),
    (
      "author_projects_overview",
      """
代表性项目与方向：
1) plankbevelen.cn
- 定位：技术博客与实践记录
- 目标：沉淀可复用方法，而非一次性答案
2) tool.plankbevelen.cn
- 定位：轻量在线工具集
- 特点：打开即用、低门槛、聚焦效率
3) Agent/RAG 实验项目
- 目标：把 LLM 能力落到可观测、可评估、可迭代的工程中
- 重点：检索质量、响应时延、稳定性、失败兜底策略
""".strip(),
      {"source": "author", "type": "projects"},
    ),
    (
      "author_response_style",
      """
PlankBevelen 回答风格约定：
- 先结论，后细节
- 直说可行方案，不绕弯
- 信息不足时明确不确定点，不编造事实
- 优先给可执行步骤与排错路径
- 对工程问题优先考虑：稳定性、可维护性、可复现
禁忌：
- 不用“空话式鼓励”替代实操建议
- 不把系统内部状态当作对用户的主要回答
- 不在未知事实上给出确定性口吻
""".strip(),
      {"source": "author", "type": "style"},
    ),
  ]

  for doc_id, text, metadata in seed_docs:
    if kb.add(doc_id, text, metadata, force=False):
      added += 1
  return added


def crawl_plankbevelen_by_page(page: int = 1, limit: int = 100) -> tuple[list[dict[str, Any]], int]:
  try:
    url = f"{BASE_API}?page={page}&limit={limit}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("data", []) or [], int(data.get("total", 0) or 0)
  except Exception:
    return [], 0


def fetch_plankbevelen_articles() -> list[dict[str, Any]]:
  all_articles: list[dict[str, Any]] = []
  page = 1
  limit = 100
  while True:
    articles, _total = crawl_plankbevelen_by_page(page=page, limit=limit)
    if not articles:
      break
    all_articles.extend(articles)
    page += 1
  return all_articles


def parse_article(art: dict[str, Any]) -> dict[str, Any] | None:
  try:
    title = art.get("title", "无标题")
    short_content = art.get("shortContent", "")
    long_content = art.get("longContent", "")
    slug = art.get("slug", f"{BLOG_HOST}/article/{art.get('id', '')}")
    url = slug
    tags = art.get("tags", [])
    category = art.get("category", "")
    create_time = art.get("create_time", "")
    update_time = art.get("update_time", "")
    return {
      "title": title,
      "shortContent": short_content,
      "longContent": long_content,
      "url": url,
      "tags": tags,
      "category": category,
      "create_time": create_time,
      "update_time": update_time,
    }
  except Exception:
    return None


def ingest_blog_articles(kb: KnowledgeBase) -> int:
  articles = fetch_plankbevelen_articles()
  added = 0
  print(f"[blog] fetched articles={len(articles)}")

  for idx, art in enumerate(articles):
    item = parse_article(art)
    if item is None:
      continue
    doc_id = f"blog_{idx + 1:03d}"
    text = (
      f"文章标题：{item['title']}\n"
      f"文章链接：{item['url']}\n"
      f"文章摘要：{item['shortContent']}\n"
      f"文章内容：{item['longContent']}\n"
      f"文章分类：{item['category']}\n"
      f"文章标签：{item['tags']}\n"
      f"文章创建时间：{item['create_time']}\n"
      f"文章更新时间：{item['update_time']}"
    )
    metadata = {
      "source": item["url"],
      "type": "blog_article",
      "title": item["title"],
      "category": item["category"],
      "tags": item["tags"],
      "create_time": item["create_time"],
      "update_time": item["update_time"],
      "content": item["longContent"],
      "shortContent": item["shortContent"],
    }
    if kb.add(doc_id, text, metadata, force=False):
      added += 1
  return added


def chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
  text = (text or "").strip()
  if not text:
    return []
  chunks: list[str] = []
  start = 0
  n = len(text)
  while start < n:
    end = min(n, start + size)
    chunk = text[start:end].strip()
    if chunk:
      chunks.append(chunk)
    if end >= n:
      break
    start = max(start + size - overlap, start + 1)
  return chunks


def read_text_file(path: Path) -> str:
  for enc in ("utf-8", "utf-8-sig", "gbk"):
    try:
      return path.read_text(encoding=enc)
    except Exception:
      continue
  return path.read_text(encoding="utf-8", errors="ignore")


def ingest_local_files(kb: KnowledgeBase, project_root: Path) -> int:
  added_total = 0
  candidates: list[Path] = []

  readme = project_root / "README.md"
  if readme.is_file():
    candidates.append(readme)

  for rel_dir in ("docs", "datasets"):
    d = project_root / rel_dir
    if not d.is_dir():
      continue
    for p in d.rglob("*"):
      if p.is_file() and p.suffix.lower() in {".md", ".txt", ".json"}:
        candidates.append(p)

  for path in candidates:
    content = read_text_file(path)
    if not content.strip():
      continue
    rel_source = str(path.relative_to(project_root)).replace("\\", "/")
    doc_type = path.suffix.lower().lstrip(".") or "text"

    added = 0
    for idx, chunk in enumerate(chunk_text(content), start=1):
      digest = hashlib.sha1(f"{rel_source}|{idx}|{chunk}".encode("utf-8")).hexdigest()[:16]
      doc_id = f"local_{doc_type}_{digest}_{idx:03d}"
      metadata = {
        "source": rel_source,
        "type": f"local_{doc_type}",
        "chunk_index": idx,
      }
      if kb.add(doc_id=doc_id, text=chunk, metadata=metadata, force=False):
        added += 1

    added_total += added
    print(f"[local] {rel_source}: +{added}")

  return added_total


def main() -> None:
  project_root = Path(__file__).resolve().parent
  kb = KnowledgeBase(collection_name=KB_COLLECTION)

  added_seed = add_seed_docs(kb)
  added_blog = ingest_blog_articles(kb)
  added_local = ingest_local_files(kb, project_root)

  print(
    json.dumps(
      {
        "collection": KB_COLLECTION,
        "added_seed": added_seed,
        "added_blog": added_blog,
        "added_local": added_local,
        "added_total": added_seed + added_blog + added_local,
      },
      ensure_ascii=False,
    )
  )


if __name__ == "__main__":
  main()
