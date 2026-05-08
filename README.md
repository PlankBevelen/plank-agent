# Plank Agent

基于 ReAct 架构的个人 AI Agent 系统，具备对话推理、向量知识库检索、长短期记忆管理、工具调用、Web 服务和离线评测能力。

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 架构设计](#2-架构设计)
- [3. 项目结构](#3-项目结构)
- [4. 核心模块详解](#4-核心模块详解)
  - [4.1 Agent — 主控制器](#41-agent--主控制器)
  - [4.2 LLM — 大模型封装](#42-llm--大模型封装)
  - [4.3 KnowledgeBase — 向量知识库](#43-knowledgebase--向量知识库)
  - [4.4 MemoryManager — 记忆管理](#44-memorymanager--记忆管理)
  - [4.5 ContextBuilder — 上下文构建](#45-contextbuilder--上下文构建)
  - [4.6 PromptLoader — Prompt 模板引擎](#46-promptloader--prompt-模板引擎)
  - [4.7 Search — 搜索工具](#47-search--搜索工具)
  - [4.8 Tool / ToolExecutor / ToolChain — 工具系统](#48-tool--toolexecutor--toolchain--工具系统)
  - [4.9 AgentEvaluator — 评测系统](#49-agentevaluator--评测系统)
  - [4.10 init_kb — 知识库初始化](#410-init_kb--知识库初始化)
  - [4.11 Constant — 配置管理](#411-constant--配置管理)
- [5. Flask Web 服务](#5-flask-web-服务)
  - [5.1 REST API](#51-rest-api)
  - [5.2 限流器](#52-限流器)
  - [5.3 会话管理](#53-会话管理)
- [6. Prompt 模板系统](#6-prompt-模板系统)
- [7. 测试](#7-测试)
- [8. 环境要求与安装](#8-环境要求与安装)
- [9. 配置参考](#9-配置参考)
- [10. 使用指南](#10-使用指南)
  - [10.1 命令行交互](#101-命令行交互)
  - [10.2 Flask Web 服务](#102-flask-web-服务)
  - [10.3 知识库初始化](#103-知识库初始化)
  - [10.4 运行评测](#104-运行评测)
  - [10.5 PM2 部署](#105-pm2-部署)
- [11. 数据流全览](#11-数据流全览)
- [12. 技术栈](#12-技术栈)

---

## 1. 项目概述

Plank Agent 是一个全栈 AI Agent 项目，围绕 **ReAct（Reasoning + Acting）** 范式设计。核心流程为：

1. 用户输入问题
2. 系统检索知识库（ChromaDB 向量检索）和记忆（历史对话摘要）
3. 构建完整上下文（对话历史 + 知识库结果 + 记忆 + 工具观察）
4. 决策阶段：判断是否需要调用外部工具（如搜索）
5. 若需要工具，执行并观察结果，可多步循环
6. 最终生成自然语言回答并持久化记忆

支持两种交互方式：
- **CLI 模式**：`python Agent.py` 命令行交互
- **Web 服务模式**：Flask REST API，支持非流式和 SSE 流式响应

---

## 2. 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent (主控)                         │
│  ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────┐ │
│  │  Prompt  │  │  Context  │  │   Memory   │  │   Tool   │ │
│  │  Loader  │  │  Builder  │  │  Manager   │  │ Executor │ │
│  └──────────┘  └───────────┘  └────────────┘  └──────────┘ │
│       │              │               │               │       │
│       ▼              ▼               ▼               ▼       │
│  ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────┐ │
│  │ prompts/ │  │  LLM.py   │  │KnowledgeBase│  │  Search  │ │
│  │ *.txt    │  │ (OpenAI   │  │ (ChromaDB  │  │ (SerpAPI)│ │
│  │          │  │  compat)  │  │  + SBERT)  │  │          │ │
│  └──────────┘  └───────────┘  └────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │      Flask Web 服务        │
              │  /health  │  /chat (SSE)  │
              │  Rate Limiter  │  Session │
              └───────────────────────────┘
```

**关键设计决策：**

- **单例共享**：KnowledgeBase 的 embedder、ChromaDB client 和 collection 在进程内共享，避免重复加载模型和连接
- **线程安全**：使用 `threading.Lock` 保护共享状态（KB、Memory、Session、Rate Limiter）
- **双后端支持**：Session 存储和 Rate Limiter 均支持内存（开发）和 Redis（生产）两种后端
- **嵌入缓存**：查询向量嵌入带 TTL 的 LRU 缓存，减少重复编码

---

## 3. 项目结构

```text
plank-agent/
├── Agent.py                     # 主 Agent，ReAct 循环，交互式 CLI
├── AgentEvaluator.py            # 离线评测脚本（Exact Match、分 level 指标）
├── KnowledgeBase.py             # ChromaDB 向量知识库封装
├── MemoryManager.py             # 记忆读写、重要性评分、排序检索
├── ContextBuilder.py            # 上下文拼装与自适应裁剪
├── LLM.py                       # LLM 调用（OpenAI SDK 兼容接口，支持流式）
├── Search.py                    # SerpAPI 搜索工具
├── PromptLoader.py              # Prompt 模板加载与占位符验证
├── Tool.py                      # 工具抽象（名称、描述、函数、参数 schema）
├── ToolExecutor.py              # 工具注册与调度
├── ToolChain.py                 # 工具链（多工具串联）
├── Constant.py                  # 环境变量读取与默认配置
├── init_kb.py                   # 知识库初始化（种子文档 + 博客抓取 + 本地文件）
│
├── flask/
│   ├── app.py                   # Flask Web API（/health, /chat, SSE stream）
│   ├── rate_limiter.py          # 固定窗口限流器（内存 + Redis）
│   ├── session_agent_store.py   # 会话 Agent 存储（会话隔离、TTL、分布式锁）
│   └── session_memory.py        # 单轮对话记忆存储
│
├── prompts/
│   ├── system.txt               # 系统人设 Prompt
│   ├── decision.txt             # ReAct 决策 Prompt
│   ├── answer.txt               # 无搜索最终回答 Prompt
│   └── answer_with_search.txt   # 带搜索结果最终回答 Prompt
│
├── tests/
│   ├── test_agent_core.py       # Agent 核心逻辑单元测试
│   ├── test_context_builder.py  # ContextBuilder 单元测试
│   ├── test_memory_manager.py   # MemoryManager 单元测试
│   └── test_prompt_loader.py    # PromptLoader 单元测试
│
├── datasets/
│   └── eval_samples.json        # 评测样本数据集
│
├── reports/                     # 评测报告输出目录
├── chroma_db/                   # 本地向量数据库（ChromaDB 持久化）
├── models/                      # 本地 Embedding 模型
│   └── paraphrase-multilingual-MiniLM-L12-v2/
│
├── .env.example                 # 环境变量配置模板
├── .gitignore                   # Git 忽略规则
├── ecosystem.config.cjs         # PM2 进程管理配置
├── requirements.txt             # Python 依赖
└── README.md                    # 本文件
```

---

## 4. 核心模块详解

### 4.1 Agent — 主控制器

**文件**: `Agent.py`

Agent 是整个系统的入口和调度中心。核心流程：

```
用户输入
  │
  ├─► 检索知识库（KB search）+ 记忆检索（Memory retrieve）
  ├─► 构建上下文（ContextBuilder.build）
  │
  ├─► ReAct 循环（最多 PLANK_REACT_MAX_STEPS 步）:
  │    ├─ 生成决策 Prompt → LLM 推理
  │    ├─ 解析 Action（正则匹配 `Action: ToolName[query]`）
  │    ├─ 执行工具 → 收集 Observation
  │    └─ 若无 Action 则退出循环
  │
  ├─► 选择最终 Prompt（answer / answer_with_search）
  ├─► LLM 生成最终回答
  └─► 保存交互记录到 Memory
```

**关键数据结构：**

- `ReactLoopResult`：循环结果（observations, step_traces, planner_note, used_tools）
- `TurnPreparation`：单轮准备（loop_result, context_pack, prompt_name, prompt_text）
- `Agent.export_session_state()` / `restore_session_state()`：会话状态导出/恢复，用于 Web 服务跨请求对话连续性

**重要方法：**

| 方法 | 说明 |
|------|------|
| `run(user_input, return_trace, silent, include_memory, persist_memory)` | 完整执行一轮对话，可返回详细 trace |
| `run_stream(user_input, ...)` | 流式生成器，逐 token yield |
| `_prepare_turn()` | 执行 ReAct 循环 + 构建上下文 + 选择 prompt |
| `_finalize_turn()` | 存储消息到历史、持久化记忆、截断历史（最多 20 条） |
| `prewarm()` | 预热：提前加载模型和 KB，减少首次请求延迟 |

---

### 4.2 LLM — 大模型封装

**文件**: `LLM.py`

基于 OpenAI SDK 的 LLM 调用封装。

**特性：**
- 兼容所有 OpenAI 协议接口（通过 `PLANK_LLM_API_BASE_URL` 配置）
- 支持流式输出：`stream_think()` 返回 `Iterator[str]`
- 非流式模式：`think()` 内部调用 `stream_think()` 并拼接结果
- 默认模型：`deepseek-v3-2-251201`（可通过环境变量覆盖）
- API Key 优先级：`PLANK_LLM_API_KEY` > `OPENAI_API_KEY`

**使用示例：**
```python
llm = LLM()
response = llm.think(messages, temperature=0.7, max_new_tokens=512)

# 流式
for chunk in llm.stream_think(messages):
    print(chunk, end="")
```

---

### 4.3 KnowledgeBase — 向量知识库

**文件**: `KnowledgeBase.py`

基于 ChromaDB + SentenceTransformer 的向量知识库。

**Embedding 模型：** `paraphrase-multilingual-MiniLM-L12-v2`（sentence-transformers）
- 多语言支持（中英文均可）
- 本地存储于 `models/` 目录，自动下载 tokenizer
- 支持 CUDA / CPU 自动检测

**核心功能：**

| 方法 | 说明 |
|------|------|
| `add(doc_id, text, metadata, force)` | 添加/更新文档（upsert），自动去重 |
| `search(query, top_k, threshold)` | 语义检索，返回文本列表 |
| `search_with_meta(query, top_k, threshold, where)` | 带元数据的检索，支持 ChromaDB where 过滤 |

**嵌入缓存：**

- LRU 缓存查询向量（默认最大 1024 条，TTL 3600 秒）
- 避免重复编码相同查询

**共享模型：**

- 进程内全局共享 `SentenceTransformer` 实例
- 首次加载后，所有 `KnowledgeBase` 实例复用同一个模型

---

### 4.4 MemoryManager — 记忆管理

**文件**: `MemoryManager.py`

实现 Agent 的长期记忆系统，存储历史对话交互。

**记忆模型：**

每条记忆记录包含：
- `user_id`：用户标识
- `text`：结构化文本（时间 + 用户输入 + 助手输出 + 工具观察）
- `type`：固定为 `episodic`
- `importance`：重要性评分（0.0 ~ 1.0）
- `tags`：标签（如 `["chat", "react"]`）

**重要性评分算法（`estimate_importance`）：**

| 条件 | 加分 |
|------|------|
| 基础分 | 0.4 |
| 用户输入长度 > 80 | +0.1 |
| 有工具观察 | +0.1 |
| 含关键词（preference/habit/long-term/plan/goal/todo） | +0.25 |
| 含关键词（name/my project/my team/my company） | +0.2 |

**检索排序（`retrieve`）：**

最终排序分数 = `0.7 × 向量相似度 + 0.25 × 重要性 + 0.05 × 新鲜度奖励`

其中新鲜度奖励基于创建时间，越新的记忆分数越高（最大 +0.05）。

**安全机制：**
- `write_enabled` 开关：可关闭记忆写入（评测时避免污染）
- 硬阈值过滤：`PLANK_MEMORY_THRESHOLD`（默认 0.65）
- 上下文格式化：限制最大条目数和字符数

---

### 4.5 ContextBuilder — 上下文构建

**文件**: `ContextBuilder.py`

负责将各数据源组装为 LLM 可见的最终上下文。

**数据源（ContextPack）：**

```
[Current User Query]     ← 用户当前问题
[Recent Conversation]    ← 最近 N 轮对话历史
[Relevant Memory]        ← 记忆检索结果
[Knowledge Base]         ← 知识库检索结果
[Tool Observations]      ← ReAct 循环中的工具观察
```

**自适应裁剪：**

当上下文超出 `max_chars`（默认 5000）时，按预算比例裁剪：
- 查询：固定 800 字符
- 历史：35%
- 记忆：25%
- 知识库：25%
- 工具观察：15%

**可配置参数：**
- `PLANK_CONTEXT_MAX_CHARS`：最大总字符数
- `PLANK_CONTEXT_MAX_HISTORY_TURNS`：最大历史轮数
- `PLANK_CONTEXT_MAX_KB_ITEMS`：最大知识库条目
- `PLANK_CONTEXT_MAX_MEMORY_ITEMS`：最大记忆条目

---

### 4.6 PromptLoader — Prompt 模板引擎

**文件**: `PromptLoader.py`

模板化 Prompt 系统，将 prompt 逻辑与代码分离。

**模板语法：**
- `{variable}` — 占位符，运行时替换
- `{{ literal }}` — 字面量花括号（输出为 `{ literal }`）

**Prompt 契约验证：**

`PromptLoader.validate()` 检查每个模板文件的占位符是否与预期一致：

```python
REQUIRED_PROMPTS = {
    "system": set(),
    "decision": {"tools", "context", "observations"},
    "answer": {"context", "kb_context", "planner_note"},
    "answer_with_search": {"context", "kb_context", "search_result", "planner_note"},
}
```

不匹配时抛出 `ValueError`，防止运行时因缺变量导致异常。

**缓存：**
- 文件内容读取后缓存
- 验证结果按目录+清单缓存，避免重复验证

---

### 4.7 Search — 搜索工具

**文件**: `Search.py`

基于 [SerpAPI](https://serpapi.com/) 的 Google 搜索工具。

**结果提取优先级：**
1. `answer_box_list` → 结构化答案列表
2. `answer_box.answer` → 知识图谱直接回答
3. `knowledge_graph.description` → 知识图谱描述
4. `organic_results` → 前 3 条自然搜索结果的标题+摘要

**参数：**
- 搜索引擎：Google
- 地区：`cn`，语言：`zh-CN`
- 超时：10 秒
- 需要配置 `SERPAPI_KEY`，未配置时返回错误提示

---

### 4.8 Tool / ToolExecutor / ToolChain — 工具系统

**文件**: `Tool.py`, `ToolExecutor.py`, `ToolChain.py`

**Tool** — 单个工具定义：

```python
@dataclass
class Tool:
    name: str           # 工具名（如 "Search"）
    description: str    # 描述（给 LLM 看）
    func: Callable      # 实际执行函数
    params_schema: dict # 参数 schema
```

**ToolExecutor** — 工具注册与调度：

- `register(item)`：注册 Tool 或 ToolChain
- `run(name, query)`：按名称执行工具
- `describe_all()`：生成工具列表描述，注入 decision prompt

**ToolChain** — 工具链：

将多个 Tool 串联执行，上一步输出作为下一步输入：

```python
chain = ToolChain("analyze", "Search then summarize")
chain.add_step(search_tool)
chain.add_step(summarize_tool, mapper=lambda prev: {"query": f"总结: {prev}"})
```

---

### 4.9 AgentEvaluator — 评测系统

**文件**: `AgentEvaluator.py`

离线评测框架，评估 Agent 在 QA 数据集上的表现。

**评测指标：**

| 指标 | 说明 |
|------|------|
| Exact Match Rate | 标准化后的精确匹配率 |
| Avg Steps | 平均 ReAct 推理步数 |
| Avg Latency (ms) | 平均响应延迟 |
| Avg Tool Calls | 平均工具调用次数 |
| Level Metrics | 按难度分层的指标 |
| Degradation | 跨难度级别的性能退化分析 |

**标准化规则（`normalize_text`）：**
- 小写化
- 去除标点
- 压缩空格

**数据集格式（JSON）：**

```json
[
  {
    "id": "1",
    "question": "What is X?",
    "answer": "Expected answer",
    "level": 1,
    "tags": ["topic"]
  }
]
```

**运行：**

```bash
python AgentEvaluator.py --dataset datasets/eval_samples.json
```

**输出：**
- JSON 格式详细报告（每条样本的预测、trace、耗时）
- Markdown 格式摘要

---

### 4.10 init_kb — 知识库初始化

**文件**: `init_kb.py`

初始化并填充向量知识库。三类数据源：

**1. 种子文档（`add_seed_docs`）**

硬编码的基础知识，包括：
- `site_plankbevelen_cn`：博客站点信息
- `site_tool_plankbevelen_cn`：工具站信息
- `author_intro`：作者身份与技能
- `author_identity_alias`：名称别名与检索别称
- `author_tech_stack`：技术栈画像
- `author_projects_overview`：代表项目
- `author_response_style`：回答风格约定

**2. 博客文章抓取（`ingest_blog_articles`）**

从 `https://plankbevelen.cn/api/article` 分页抓取所有文章，提取标题、摘要、内容、分类、标签。

**3. 本地文件导入（`ingest_local_files`）**

扫描 `docs/`、`datasets/` 目录下的 `.md`、`.txt`、`.json` 文件，分块存入知识库（900 字符/块，120 字符重叠）。

**特性：**
- 自动去重（force=False）
- 多编码支持（UTF-8 / UTF-8-SIG / GBK）
- 基于 SHA1 的文档 ID 生成

---

### 4.11 Constant — 配置管理

**文件**: `Constant.py`

所有配置通过环境变量读取，提供合理默认值。

**辅助函数：**

| 函数 | 说明 |
|------|------|
| `get_env(name, default)` | 读取字符串环境变量 |
| `get_env_bool(name, default)` | 读取布尔环境变量（true/1/yes/on） |
| `get_first_env(names, default)` | 读取第一个存在的环境变量（支持多 key 回退） |

**配置优先级示例：**
- `PLANK_LLM_API_KEY` > `OPENAI_API_KEY`
- `PLANK_LLM_API_BASE_URL` > `ARK_BASE_URL` > `OPENAI_BASE_URL` > 默认值

---

## 5. Flask Web 服务

### 5.1 REST API

**文件**: `flask/app.py`

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查，返回 `{"status": "ok"}` |
| `/chat` | POST | 对话接口，支持流式（SSE）和非流式 |

**请求格式（POST /chat）：**

```json
{
  "message": "你好，请介绍一下你自己",
  "stream": false,
  "include_memory": true,
  "persist_memory": true
}
```

**请求头：**
- `X-Session-Id`（必需）：会话标识，用于多轮对话连续性

**非流式响应：**

```json
{
  "session_id": "abc123",
  "answer": "我是 PlankBevelen...",
  "elapsed_ms": 1523
}
```

**SSE 流式响应：**

```
event: delta
data: {"text": "我"}

event: delta
data: {"text": "是"}

event: done
data: {"session_id": "abc123", "elapsed_ms": 1523, "first_token_ms": 320}
```

**启动：**

```bash
python flask/app.py
```

默认监听 `0.0.0.0:6543`。

**会话管理：**
- 每个 `X-Session-Id` 对应独立的 Agent 实例和对话历史
- 会话 TTL 默认 1800 秒（30 分钟）
- 支持内存和 Redis 两种后端

**重启预热：**
- 首次请求时自动预热 Agent（加载模型、初始化 KB）
- 可通过 `PLANK_AGENT_PREWARM=false` 关闭

---

### 5.2 限流器

**文件**: `flask/rate_limiter.py`

固定窗口限流器，支持内存和 Redis 两种后端。

**两层限流：**

| 限流层 | 键 | 默认限制 |
|--------|-----|----------|
| 会话级 | `sid:{session_id}` | 20 次/窗口 |
| IP 级 | `ip:{client_ip}` | 100 次/窗口 |

**配置：**
- `PLANK_RATE_LIMIT_WINDOW_SECONDS`：窗口大小（默认 60 秒）
- `PLANK_SESSION_RATE_LIMIT_MAX_REQUESTS`：会话级限制
- `PLANK_IP_RATE_LIMIT_MAX_REQUESTS`：IP 级限制

**超限响应：**

```json
{
  "error": "rate_limited",
  "message": "Too many requests for this session",
  "retry_after": 42
}
```

HTTP 状态码 429，带 `Retry-After` 头。

---

### 5.3 会话管理

**文件**: `flask/session_agent_store.py`, `flask/session_memory.py`

**SessionAgentStore：**
- 管理用户会话 → Agent 状态的映射
- 支持 `session_lock` 上下文管理器（`threading.Lock` 或 Redis 分布式锁）
- 支持 `load_messages` / `save_messages` 持久化对话历史
- TTL 过期自动清理

**后端对比：**

| 特性 | Memory | Redis |
|------|--------|-------|
| 持久化 | 否 | 是 |
| 多进程共享 | 否 | 是 |
| 分布式锁 | threading.Lock | Redis SET NX |
| 适用场景 | 开发/单进程 | 生产/多进程 |

**SingleTurnMemoryStore：**
- 只保留每个会话的最新一轮对话
- 与 SessionAgentStore 独立，用于轻量场景

---

## 6. Prompt 模板系统

**目录**: `prompts/`

四个 Prompt 模板对应 Agent 推理的不同阶段：

| 模板 | 触发时机 | 占位符 |
|------|----------|--------|
| `system.txt` | 初始化时加载为 system message | 无 |
| `decision.txt` | ReAct 每步决策 | `{tools}`, `{context}`, `{observations}` |
| `answer.txt` | 无需搜索时的最终回答 | `{context}`, `{kb_context}`, `{planner_note}` |
| `answer_with_search.txt` | 有搜索结果时的最终回答 | `{context}`, `{kb_context}`, `{search_result}`, `{planner_note}` |

**System Prompt 设计要点：**
- 人设：PlankBevelen 的个人 AI 分身
- 语言：优先中文，自然直接
- 行为：先理解再回答，区分事实与推断
- 禁止：暴露系统内部状态、模板腔、编造事实

**Decision Prompt 设计要点：**
- 严格输出格式：`Action: ToolName[query]` 或直接回答草稿
- 单步单动作
- 寒暄/已知信息直接回答，时效性/不确定内容调用搜索

---

## 7. 测试

**目录**: `tests/`

| 文件 | 测试内容 |
|------|----------|
| `test_agent_core.py` | `_parse_action` 解析、`_retrieve_kb_results` 回退、`_prepare_turn` 上下文传递 |
| `test_context_builder.py` | 上下文组装包含所有段、超出预算时的裁剪行为 |
| `test_memory_manager.py` | 检索排序（综合分数 = 相似度 + 重要性 + 新鲜度） |
| `test_prompt_loader.py` | 占位符替换、字面量花括号保留、契约验证异常 |

**运行测试：**

```bash
# 全部测试
python -m pytest tests/ -v

# 单个测试文件
python -m pytest tests/test_agent_core.py -v
```

---

## 8. 环境要求与安装

**系统要求：**
- Python 3.10+
- 建议 4GB+ 内存（加载 embedding 模型需要）
- 可选：CUDA GPU（加速 embedding）

**安装步骤：**

```bash
# 1. 克隆项目
git clone <repo-url>
cd plank-agent

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 PLANK_LLM_API_KEY 等必要配置
```

**依赖列表（`requirements.txt`）：**

| 包 | 版本 | 用途 |
|----|------|------|
| chromadb | 1.5.8 | 向量数据库 |
| sentence_transformers | 5.4.1 | 文本嵌入 |
| torch | 2.8.0 | 深度学习框架 |
| transformers | 5.5.4 | Hugging Face 模型库 |
| openai | >=1.30.0 | LLM API 调用 |
| Flask | >=3.0.0 | Web 服务框架 |
| serpapi | 1.0.2 | Google 搜索 API |
| python-dotenv | >=1.0.1 | .env 配置加载 |
| huggingface_hub | >=0.24.0 | 模型下载 |
| redis | >=5.0.0 | Redis 后端支持（可选） |
| Requests | 2.33.1 | HTTP 请求 |

---

## 9. 配置参考

**LLM 配置：**

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `PLANK_LLM_API_KEY` | - | LLM API 密钥（优先于 OPENAI_API_KEY） |
| `OPENAI_API_KEY` | - | OpenAI API 密钥（回退） |
| `PLANK_LLM_API_BASE_URL` | `https://ark.cn-beijing.volces.com/api/v3` | API 基础 URL |
| `PLANK_LLM_MODEL` | `deepseek-v3-2-251201` | 模型名称 |

**Embedding 配置：**

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `PLANK_EMBEDDING_MODEL_PATH` | `./models/paraphrase-multilingual-MiniLM-L12-v2` | 本地模型路径 |
| `PLANK_EMBEDDING_DEVICE` | `cuda`（可用时）否则 `cpu` | 推理设备 |

**ReAct 与上下文：**

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `PLANK_REACT_MAX_STEPS` | 5 | 最大 ReAct 循环步数 |
| `PLANK_CONTEXT_MAX_CHARS` | 5000 | 上下文最大字符数 |
| `PLANK_CONTEXT_MAX_HISTORY_TURNS` | 4 | 最大历史轮数 |
| `PLANK_CONTEXT_MAX_KB_ITEMS` | 3 | 最大知识库条目数 |
| `PLANK_CONTEXT_MAX_MEMORY_ITEMS` | 4 | 最大记忆条目数 |

**记忆配置：**

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `PLANK_MEMORY_WRITE_ENABLED` | true | 是否写入记忆 |
| `PLANK_MEMORY_DB_PATH` | `./chroma_db` | 记忆数据库路径 |
| `PLANK_MEMORY_COLLECTION` | `agent_memory` | 记忆集合名 |
| `PLANK_MEMORY_TOP_K` | 6 | 检索返回数 |
| `PLANK_MEMORY_THRESHOLD` | 0.65 | 语义检索阈值 |

**知识库缓存：**

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `PLANK_KB_QUERY_CACHE_MAX_SIZE` | 1024 | 查询嵌入缓存大小 |
| `PLANK_KB_QUERY_CACHE_TTL_SECONDS` | 3600 | 缓存 TTL（秒） |

**Web 服务配置：**

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `PLANK_AGENT_PREWARM` | true | 启动时预热 Agent |
| `PLANK_WEB_INCLUDE_MEMORY` | true | Web 模式默认包含记忆 |
| `PLANK_WEB_PERSIST_MEMORY` | true | Web 模式默认持久化记忆 |
| `PLANK_SESSION_BACKEND` | memory | 会话后端（memory/redis） |
| `PLANK_SESSION_TTL_SECONDS` | 1800 | 会话过期时间 |
| `PLANK_SESSION_LOCK_TIMEOUT_SECONDS` | 30 | 锁超时 |
| `PLANK_RATE_LIMIT_BACKEND` | memory | 限流后端（memory/redis） |
| `PLANK_RATE_LIMIT_WINDOW_SECONDS` | 60 | 限流窗口 |
| `PLANK_SESSION_RATE_LIMIT_MAX_REQUESTS` | 20 | 会话级限流 |
| `PLANK_IP_RATE_LIMIT_MAX_REQUESTS` | 100 | IP 级限流 |
| `PLANK_REDIS_URL` | `redis://localhost:6379/0` | Redis 连接 URL |

**其他：**

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `SERPAPI_KEY` | - | SerpAPI 密钥（搜索工具需要） |

---

## 10. 使用指南

### 10.1 命令行交互

```bash
python Agent.py
```

交互示例：
```
Agent started. Type your question, or 'exit' to quit.
Question (or 'exit'): 介绍一下你自己
Step 1 decision: 我是 PlankBevelen，一个全栈开发者...
Answer: 我是 PlankBevelen，专注于全栈开发和 AI Agent 工程...
```

### 10.2 Flask Web 服务

**启动服务：**

```bash
python flask/app.py
```

服务运行在 `http://0.0.0.0:6543`。

**非流式请求示例：**

```bash
curl -X POST http://localhost:6543/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: my-session-1" \
  -d '{"message": "你好，介绍一下你自己"}'
```

**流式请求示例：**

```bash
curl -X POST http://localhost:6543/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: my-session-1" \
  -d '{"message": "解释一下 ReAct 架构", "stream": true}'
```

**健康检查：**

```bash
curl http://localhost:6543/health
# {"status":"ok"}
```

**多轮对话：**

使用相同的 `X-Session-Id` 可以保持对话连续性：

```bash
# 第一轮
curl -X POST http://localhost:6543/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: session-001" \
  -d '{"message": "我叫张三"}'

# 第二轮：Agent 会记得你的名字
curl -X POST http://localhost:6543/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: session-001" \
  -d '{"message": "我叫什么？"}'
```

### 10.3 知识库初始化

```bash
python init_kb.py
```

执行内容：
1. 写入 7 条种子文档（作者信息、站点信息、风格约定）
2. 从 plankbevelen.cn API 抓取博客文章
3. 扫描 `docs/`、`datasets/` 目录下的本地文件

输出示例：
```json
{"collection": "plankbevelen", "added_seed": 7, "added_blog": 15, "added_local": 3, "added_total": 25}
```

### 10.4 运行评测

```bash
python AgentEvaluator.py --dataset datasets/eval_samples.json
```

完整参数：

```bash
python AgentEvaluator.py \
  --dataset datasets/eval_samples.json \
  --out-json reports/eval_report.json \
  --out-md reports/eval_report.md \
  --agent-name PlankAgent \
  --user-id eval_user \
  --include-memory
```

输出示例：
```
Evaluation complete.
Total: 3
Exact Match: 2
Exact Match Rate: 0.6667
Avg Steps: 1.33
Avg Latency(ms): 2340.50
Avg Tool Calls: 0.33
```

### 10.5 PM2 部署

项目包含 PM2 配置文件 `ecosystem.config.cjs`：

```bash
# 安装 PM2
npm install -g pm2

# 启动
pm2 start ecosystem.config.cjs

# 查看状态
pm2 status

# 查看日志
pm2 logs plank-agent
```

---

## 11. 数据流全览

```
┌──────────────────────────────────────────────────────────────────┐
│                           一次完整对话                             │
│                                                                  │
│  ① 用户输入 "今天天气怎么样？"                                     │
│       │                                                          │
│       ▼                                                          │
│  ② Agent._retrieve_kb_results()                                  │
│     └─► KnowledgeBase.search_with_meta("今天天气怎么样？")         │
│         └─► SentenceTransformer.encode() → ChromaDB.query()      │
│       │                                                          │
│       ▼                                                          │
│  ③ Agent._build_memory_context()                                 │
│     └─► MemoryManager.retrieve() → ranked by importance+recency  │
│       │                                                          │
│       ▼                                                          │
│  ④ ContextBuilder.build()                                        │
│     └─► 组装 5 段上下文 → ContextPack                             │
│       │                                                          │
│       ▼                                                          │
│  ⑤ ReAct Loop (step 1):                                          │
│     ├─► decision.txt + {tools} + {context} + {observations}      │
│     ├─► LLM.think() → "Action: Search[今天天气]"                  │
│     ├─► Agent._parse_action() → ("Search", "今天天气")            │
│     └─► ToolExecutor.run("Search", "今天天气")                    │
│         └─► Search.search() → SerpAPI → "今天北京晴 25°C"        │
│       │                                                          │
│       ▼                                                          │
│  ⑥ ReAct Loop (step 2):                                          │
│     ├─► decision.txt + updated {observations}                    │
│     ├─► LLM.think() → "今天北京晴天，气温25°C，适合出行。"         │
│     └─► _parse_action() → None → 退出循环                        │
│       │                                                          │
│       ▼                                                          │
│  ⑦ Agent._prepare_turn() → answer_with_search.txt               │
│     └─► {context} + {kb_context} + {search_result} + {planner}   │
│       │                                                          │
│       ▼                                                          │
│  ⑧ LLM.think() → 最终回答                                        │
│     └─► "今天北京是晴天，气温约25°C..."                           │
│       │                                                          │
│       ▼                                                          │
│  ⑨ Agent._finalize_turn()                                        │
│     ├─► messages.append(user + assistant)                        │
│     ├─► 截断至 20 条                                              │
│     └─► MemoryManager.save_interaction() → ChromaDB.upsert()     │
│                                                                  │
│  ⑩ 返回最终回答给用户                                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 12. 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| LLM 推理 | OpenAI SDK (兼容) | 支持 DeepSeek、OpenAI 等任何兼容接口 |
| 向量数据库 | ChromaDB | 本地持久化，无需外部服务 |
| 文本嵌入 | SentenceTransformer + MiniLM | 多语言支持，本地运行 |
| 搜索工具 | SerpAPI (Google) | 实时信息检索 |
| Web 框架 | Flask | REST API + SSE 流式 |
| 限流 | 固定窗口算法 | 内存/Redis 双后端 |
| 会话存储 | TTL + 锁 | 内存/Redis 双后端 |
| 进程管理 | PM2 | 生产环境守护进程 |
| 深度学习 | PyTorch + Transformers | 模型加载与推理 |
| 评测 | Exact Match + 分层分析 | 离线回归测试 |
| 配置 | python-dotenv | .env 文件管理 |
