# Plank Agent

一个基于 ReAct 思路实现的个人 Agent 项目，支持：

- 对话推理（LLM）
- 知识库检索（ChromaDB + SentenceTransformer）
- 长短期记忆管理（向量记忆）
- 搜索工具调用（SerpAPI）
- 离线评测（Exact Match、延迟、步骤统计）

## 1. 项目结构

```text
plank-agent/
├─ Agent.py                 # 主 Agent 入口（交互式对话）
├─ AgentEvaluator.py        # 评测脚本
├─ KnowledgeBase.py         # 向量知识库封装
├─ MemoryManager.py         # 记忆读写与排序
├─ ContextBuilder.py        # 上下文拼装与裁剪
├─ LLM.py                   # LLM 调用封装（OpenAI SDK 兼容接口）
├─ Search.py                # 搜索工具（SerpAPI）
├─ PromptLoader.py          # Prompt 模板加载器
├─ Constant.py              # 环境变量与默认配置
├─ init_kb.py               # 初始化知识库脚本
├─ prompts/                 # system / decision / answer 模板
├─ datasets/                # 评测数据集
├─ reports/                 # 评测输出
├─ chroma_db/               # 本地向量数据库
└─ models/                  # 本地 embedding 模型
```

## 2. 环境要求

- Python 3.10+
- 建议使用虚拟环境（venv 或 conda）

安装依赖：

```bash
pip install -r requirements.txt
```

## 3. 环境变量配置

项目通过 `.env` 读取配置。至少需要：

- `PLANK_LLM_API_KEY` 或 `OPENAI_API_KEY`
- `PLANK_LLM_API_BASE_URL`（如果使用兼容 OpenAI 协议的三方服务）
- `PLANK_LLM_MODEL`

搜索工具可选：

- `SERPAPI_KEY`（未配置时 Search 工具会返回错误提示，不影响纯对话）

常用可调参数：

- `PLANK_REACT_MAX_STEPS`（默认 5）
- `PLANK_CONTEXT_MAX_CHARS`（默认 5000）
- `PLANK_CONTEXT_MAX_HISTORY_TURNS`（默认 4）
- `PLANK_MEMORY_WRITE_ENABLED`（默认 true）
- `PLANK_MEMORY_DB_PATH`（默认 `./chroma_db`）

## 4. 启动 Agent

```bash
python Agent.py
```

启动后输入问题进行交互，输入 `exit` 退出。

## 5. 初始化知识库（可选）

如需导入预置知识：

```bash
python init_kb.py
```

说明：该脚本会写入 ChromaDB 集合（默认 `plankbevelen`），并尝试抓取博客内容。

## 6. 运行评测

```bash
python AgentEvaluator.py --dataset datasets/your_eval.json
```

常用参数：

- `--out-json reports/eval_report.json`
- `--out-md reports/eval_report.md`
- `--agent-name PlankAgent`
- `--user-id eval_user`
- `--include-memory`（默认关闭，便于复现实验）

评测输出包含：

- Exact Match / Exact Match Rate
- 平均推理步数（avg_steps）
- 平均延迟（avg_latency_ms）
- 平均工具调用次数（avg_tool_calls）
- 分 level 指标和退化分析（degradation）

## 7. Prompt 机制

`prompts/` 目录内模板：

- `system.txt`
- `decision.txt`
- `answer_with_search.txt`
- `answer.txt`

运行流程中，Agent 会先依据 `decision` 选择是否执行 `Action: Tool[query]`，再基于 observation 生成最终回答。

## 8. 注意事项

- 首次加载 embedding 模型会较慢。
- `init_kb.py` 中如果外部接口不可用，可能只导入静态内容。
- PowerShell 若提示 profile 脚本执行策略错误，不影响本项目 Python 运行本身。

## 9. 后续建议

- 增加 `.env.example`（避免把真实密钥提交到仓库）
- 为 `init_kb.py` 增加编码与异常处理清理
- 补充单元测试（`KnowledgeBase` / `MemoryManager` / `ContextBuilder`）
