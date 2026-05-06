# Agent 性能评估

## 1. 数据集格式
评测数据是 JSON 数组，每条样本包含：
- `id`: 样本 ID（可选）
- `question`: 问题
- `answer`: 标准答案
- `level`: 难度等级（默认 1）
- `tags`: 标签（可选）

示例见：
- `datasets/eval_samples.json`

## 2. 运行评测
```bash
python AgentEvaluator.py --dataset datasets/eval_samples.json
```

可选参数：
- `--out-json reports/eval_report.json`
- `--out-md reports/eval_report.md`
- `--agent-name PlankAgent`
- `--user-id eval_user`
- `--include-memory`（启用记忆检索参与评测）

## 3. 指标说明
- `exact_match_rate`：准精确匹配率（归一化后字符串相等）
- `avg_steps`：平均推理步数
- `avg_latency_ms`：平均延迟（毫秒）
- `avg_tool_calls`：平均工具调用次数
- `level_metrics`：分难度等级统计
- `degradation`：难度递进下准确率下降

## 4. 结果输出
- JSON 全量报告：`reports/eval_report.json`
- Markdown 摘要：`reports/eval_report.md`
