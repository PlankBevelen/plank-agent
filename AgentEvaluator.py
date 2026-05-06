import argparse
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from Agent import Agent


def normalize_text(text: str) -> str:
  if text is None:
    return ""
  text = text.lower().strip()
  text = text.translate(str.maketrans("", "", string.punctuation))
  text = re.sub(r"\s+", " ", text)
  return text


def quasi_exact_match(pred: str, gold: str) -> bool:
  return normalize_text(pred) == normalize_text(gold)


@dataclass
class EvalSample:
  id: str
  question: str
  answer: str
  level: int = 1
  tags: list[str] | None = None


class AgentEvaluator:
  def __init__(self, agent: Agent):
    self.agent = agent

  def evaluate(self, samples: list[EvalSample], include_memory: bool = False) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for sample in samples:
      trace = self.agent.run(
        sample.question,
        return_trace=True,
        silent=True,
        include_memory=include_memory,
        persist_memory=False,
      )
      pred = trace["answer"]
      matched = quasi_exact_match(pred, sample.answer)
      records.append(
        {
          "id": sample.id,
          "level": sample.level,
          "question": sample.question,
          "gold_answer": sample.answer,
          "pred_answer": pred,
          "match": matched,
          "elapsed_ms": trace["elapsed_ms"],
          "steps": trace["steps"],
          "tool_calls": trace["tool_calls"],
          "trace": trace["trace"],
        }
      )

    return self._aggregate(records)

  def _aggregate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    matches = sum(1 for r in records if r["match"])
    avg_steps = mean([r["steps"] for r in records]) if records else 0.0
    avg_latency_ms = mean([r["elapsed_ms"] for r in records]) if records else 0.0
    avg_tool_calls = mean([r["tool_calls"] for r in records]) if records else 0.0

    by_level: dict[int, list[dict[str, Any]]] = {}
    for r in records:
      by_level.setdefault(int(r["level"]), []).append(r)

    level_metrics: dict[str, dict[str, Any]] = {}
    for level, items in sorted(by_level.items(), key=lambda x: x[0]):
      level_total = len(items)
      level_match = sum(1 for it in items if it["match"])
      level_metrics[str(level)] = {
        "total": level_total,
        "exact_matches": level_match,
        "exact_match_rate": (level_match / level_total) if level_total else 0.0,
        "avg_steps": mean([it["steps"] for it in items]) if items else 0.0,
        "avg_latency_ms": mean([it["elapsed_ms"] for it in items]) if items else 0.0,
      }

    level_keys = sorted(level_metrics.keys(), key=lambda x: int(x))
    degradation = []
    for i in range(1, len(level_keys)):
      prev = level_metrics[level_keys[i - 1]]["exact_match_rate"]
      curr = level_metrics[level_keys[i]]["exact_match_rate"]
      degradation.append(
        {
          "from_level": int(level_keys[i - 1]),
          "to_level": int(level_keys[i]),
          "drop": prev - curr,
        }
      )

    return {
      "summary": {
        "total_samples": total,
        "exact_matches": matches,
        "exact_match_rate": (matches / total) if total else 0.0,
        "avg_steps": avg_steps,
        "avg_latency_ms": avg_latency_ms,
        "avg_tool_calls": avg_tool_calls,
      },
      "level_metrics": level_metrics,
      "degradation": degradation,
      "records": records,
    }


def load_dataset(path: str) -> list[EvalSample]:
  raw = json.loads(Path(path).read_text(encoding="utf-8"))
  samples: list[EvalSample] = []
  for idx, item in enumerate(raw):
    sample = EvalSample(
      id=str(item.get("id", idx + 1)),
      question=str(item["question"]),
      answer=str(item["answer"]),
      level=int(item.get("level", 1)),
      tags=item.get("tags"),
    )
    samples.append(sample)
  return samples


def save_report(report: dict[str, Any], output_path: str):
  Path(output_path).parent.mkdir(parents=True, exist_ok=True)
  Path(output_path).write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )


def build_markdown_summary(report: dict[str, Any]) -> str:
  s = report["summary"]
  lines = [
    "# Agent Evaluation Report",
    "",
    f"- total_samples: {s['total_samples']}",
    f"- exact_matches: {s['exact_matches']}",
    f"- exact_match_rate: {s['exact_match_rate']:.4f}",
    f"- avg_steps: {s['avg_steps']:.2f}",
    f"- avg_latency_ms: {s['avg_latency_ms']:.2f}",
    f"- avg_tool_calls: {s['avg_tool_calls']:.2f}",
    "",
    "## Level Metrics",
  ]
  for level, item in report["level_metrics"].items():
    lines.append(
      f"- level {level}: em={item['exact_match_rate']:.4f}, total={item['total']}, avg_steps={item['avg_steps']:.2f}, avg_latency_ms={item['avg_latency_ms']:.2f}"
    )
  lines.append("")
  lines.append("## Degradation")
  if report["degradation"]:
    for d in report["degradation"]:
      lines.append(f"- L{d['from_level']} -> L{d['to_level']}: drop={d['drop']:.4f}")
  else:
    lines.append("- no multi-level data")
  return "\n".join(lines)


def main():
  parser = argparse.ArgumentParser(description="Evaluate Agent performance on a QA dataset.")
  parser.add_argument("--dataset", required=True, help="Path to evaluation dataset JSON file.")
  parser.add_argument("--out-json", default="reports/eval_report.json", help="Path to write JSON report.")
  parser.add_argument("--out-md", default="reports/eval_report.md", help="Path to write markdown summary.")
  parser.add_argument("--agent-name", default="PlankAgent", help="Agent name.")
  parser.add_argument("--user-id", default="eval_user", help="User id used in evaluation run.")
  parser.add_argument(
    "--include-memory",
    action="store_true",
    help="Include memory retrieval during evaluation. Disabled by default for reproducibility.",
  )
  args = parser.parse_args()

  samples = load_dataset(args.dataset)
  agent = Agent(name=args.agent_name, user_id=args.user_id)
  evaluator = AgentEvaluator(agent)
  report = evaluator.evaluate(samples=samples, include_memory=args.include_memory)

  save_report(report, args.out_json)
  Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out_md).write_text(build_markdown_summary(report), encoding="utf-8")

  summary = report["summary"]
  print("Evaluation complete.")
  print(f"Total: {summary['total_samples']}")
  print(f"Exact Match: {summary['exact_matches']}")
  print(f"Exact Match Rate: {summary['exact_match_rate']:.4f}")
  print(f"Avg Steps: {summary['avg_steps']:.2f}")
  print(f"Avg Latency(ms): {summary['avg_latency_ms']:.2f}")
  print(f"Avg Tool Calls: {summary['avg_tool_calls']:.2f}")
  print(f"JSON report: {args.out_json}")
  print(f"Markdown report: {args.out_md}")


if __name__ == "__main__":
  main()
