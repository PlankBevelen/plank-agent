import hashlib
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional

from constant import (
  get_memory_collection_name,
  get_memory_db_path,
  get_memory_threshold,
  get_memory_top_k,
  get_memory_write_enabled,
)
from KnowledgeBase import KnowledgeBase


class MemoryManager:
  _shared_instances: dict[tuple[str, str, int, float, bool], "MemoryManager"] = {}
  _shared_instances_lock = Lock()

  @classmethod
  def get_shared(
    cls,
    db_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    write_enabled: Optional[bool] = None,
  ) -> "MemoryManager":
    resolved_db_path = db_path or get_memory_db_path()
    resolved_collection = collection_name or get_memory_collection_name()
    resolved_top_k = top_k if top_k is not None else get_memory_top_k()
    resolved_threshold = threshold if threshold is not None else get_memory_threshold()
    resolved_write_enabled = (
      write_enabled if write_enabled is not None else get_memory_write_enabled()
    )
    key = (
      resolved_db_path,
      resolved_collection,
      resolved_top_k,
      resolved_threshold,
      resolved_write_enabled,
    )
    instance = cls._shared_instances.get(key)
    if instance is not None:
      return instance

    with cls._shared_instances_lock:
      instance = cls._shared_instances.get(key)
      if instance is None:
        instance = cls(
          db_path=resolved_db_path,
          collection_name=resolved_collection,
          top_k=resolved_top_k,
          threshold=resolved_threshold,
          write_enabled=resolved_write_enabled,
        )
        cls._shared_instances[key] = instance
    return instance

  def __init__(
    self,
    kb: Optional[KnowledgeBase] = None,
    db_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    write_enabled: Optional[bool] = None,
  ):
    self.kb = kb or KnowledgeBase(
      db_path=db_path or get_memory_db_path(),
      collection_name=collection_name or get_memory_collection_name(),
    )
    self.top_k = top_k if top_k is not None else get_memory_top_k()
    self.threshold = threshold if threshold is not None else get_memory_threshold()
    self.write_enabled = write_enabled if write_enabled is not None else get_memory_write_enabled()

  def _utc_now(self) -> str:
    return datetime.now(timezone.utc).isoformat()

  def _memory_id(self, user_id: str, text: str, ts: str) -> str:
    digest = hashlib.sha1(f"{user_id}|{ts}|{text}".encode("utf-8")).hexdigest()
    return f"mem_{digest[:20]}"

  def _normalize_importance(self, score: float) -> float:
    return max(0.0, min(1.0, float(score)))

  def estimate_importance(
    self,
    user_input: str,
    assistant_output: str,
    observations: Optional[list[str]] = None,
  ) -> float:
    text = (user_input + "\n" + assistant_output).lower()
    score = 0.4
    if len(user_input) > 80:
      score += 0.1
    if observations:
      score += 0.1
    if any(k in text for k in ["preference", "habit", "long-term", "plan", "goal", "todo"]):
      score += 0.25
    if any(k in text for k in ["name is", "my name", "my project", "my team", "my company"]):
      score += 0.2
    return self._normalize_importance(score)

  def save_interaction(
    self,
    user_id: str,
    user_input: str,
    assistant_output: str,
    observations: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    importance: Optional[float] = None,
  ) -> bool:
    if not self.write_enabled:
      return False

    ts = self._utc_now()
    obs_text = "\n".join(observations or [])
    mem_text = (
      f"[Time]\n{ts}\n\n"
      f"[User]\n{user_input.strip()}\n\n"
      f"[Assistant]\n{assistant_output.strip()}\n\n"
      f"[Observations]\n{obs_text.strip() if obs_text else '(none)'}"
    )
    importance_score = (
      self.estimate_importance(user_input, assistant_output, observations)
      if importance is None
      else self._normalize_importance(importance)
    )
    doc_id = self._memory_id(user_id=user_id, text=mem_text, ts=ts)
    metadata = {
      "type": "episodic",
      "user_id": user_id,
      "created_at": ts,
      "importance": importance_score,
      "tags": tags or [],
    }
    return self.kb.add(doc_id=doc_id, text=mem_text, metadata=metadata, force=False)

  def retrieve(
    self,
    user_id: str,
    query: str,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
  ) -> list[dict[str, Any]]:
    records = self.kb.search_with_meta(
      query=query,
      top_k=top_k if top_k is not None else self.top_k,
      threshold=threshold if threshold is not None else self.threshold,
      where={"user_id": user_id},
    )
    ranked = []
    for item in records:
      metadata = item.get("metadata", {})
      importance = metadata.get("importance", 0.5)
      try:
        importance = float(importance)
      except Exception:
        importance = 0.5

      created_at = metadata.get("created_at", "")
      recency_bonus = 0.0
      if created_at:
        try:
          created_at_dt = datetime.fromisoformat(str(created_at))
          age_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - created_at_dt.astimezone(timezone.utc)).total_seconds(),
          )
          recency_bonus = max(0.0, min(0.1, 0.1 * (1.0 / (1.0 + age_seconds / 86400.0))))
        except Exception:
          recency_bonus = 0.0

      score = (
        0.7 * float(item.get("score", 0.0))
        + 0.25 * self._normalize_importance(importance)
        + 0.05 * recency_bonus
      )
      new_item = dict(item)
      new_item["rank_score"] = score
      ranked.append(new_item)
    ranked.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
    return ranked

  def format_for_context(self, records: list[dict[str, Any]], max_items: int = 4, max_chars: int = 1200) -> str:
    if not records:
      return ""

    lines: list[str] = []
    total = 0
    for idx, item in enumerate(records[:max_items], start=1):
      text = (item.get("text") or "").strip()
      if not text:
        continue
      metadata = item.get("metadata", {})
      created_at = metadata.get("created_at", "unknown")
      rank = item.get("rank_score", 0.0)
      block = f"[Memory {idx}] rank={rank:.3f} time={created_at}\n{text}"
      if total + len(block) > max_chars:
        break
      lines.append(block)
      total += len(block)
    return "\n\n".join(lines)
