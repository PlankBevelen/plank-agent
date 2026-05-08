from dataclasses import dataclass

from constant import (
  get_context_max_chars,
  get_context_max_history_turns,
  get_context_max_kb_items,
  get_context_max_memory_items,
)


@dataclass
class ContextPack:
  user_input: str
  history_text: str
  kb_text: str
  memory_text: str
  observations_text: str
  final_context: str


class ContextBuilder:
  def __init__(
    self,
    max_chars: int | None = None,
    max_history_turns: int | None = None,
    max_kb_items: int | None = None,
    max_memory_items: int | None = None,
  ):
    self.max_chars = max_chars if max_chars is not None else get_context_max_chars()
    self.max_history_turns = (
      max_history_turns if max_history_turns is not None else get_context_max_history_turns()
    )
    self.max_kb_items = max_kb_items if max_kb_items is not None else get_context_max_kb_items()
    self.max_memory_items = (
      max_memory_items if max_memory_items is not None else get_context_max_memory_items()
    )

  def _trim(self, text: str, limit: int) -> str:
    if len(text) <= limit:
      return text
    if limit <= 3:
      return text[:limit]
    return text[: limit - 3] + "..."

  def build_history_text(self, messages: list[dict[str, str]]) -> str:
    if not messages:
      return ""
    convo = [m for m in messages if m.get("role") in {"user", "assistant"}]
    if not convo:
      return ""

    take = self.max_history_turns * 2
    window = convo[-take:]
    blocks = []
    for item in window:
      role = item.get("role", "unknown").upper()
      content = (item.get("content") or "").strip()
      if not content:
        continue
      blocks.append(f"[{role}]\n{content}")
    return "\n\n".join(blocks)

  def build_kb_text(self, kb_results: list[str]) -> str:
    if not kb_results:
      return ""
    blocks = []
    for idx, text in enumerate(kb_results[: self.max_kb_items], start=1):
      content = (text or "").strip()
      if not content:
        continue
      blocks.append(f"[KB {idx}]\n{content}")
    return "\n\n".join(blocks)

  def build_observations_text(self, observations: list[str]) -> str:
    if not observations:
      return ""
    blocks = []
    for idx, obs in enumerate(observations, start=1):
      content = (obs or "").strip()
      if not content:
        continue
      blocks.append(f"[Observation {idx}]\n{content}")
    return "\n\n".join(blocks)

  def build(
    self,
    user_input: str,
    messages: list[dict[str, str]],
    kb_results: list[str],
    memory_text: str,
    observations: list[str],
  ) -> ContextPack:
    history_text = self.build_history_text(messages)
    kb_text = self.build_kb_text(kb_results)
    observations_text = self.build_observations_text(observations)

    sections: list[str] = [f"[Current User Query]\n{user_input.strip()}"]
    if history_text:
      sections.append(f"[Recent Conversation]\n{history_text}")
    if memory_text:
      sections.append(f"[Relevant Memory]\n{memory_text}")
    if kb_text:
      sections.append(f"[Knowledge Base]\n{kb_text}")
    if observations_text:
      sections.append(f"[Tool Observations]\n{observations_text}")

    context = "\n\n".join(sections)
    if len(context) > self.max_chars:
      history_budget = int(self.max_chars * 0.35)
      memory_budget = int(self.max_chars * 0.25)
      kb_budget = int(self.max_chars * 0.25)
      obs_budget = int(self.max_chars * 0.15)
      sections = [f"[Current User Query]\n{self._trim(user_input.strip(), 800)}"]
      if history_text:
        sections.append(f"[Recent Conversation]\n{self._trim(history_text, history_budget)}")
      if memory_text:
        sections.append(f"[Relevant Memory]\n{self._trim(memory_text, memory_budget)}")
      if kb_text:
        sections.append(f"[Knowledge Base]\n{self._trim(kb_text, kb_budget)}")
      if observations_text:
        sections.append(f"[Tool Observations]\n{self._trim(observations_text, obs_budget)}")
      context = self._trim("\n\n".join(sections), self.max_chars)

    return ContextPack(
      user_input=user_input,
      history_text=history_text,
      kb_text=kb_text,
      memory_text=memory_text,
      observations_text=observations_text,
      final_context=context,
    )
