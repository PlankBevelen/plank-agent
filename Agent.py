import re
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Iterator

from Constant import get_context_max_memory_items, get_react_max_steps
from ContextBuilder import ContextBuilder, ContextPack
from KnowledgeBase import KnowledgeBase
from LLM import LLM
from MemoryManager import MemoryManager
from PromptLoader import PromptLoader
from Search import search
from Tool import Tool
from ToolExecutor import ToolExecutor


@dataclass
class ReactLoopResult:
    observations: list[str]
    step_traces: list[dict]
    planner_note: str | None
    used_tools: bool


@dataclass
class TurnPreparation:
    loop_result: ReactLoopResult
    context_pack: ContextPack
    prompt_name: str
    prompt_text: str


class Agent:
    _shared_kb = None
    _shared_kb_lock = Lock()

    @classmethod
    def _get_shared_kb(cls) -> KnowledgeBase:
        if cls._shared_kb is not None:
            return cls._shared_kb
        with cls._shared_kb_lock:
            if cls._shared_kb is None:
                cls._shared_kb = KnowledgeBase(collection_name="plankbevelen")
        return cls._shared_kb

    @classmethod
    def prewarm(cls) -> None:
        cls._get_shared_kb()

    def _retrieve_kb_results(self, user_input: str, top_k: int = 4) -> list[str]:
        strict = self.kb.search(user_input, top_k=top_k, threshold=0.55)
        if strict:
            return strict

        relaxed = self.kb.search_with_meta(
            query=user_input,
            top_k=top_k,
            threshold=1.25,
        )
        return [item.get("text", "") for item in relaxed if item.get("text")]

    def __init__(self, name: str, user_id: str = "default_user"):
        self.name = name
        self.user_id = user_id

        self.llm = LLM()
        self.prompt = PromptLoader()
        self.prompt.validate()

        self.kb = self._get_shared_kb()
        self.memory = MemoryManager.get_shared()
        self.context_builder = ContextBuilder()

        self.executor = ToolExecutor()
        self.executor.register(Tool("Search", "Web search via SerpAPI.", search))

        self.system_prompt = self.prompt.load("system")
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def export_session_state(self) -> list[dict[str, str]]:
        return [
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
            for message in self.messages
            if message.get("role") in {"user", "assistant"} and message.get("content")
        ]

    def restore_session_state(self, messages: list[dict[str, str]] | None) -> None:
        restored = [{"role": "system", "content": self.system_prompt}]
        for message in messages or []:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            restored.append({"role": role, "content": content})
        self.messages = restored[:1] + restored[-18:]

    def _llm_messages(self, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_action(self, text: str) -> tuple[str, str] | None:
        match = re.search(r"Action:\s*(\w+)\[(.+?)\]", text)
        if match:
            return match.group(1), match.group(2)
        return None

    def _build_memory_context(self, user_input: str) -> str:
        memory_records = self.memory.retrieve(user_id=self.user_id, query=user_input)
        return self.memory.format_for_context(
            memory_records,
            max_items=get_context_max_memory_items(),
            max_chars=1400,
        )

    def _build_context(self, user_input: str, observations: list[str], include_memory: bool = True) -> str:
        return self._build_context_pack(
            user_input=user_input,
            observations=observations,
            include_memory=include_memory,
        ).final_context

    def _build_context_pack(
        self,
        user_input: str,
        observations: list[str],
        include_memory: bool = True,
    ) -> ContextPack:
        kb_results = self._retrieve_kb_results(user_input, top_k=4)
        memory_text = self._build_memory_context(user_input) if include_memory else ""
        return self.context_builder.build(
            user_input=user_input,
            messages=self.messages,
            kb_results=kb_results,
            memory_text=memory_text,
            observations=observations,
        )

    def _run_react_loop(
        self,
        user_input: str,
        include_memory: bool = True,
        silent: bool = False,
    ) -> ReactLoopResult:
        tool_descriptions = self.executor.describe_all()
        observations: list[str] = []
        step_traces: list[dict] = []

        for step in range(get_react_max_steps()):
            step_context = self._build_context(
                user_input=user_input,
                observations=observations,
                include_memory=include_memory,
            )
            obs_text = "\n".join(f"Observation {i + 1}: {o}" for i, o in enumerate(observations))
            decision_prompt = self.prompt.load(
                "decision",
                context=step_context,
                tools=tool_descriptions,
                observations=obs_text,
            )
            decision = self.llm.think(
                self._llm_messages(decision_prompt),
                temperature=0,
                max_new_tokens=96,
                stream_output=not silent,
            )
            if not silent:
                print(f"Step {step + 1} decision: {decision}")

            action = self._parse_action(decision)
            if action is None:
                step_traces.append(
                    {
                        "step": step + 1,
                        "decision": decision,
                        "action": None,
                        "observation": None,
                    }
                )
                return ReactLoopResult(
                    observations=observations,
                    step_traces=step_traces,
                    planner_note=decision,
                    used_tools=bool(observations),
                )

            tool_name, query = action
            if not silent:
                print(f"  -> Tool: {tool_name}, Query: {query}")
            result = self.executor.run(tool_name, query)
            observations.append(result)
            step_traces.append(
                {
                    "step": step + 1,
                    "decision": decision,
                    "action": {"tool": tool_name, "query": query},
                    "observation": result,
                }
            )

        return ReactLoopResult(
            observations=observations,
            step_traces=step_traces,
            planner_note=None,
            used_tools=bool(observations),
        )

    def _prepare_turn(
        self,
        user_input: str,
        include_memory: bool = True,
        silent: bool = False,
    ) -> TurnPreparation:
        loop_result = self._run_react_loop(
            user_input=user_input,
            include_memory=include_memory,
            silent=silent,
        )
        pack = self._build_context_pack(
            user_input=user_input,
            observations=loop_result.observations,
            include_memory=include_memory,
        )
        prompt_name = "answer_with_search" if loop_result.used_tools else "answer"
        prompt_text = self.prompt.load(
            prompt_name,
            context=pack.final_context,
            kb_context=pack.kb_text,
            search_result="\n".join(loop_result.observations),
            planner_note=loop_result.planner_note or "",
        )
        return TurnPreparation(
            loop_result=loop_result,
            context_pack=pack,
            prompt_name=prompt_name,
            prompt_text=prompt_text,
        )

    def _finalize_turn(
        self,
        user_input: str,
        final_answer: str,
        observations: list[str],
        persist_memory: bool = True,
    ) -> str:
        final_answer = final_answer.strip()

        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": final_answer})

        if len(self.messages) > 20:
            self.messages = self.messages[:1] + self.messages[-18:]

        if persist_memory:
            self.memory.save_interaction(
                user_id=self.user_id,
                user_input=user_input,
                assistant_output=final_answer,
                observations=observations,
                tags=["chat", "react"],
            )

        return final_answer

    def run(
        self,
        user_input: str,
        return_trace: bool = False,
        silent: bool = False,
        include_memory: bool = True,
        persist_memory: bool = True,
    ) -> str | dict:
        start = perf_counter()
        if not silent:
            print(f"User: {user_input}")

        preparation = self._prepare_turn(
            user_input=user_input,
            include_memory=include_memory,
            silent=silent,
        )
        final_answer = self.llm.think(
            self._llm_messages(preparation.prompt_text),
            stream_output=not silent,
        )
        if not final_answer.strip():
            final_answer = preparation.loop_result.planner_note or ""

        final_answer = self._finalize_turn(
            user_input=user_input,
            final_answer=final_answer,
            observations=preparation.loop_result.observations,
            persist_memory=persist_memory,
        )

        elapsed_ms = int((perf_counter() - start) * 1000)
        if not silent:
            print(f"Answer: {final_answer}")
            print(f"Elapsed: {elapsed_ms} ms")

        if return_trace:
            return {
                "answer": final_answer,
                "elapsed_ms": elapsed_ms,
                "steps": len(preparation.loop_result.step_traces),
                "tool_calls": sum(1 for s in preparation.loop_result.step_traces if s.get("action")),
                "observations": preparation.loop_result.observations,
                "trace": preparation.loop_result.step_traces,
            }
        return final_answer

    def run_stream(
        self,
        user_input: str,
        include_memory: bool = True,
        persist_memory: bool = True,
    ) -> Iterator[str]:
        if not user_input:
            return

        preparation = self._prepare_turn(
            user_input=user_input,
            include_memory=include_memory,
            silent=True,
        )
        collected: list[str] = []

        for delta in self.llm.stream_think(
            self._llm_messages(preparation.prompt_text),
            stream_output=False,
        ):
            collected.append(delta)
            yield delta

        final_answer = "".join(collected).strip() or (preparation.loop_result.planner_note or "")
        if not collected and final_answer:
            yield final_answer

        self._finalize_turn(
            user_input=user_input,
            final_answer=final_answer,
            observations=preparation.loop_result.observations,
            persist_memory=persist_memory,
        )


if __name__ == "__main__":
    try:
        agent = Agent("PlankAgent")
        print("Agent started. Type your question, or 'exit' to quit.")
        while True:
            user_input = input("Question (or 'exit'): ").strip()
            if user_input.lower() == "exit":
                print("Agent exited.")
                break
            if not user_input:
                continue
            agent.run(user_input)
    except Exception as e:
        print(f"Agent startup failed: {e}")
        print("Check your LLM API settings in .env, then rerun `python Agent.py`.")
