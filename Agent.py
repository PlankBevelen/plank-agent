import re
from time import perf_counter
from typing import Iterator

from Constant import get_context_max_memory_items, get_react_max_steps
from ContextBuilder import ContextBuilder
from KnowledgeBase import KnowledgeBase
from LLM import LLM
from MemoryManager import MemoryManager
from PromptLoader import PromptLoader
from Search import search
from Tool import Tool
from ToolExecutor import ToolExecutor


class Agent:
    def __init__(self, name: str, user_id: str = "default_user"):
        self.name = name
        self.user_id = user_id

        self.llm = LLM()
        self.prompt = PromptLoader()

        self.kb = KnowledgeBase(collection_name="plankbevelen")
        self.memory = MemoryManager()
        self.context_builder = ContextBuilder()

        self.executor = ToolExecutor()
        self.executor.register(Tool("Search", "Web search via SerpAPI.", search))

        system_prompt = self.prompt.load("system")
        self.messages = [{"role": "system", "content": system_prompt}]

    def _parse_action(self, text: str) -> tuple[str, str] | None:
        """Parse `Action: ToolName[query]` from model output."""
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
        kb_results = self.kb.search(user_input, top_k=3, threshold=0.5)
        memory_text = self._build_memory_context(user_input) if include_memory else ""
        pack = self.context_builder.build(
            user_input=user_input,
            messages=self.messages,
            kb_results=kb_results,
            memory_text=memory_text,
            observations=observations,
        )
        return pack.final_context

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

        tool_descriptions = self.executor.describe_all()
        observations: list[str] = []
        final_answer = None
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
                user_input=step_context,
                tools=tool_descriptions,
                observations=obs_text,
            )
            decision = self.llm.think(
                [{"role": "user", "content": decision_prompt}],
                temperature=0,
                max_new_tokens=96,
                stream_output=not silent,
            )
            if not silent:
                print(f"Step {step + 1} decision: {decision}")

            action = self._parse_action(decision)
            if action is None:
                final_answer = decision
                step_traces.append(
                    {
                        "step": step + 1,
                        "decision": decision,
                        "action": None,
                        "observation": None,
                    }
                )
                break

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

        if final_answer is None:
            answer_prompt = self.prompt.load(
                "answer_with_search",
                user_input=user_input,
                search_result="\n".join(observations),
            )
            final_answer = self.llm.think(
                [{"role": "user", "content": answer_prompt}],
                stream_output=not silent,
            )

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

        elapsed_ms = int((perf_counter() - start) * 1000)
        if not silent:
            print(f"Answer: {final_answer}")
            print(f"Elapsed: {elapsed_ms} ms")

        if return_trace:
            return {
                "answer": final_answer,
                "elapsed_ms": elapsed_ms,
                "steps": len(step_traces),
                "tool_calls": sum(1 for s in step_traces if s.get("action")),
                "observations": observations,
                "trace": step_traces,
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

        tool_descriptions = self.executor.describe_all()
        observations: list[str] = []
        final_answer = None

        for _step in range(get_react_max_steps()):
            step_context = self._build_context(
                user_input=user_input,
                observations=observations,
                include_memory=include_memory,
            )
            obs_text = "\n".join(f"Observation {i + 1}: {o}" for i, o in enumerate(observations))
            decision_prompt = self.prompt.load(
                "decision",
                user_input=step_context,
                tools=tool_descriptions,
                observations=obs_text,
            )
            decision = self.llm.think(
                [{"role": "user", "content": decision_prompt}],
                temperature=0,
                max_new_tokens=96,
                stream_output=False,
            )

            action = self._parse_action(decision)
            if action is None:
                final_answer = decision
                break

            tool_name, query = action
            result = self.executor.run(tool_name, query)
            observations.append(result)

        if final_answer is None:
            answer_prompt = self.prompt.load(
                "answer_with_search",
                user_input=user_input,
                search_result="\n".join(observations),
            )
            collected: list[str] = []
            for delta in self.llm.stream_think(
                [{"role": "user", "content": answer_prompt}],
                stream_output=False,
            ):
                collected.append(delta)
                yield delta
            final_answer = "".join(collected)
        else:
            yield final_answer

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
