import unittest

from Agent import Agent, ReactLoopResult, TurnResources
from ContextBuilder import ContextPack


class FakePromptLoader:
    def __init__(self):
        self.calls = []

    def load(self, name: str, **kwargs) -> str:
        self.calls.append((name, kwargs))
        return f"{name}:{kwargs}"


class AgentCoreTests(unittest.TestCase):
    def test_parse_action_extracts_tool_and_query(self):
        agent = Agent.__new__(Agent)
        self.assertEqual(agent._parse_action("Action: Search[latest news]"), ("Search", "latest news"))
        self.assertIsNone(agent._parse_action("No tool needed"))

    def test_retrieve_kb_results_falls_back_when_strict_is_empty(self):
        class FakeKB:
            def __init__(self):
                self.calls = []

            def search(self, query: str, top_k: int, threshold: float):
                self.calls.append(("search", query, top_k, threshold))
                return []

            def search_with_meta(self, query: str, top_k: int, threshold: float):
                self.calls.append(("search_with_meta", query, top_k, threshold))
                return [{"text": "fallback hit"}, {"text": ""}]

        agent = Agent.__new__(Agent)
        agent.kb = FakeKB()
        self.assertEqual(agent._retrieve_kb_results("topic"), ["fallback hit"])

    def test_prepare_turn_uses_full_context_for_search_response(self):
        agent = Agent.__new__(Agent)
        agent.prompt = FakePromptLoader()
        agent._prepare_turn_resources = lambda **_: TurnResources(kb_results=["kb"], memory_text="memory")
        agent._run_react_loop = lambda **_: ReactLoopResult(
            observations=["obs-1"],
            step_traces=[],
            planner_note="draft",
            used_tools=True,
        )
        agent._build_context_pack = lambda **_: ContextPack(
            user_input="question",
            history_text="history",
            kb_text="[KB 1]\nkb",
            memory_text="memory",
            observations_text="obs",
            final_context="FULL_CONTEXT",
        )

        preparation = agent._prepare_turn("question", include_memory=True, silent=True)

        self.assertEqual(preparation.prompt_name, "answer_with_search")
        prompt_name, kwargs = agent.prompt.calls[-1]
        self.assertEqual(prompt_name, "answer_with_search")
        self.assertEqual(kwargs["context"], "FULL_CONTEXT")
        self.assertEqual(kwargs["search_result"], "obs-1")
        self.assertEqual(kwargs["planner_note"], "draft")

    def test_prepare_turn_reuses_turn_resources_across_react_and_final_answer(self):
        agent = Agent.__new__(Agent)
        agent.prompt = FakePromptLoader()
        agent.messages = [{"role": "system", "content": "system"}]

        counts = {"kb": 0, "memory": 0}

        def fake_retrieve(user_input: str, top_k: int = 4):
            counts["kb"] += 1
            return ["kb"]

        def fake_memory(user_input: str):
            counts["memory"] += 1
            return "memory"

        class FakeContextBuilder:
            def build(self, **kwargs):
                return ContextPack(
                    user_input=kwargs["user_input"],
                    history_text="history",
                    kb_text="[KB 1]\nkb",
                    memory_text=kwargs["memory_text"],
                    observations_text="",
                    final_context="FULL_CONTEXT",
                )

        def fake_run_react_loop(**kwargs):
            Agent._build_context_pack(
                agent,
                user_input=kwargs["user_input"],
                observations=[],
                include_memory=kwargs["include_memory"],
                turn_resources=kwargs["turn_resources"],
            )
            return ReactLoopResult(
                observations=[],
                step_traces=[],
                planner_note="draft",
                used_tools=False,
            )

        agent._retrieve_kb_results = fake_retrieve
        agent._build_memory_context = fake_memory
        agent.context_builder = FakeContextBuilder()
        agent._run_react_loop = fake_run_react_loop

        preparation = agent._prepare_turn("question", include_memory=True, silent=True)

        self.assertEqual(preparation.prompt_name, "answer")
        self.assertEqual(counts["kb"], 1)
        self.assertEqual(counts["memory"], 1)


if __name__ == "__main__":
    unittest.main()
