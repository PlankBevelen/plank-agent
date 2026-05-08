import unittest

from ContextBuilder import ContextBuilder


class ContextBuilderTests(unittest.TestCase):
    def test_build_includes_expected_sections(self):
        builder = ContextBuilder(
            max_chars=1000,
            max_history_turns=2,
            max_kb_items=2,
            max_memory_items=2,
        )
        pack = builder.build(
            user_input="How do we deploy this?",
            messages=[
                {"role": "system", "content": "ignore"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "deploy question"},
            ],
            kb_results=["kb-1", "kb-2"],
            memory_text="memory-1",
            observations=["search-1"],
        )

        self.assertIn("[Current User Query]", pack.final_context)
        self.assertIn("[Recent Conversation]", pack.final_context)
        self.assertIn("[Relevant Memory]", pack.final_context)
        self.assertIn("[Knowledge Base]", pack.final_context)
        self.assertIn("[Tool Observations]", pack.final_context)

    def test_build_trims_when_context_exceeds_budget(self):
        builder = ContextBuilder(max_chars=120, max_history_turns=4, max_kb_items=3, max_memory_items=3)
        pack = builder.build(
            user_input="Q" * 100,
            messages=[{"role": "user", "content": "U" * 80}, {"role": "assistant", "content": "A" * 80}],
            kb_results=["K" * 80],
            memory_text="M" * 80,
            observations=["O" * 80],
        )

        self.assertLessEqual(len(pack.final_context), 120)
        self.assertIn("...", pack.final_context)


if __name__ == "__main__":
    unittest.main()
