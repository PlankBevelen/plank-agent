import unittest

from AgentEvaluator import AgentEvaluator, EvalSample


class FakeAgent:
    def __init__(self):
        self.reset_calls = 0
        self.messages: list[str] = []

    def reset_session_state(self) -> None:
        self.reset_calls += 1
        self.messages = []

    def run(self, question: str, **kwargs):
        answer = f"history={len(self.messages)}"
        self.messages.append(question)
        return {
            "answer": answer,
            "elapsed_ms": 1,
            "steps": 0,
            "tool_calls": 0,
            "trace": [],
        }


class AgentEvaluatorTests(unittest.TestCase):
    def test_evaluate_resets_session_state_for_each_sample(self):
        samples = [
            EvalSample(id="1", question="q1", answer="history=0"),
            EvalSample(id="2", question="q2", answer="history=0"),
        ]
        agent = FakeAgent()
        evaluator = AgentEvaluator(agent)

        report = evaluator.evaluate(samples)

        self.assertEqual(agent.reset_calls, 2)
        self.assertEqual(report["summary"]["exact_matches"], 2)
        self.assertEqual(report["records"][1]["pred_answer"], "history=0")


if __name__ == "__main__":
    unittest.main()
