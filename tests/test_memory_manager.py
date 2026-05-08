import unittest

from MemoryManager import MemoryManager


class MemoryManagerTests(unittest.TestCase):
    def test_retrieve_sorts_by_score_importance_and_recency(self):
        class FakeKB:
            def search_with_meta(self, **kwargs):
                return [
                    {
                        "text": "older but relevant",
                        "score": 0.9,
                        "metadata": {"importance": 0.2, "created_at": "2024-01-01T00:00:00+00:00"},
                    },
                    {
                        "text": "important and recent",
                        "score": 0.8,
                        "metadata": {"importance": 1.0, "created_at": "2099-01-01T00:00:00+00:00"},
                    },
                ]

        manager = MemoryManager(kb=FakeKB(), write_enabled=False)
        ranked = manager.retrieve(user_id="u1", query="deploy")

        self.assertEqual(ranked[0]["text"], "important and recent")
        self.assertGreater(ranked[0]["rank_score"], ranked[1]["rank_score"])


if __name__ == "__main__":
    unittest.main()
