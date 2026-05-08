import tempfile
import unittest
from pathlib import Path

from PromptLoader import PromptLoader


class PromptLoaderTests(unittest.TestCase):
    def test_load_replaces_placeholders_without_touching_literal_braces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_dir = Path(tmpdir)
            (prompt_dir / "sample.txt").write_text(
                "Hello {name}. Literal braces: {{ keep me }}.",
                encoding="utf-8",
            )
            loader = PromptLoader(str(prompt_dir))
            rendered = loader.load("sample", name="Agent")

            self.assertEqual(rendered, "Hello Agent. Literal braces: { keep me }.")

    def test_validate_detects_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_dir = Path(tmpdir)
            (prompt_dir / "answer.txt").write_text("Hi {user_input}", encoding="utf-8")
            loader = PromptLoader(str(prompt_dir))

            with self.assertRaises(ValueError):
                loader.validate({"answer": {"context"}})


if __name__ == "__main__":
    unittest.main()
