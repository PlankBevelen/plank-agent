from __future__ import annotations

import re
from pathlib import Path


_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")


class PromptLoader:
    REQUIRED_PROMPTS: dict[str, set[str]] = {
        "system": set(),
        "decision": {"tools", "context", "observations"},
        "answer": {"context", "kb_context", "planner_note"},
        "answer_with_search": {"context", "kb_context", "search_result", "planner_note"},
    }
    _validated_dirs: set[str] = set()

    def __init__(self, prompt_dir: str = "./prompts"):
        self.prompt_dir = Path(prompt_dir)
        self.cache: dict[str, str] = {}

    def _path_for(self, prompt_name: str) -> Path:
        return self.prompt_dir / f"{prompt_name}.txt"

    def _read(self, prompt_name: str) -> str:
        if prompt_name not in self.cache:
            path = self._path_for(prompt_name)
            self.cache[prompt_name] = path.read_text(encoding="utf-8")
        return self.cache[prompt_name]

    def placeholders(self, prompt_name: str) -> set[str]:
        return {match.group(1) for match in _PLACEHOLDER_RE.finditer(self._read(prompt_name))}

    def validate(self, required_prompts: dict[str, set[str]] | None = None) -> None:
        manifest = required_prompts or self.REQUIRED_PROMPTS
        cache_key = f"{self.prompt_dir.resolve()}::{sorted((k, tuple(sorted(v))) for k, v in manifest.items())}"
        if cache_key in self._validated_dirs:
            return

        errors: list[str] = []
        for prompt_name, expected_vars in manifest.items():
            path = self._path_for(prompt_name)
            if not path.is_file():
                errors.append(f"missing prompt file: {path}")
                continue

            try:
                actual_vars = self.placeholders(prompt_name)
            except UnicodeDecodeError as exc:
                errors.append(f"prompt is not valid UTF-8: {path} ({exc})")
                continue

            if actual_vars != expected_vars:
                errors.append(
                    f"prompt variables mismatch for {path}: expected {sorted(expected_vars)}, got {sorted(actual_vars)}"
                )

        if errors:
            raise ValueError("Prompt validation failed:\n- " + "\n- ".join(errors))

        self._validated_dirs.add(cache_key)

    def load(self, prompt_name: str, **kwargs) -> str:
        template = self._read(prompt_name)
        required = self.placeholders(prompt_name)
        missing = sorted(key for key in required if key not in kwargs)
        if missing:
            raise KeyError(f"Missing prompt variables for '{prompt_name}': {missing}")

        rendered = _PLACEHOLDER_RE.sub(lambda match: str(kwargs[match.group(1)]), template)
        return rendered.replace("{{", "{").replace("}}", "}")
