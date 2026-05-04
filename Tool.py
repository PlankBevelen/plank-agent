from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]
    # 参数 schema，未来可扩展为 JSON Schema 供 LLM 解析
    params_schema: dict = field(default_factory=lambda: {"query": "str"})

    def run(self, **kwargs) -> str:
        try:
            return self.func(**kwargs)
        except Exception as e:
            return f"Tool '{self.name}' error: {e}"