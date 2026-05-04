from Tool import Tool
from ToolChain import ToolChain

class ToolExecutor:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._chains: dict[str, ToolChain] = {}

    def register(self, item: Tool | ToolChain):
        if isinstance(item, Tool):
            self._tools[item.name] = item
        elif isinstance(item, ToolChain):
            self._chains[item.name] = item
        else:
            raise TypeError(f"Expected Tool or ToolChain, got {type(item)}")

    def run(self, name: str, query: str) -> str:
        if name in self._chains:
            return self._chains[name].run(query)
        if name in self._tools:
            return self._tools[name].run(query=query)
        available = list(self._tools) + list(self._chains)
        return f"Unknown tool '{name}'. Available: {available}"

    def describe_all(self) -> str:
        """生成给 LLM 看的工具列表描述（用于 decision prompt）"""
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
        for c in self._chains.values():
            lines.append(f"- {c.name} (chain): {c.description}")
        return "\n".join(lines)
