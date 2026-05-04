from Tool import Tool

class ToolChain:
    """
    串联多个 Tool，上一步的输出作为下一步的输入。
    每个 step 是 (tool, input_mapper)：
      - input_mapper(prev_output: str) -> dict  用来转换上一步结果为下一步参数
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: list[tuple[Tool, callable]] = []

    def add_step(self, tool: Tool, input_mapper: callable = None):
        # 默认 mapper：直接把上一步结果当 query 传入
        if input_mapper is None:
            input_mapper = lambda prev: {"query": prev}
        self.steps.append((tool, input_mapper))
        return self  # 支持链式调用

    def run(self, initial_input: str) -> str:
        result = initial_input
        for tool, mapper in self.steps:
            kwargs = mapper(result)
            result = tool.run(**kwargs)
        return result