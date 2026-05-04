import re
from time import perf_counter
from KnowledgeBase import KnowledgeBase
from LLM import LLM
from PromptLoader import PromptLoader
from Search import search
from Tool import Tool
from ToolChain import ToolChain
from ToolExecutor import ToolExecutor
from constant import get_react_max_steps

class Agent:
    def __init__(self, name: str):
        self.name = name
        self.llm = LLM()
        self.prompt = PromptLoader()
        self.kb = KnowledgeBase()

        self.executor = ToolExecutor()
        self.executor.register(Tool("Search", "Web search via SerpAPI.", search))

        system_prompt = self.prompt.load("system")
        self.messages = [{"role": "system", "content": system_prompt}]

    def _parse_action(self, text: str) -> tuple[str, str] | None:
        """从 LLM 输出解析 Action: ToolName[query]，返回 (tool_name, query) 或 None"""
        match = re.search(r"Action:\s*(\w+)\[(.+?)\]", text)
        if match:
            return match.group(1), match.group(2)
        return None

    def run(self, user_input: str) -> str:
        start = perf_counter()
        print(f"User: {user_input}")

        # KB 检索
        kb_results = self.kb.search(user_input, top_k=3, threshold=0.5)
        kb_context = "\n\n".join(kb_results) if kb_results else ""

        # 构建初始 context
        context = user_input
        if kb_context:
            context += f"\n\n[KB Context]\n{kb_context}"

        tool_descriptions = self.executor.describe_all()
        observations = []

        final_answer = None
        for step in range(get_react_max_steps()):
            obs_text = "\n".join(
                f"Observation {i+1}: {o}" for i, o in enumerate(observations)
            )
            decision_prompt = self.prompt.load(
                "decision",
                user_input=context,
                tools=tool_descriptions,
                observations=obs_text,
            )
            decision = self.llm.think(
                [{"role": "user", "content": decision_prompt}],
                temperature=0,
                max_new_tokens=64,
            )
            print(f"Step {step+1} decision: {decision}")

            action = self._parse_action(decision)
            if action is None:
                # LLM 决定直接回答
                final_answer = decision
                break

            tool_name, query = action
            print(f"  → Tool: {tool_name}, Query: {query}")
            result = self.executor.run(tool_name, query)
            observations.append(result)

        if final_answer is None:
            # 超过步数上限，强制生成答案
            answer_prompt = self.prompt.load(
                "answer_with_search",
                user_input=user_input,
                search_result="\n".join(observations),
            )
            final_answer = self.llm.think([{"role": "user", "content": answer_prompt}])

        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": final_answer})
        # 防止 context 无限增长
        if len(self.messages) > 20:
            self.messages = self.messages[:1] + self.messages[-18:]

        print(f"Answer: {final_answer}")
        print(f"Elapsed: {int((perf_counter() - start)*1000)} ms")
        return final_answer