from LLM import LLM
from ToolExecutor import ToolExecutor
from Search import search
from typing import Dict, Any
from PromptLoader import PromptLoader
from KnowledgeBase import KnowledgeBase

class Agent:
  def __init__(self, name):
    self.llm = LLM()
    self.tool_executor = ToolExecutor()
    self.tool_executor.registerTool(
      "Search", 
      "一个基于SerpApi的实战网页搜索引擎工具。它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。", 
      search
    )
    self.prompt = PromptLoader()
    self.kb = KnowledgeBase()
    # 从文件加载系统提示，格式化后添加到对话历史
    system_prompt = self.prompt.load("system")
    self.messages = [
      {"role": "system", "content": system_prompt}
    ]

  def run(self, user_input: str):
    print(f"用户输入: {user_input}")
    self.messages.append({"role": "user", "content": user_input})

    # 第一步检查知识库
    kb_results = self.kb.search(user_input, top_k=3, threshold=0.5)
    kb_context = "\n\n".join(kb_results) if kb_results else ""
    print(f"知识库查询结果: {kb_context if kb_context else '无'}")  

    # 第二步 decision prompt 决策
    user_input_with_kb = user_input
    if kb_context:
      user_input_with_kb = (
        f"{user_input}\n\n"
        f"（本地知识库已找到以下相关内容，如果足够回答请直接回答）：\n{kb_context}"
      )
    decision_prompt = self.prompt.load("decision", user_input=user_input_with_kb)
    decision = self.llm.think([{"role": "user", "content": decision_prompt}])
    print(f"Agent决策: {decision}")

    # 第三步 执行决策
    if "Action: Search[" in decision:
      query = decision.split("Action: Search[")[1].split("]")[0]
      print(f"调用搜索工具，查询: {query}")
      search_result = self.tool_executor.runTool("Search", query)
      print(f"工具调用结果: {search_result}")

      # KB 有内容时合并进搜索结果，一起交给 answer_with_search prompt
      combined_result = search_result
      if kb_context:
        combined_result = f"本地知识库：\n{kb_context}\n\n联网搜索：\n{search_result}"
 
      answer_prompt = self.prompt.load(
        "answer_with_search",
        user_input=user_input,
        search_result=combined_result
      )
      final_answer = self.llm.think([{"role": "user", "content": answer_prompt}])
      print(f"Agent回答: {final_answer}")
      self.messages.append({"role": "assistant", "content": final_answer})
    else:
      # decision.txt 让 LLM 直接给出了回答（基于 KB 或自身知识）
      answer_prompt = self.prompt.load(
        "answer",
        user_input=user_input,
        kb_context=kb_context
      )
      final_answer = self.llm.think([{"role": "user", "content": answer_prompt}])
      print("直接回答（基于知识库或自身知识）", final_answer)
      self.messages.append({"role": "assistant", "content": final_answer})
  
  def loop(self):
    print("Agent进入交互循环，等待用户输入...")
    while True:
      user_input = input("请输入问题（输入 'exit' 退出）：")
      if user_input.lower() == "exit":
        print("Agent已退出。")
        break
      self.run(user_input)
    
if __name__ == "__main__":
  agent = Agent("PlankBevelen")
  agent.loop()