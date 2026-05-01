from serpapi import SerpApiClient

def search(query: str) -> str:
  print(f"正在搜索: '{query}'")
  try:
    api_key = 'c87ea373d6620a0daac33e6b38d6b0caf9c1199e0b7fef6783e052cc7a1c35d8'
    params = {
      "engine": "google",
      "q": query,
      "api_key": api_key,
      "gl": "cn",
      "hl": "zh-CN",
    }
    client = SerpApiClient(params)
    results = client.get_dict()
    # 处理回答框
    if "answer_box_list" in results:
      return "\n".join(results["answer_box_list"])
    # 处理单个回答框
    if "answer_box" in results and "answer" in results["answer_box"]:
      return results["answer_box"]["answer"]
    # 处理知识图谱
    if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
      return results["knowledge_graph"]["description"]
    # 处理有机结果
    if "organic_results" in results and results["organic_results"]:
      snippets = [
        f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
        for i, res in enumerate(results["organic_results"][:3])
      ]
      return "\n\n".join(snippets)
    return f"对不起，没有找到关于 '{query}' 的信息。"
  except Exception as e:
    return f"搜索时发生错误: {e}"

if __name__ == "__main__":
  query = "vue的创始人是谁？"
  result = search(query)
  print("搜索结果:")
  print(result)

