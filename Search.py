import serpapi

from constant import get_serpapi_key


def search(query: str) -> str:
  print(f"searching: '{query}'")
  try:
    api_key = get_serpapi_key()
    if not api_key:
      return "Search error: SERPAPI_KEY is not set."

    params = {
      "engine": "google",
      "q": query,
      "api_key": api_key,
      "gl": "cn",
      "hl": "zh-CN",
    }
    search_results = serpapi.search(params, timeout=10)
    results = search_results.as_dict() if hasattr(search_results, "as_dict") else dict(search_results)

    if "answer_box_list" in results:
      return "\n".join(str(item) for item in results["answer_box_list"])
    if "answer_box" in results and "answer" in results["answer_box"]:
      return results["answer_box"]["answer"]
    if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
      return results["knowledge_graph"]["description"]
    if "organic_results" in results and results["organic_results"]:
      snippets = [
        f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
        for i, res in enumerate(results["organic_results"][:3])
      ]
      return "\n\n".join(snippets)

    return f"No results found for '{query}'."
  except Exception as e:
    return f"Search error: {e}"


if __name__ == "__main__":
  query = "who created vue"
  result = search(query)
  print("Search result:")
  print(result)
