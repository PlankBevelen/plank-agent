from openai import OpenAI
from Constant import get_llm_api_base_url, get_llm_api_key, get_llm_model


class LLM:
  def __init__(self):
    api_key = get_llm_api_key()
    if not api_key:
      raise ValueError(
        "Missing LLM API key. Set PLANK_LLM_API_KEY, or OPENAI_API_KEY in .env."
      )

    self.model = get_llm_model()
    self.client = OpenAI(
      base_url=get_llm_api_base_url(),
      api_key=api_key,
    )

  def think(self, messages: list[dict[str, str]], temperature: float = 0.7, max_new_tokens: int = 512) -> str:
    stream = self.client.chat.completions.create(
      model=self.model,
      messages=messages,
      temperature=temperature,
      max_tokens=max_new_tokens,
      stream=True,
    )

    collected = []
    print("Assistant: ", end="", flush=True)
    for chunk in stream:
      if not chunk.choices:
        continue
      delta = chunk.choices[0].delta.content
      if delta is None:
        continue
      print(delta, end="", flush=True)
      collected.append(delta)
    print()

    return "".join(collected)


if __name__ == "__main__":
  llm = LLM()
  messages = [
    {"role": "system", "content": "You are my assistant."},
    {"role": "user", "content": "Please introduce yourself."},
  ]
  response = llm.think(messages)
  print("LLM Response:", response)
