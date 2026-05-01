import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

class LLM:
  def __init__(self):
    self.model_id = "QWen/Qwen2.5-1.5B-Instruct"
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = AutoModelForCausalLM.from_pretrained(
      self.model_id,
      dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(self.device)
  
  """ 思考 """
  def think(self, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
    input_text = self.tokenizer.apply_chat_template(
      messages,
      tokenize=False, # 不需要手动分词
      add_generation_prompt=True  # 添加生成提示
    )
    inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

    # 设置流式输出
    streamer = TextIteratorStreamer(
      self.tokenizer,
      skip_prompt=True,  # 跳过提示词
      skip_special_tokens=True  # 跳过特殊标记
    )
    # 设置生成参数
    generate_kwargs = dict(
      **inputs,
      streamer=streamer,
      max_new_tokens=512,
      temperature=temperature,
      do_sample=temperature > 0,  # 温度大于0时启用采样
    )
    # 在单独的线程中生成文本
    thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
    thread.start()

    collected = []
    print("Assistant: ", end="", flush=True)
    for token_text in streamer:
      print(token_text, end="", flush=True)
      collected.append(token_text)
    print()  # 换行

    thread.join() # 等待生成线程完成
    return "".join(collected)

if __name__ == "__main__":
  # 测试代码
  llm = LLM()
  messages = [
    {"role": "system", "content": "你是我的人工智能助手，协助我解答问题。"},
    {"role": "user", "content": "请介绍一下你自己。"}
  ]
  response = llm.think(messages)
  print("LLM Response:", response)