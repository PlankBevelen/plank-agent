
class PromptLoader:
  def __init__(self, prompt_dir: str = './prompts'):
    self.prompt_dir = prompt_dir
    self.cache = {}
  
  def load(self, name: str, **kwargs) -> str:
    # 缓存，避免重复读文件
    if name not in self.cache:
      path = f"{self.prompt_dir}/{name}.txt"
      with open(path, "r", encoding="utf-8") as f:
        self.cache[name] = f.read()
    # 用传入的变量填充占位符
    return self.cache[name].format(**kwargs)
