import string

class PromptLoader:
    def __init__(self, prompt_dir: str = './prompts'):
        self.prompt_dir = prompt_dir
        self.cache = {}

    def load(self, name: str, **kwargs) -> str:
        if name not in self.cache:
            with open(f"{self.prompt_dir}/{name}.txt", encoding="utf-8") as f:
                self.cache[name] = f.read()
        # 用 safe_substitute 代替 format()，避免模板里多余的 { 导致崩溃
        return string.Template(
            self.cache[name].replace("{", "${").replace("${$", "${")
        ).safe_substitute(**kwargs)
        # 更简单的方案：直接用 str.replace 逐个替换，不依赖 Python format 语法