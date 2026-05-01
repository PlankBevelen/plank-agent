
from typing import Dict, Any

class ToolExecutor:
  def __init__(self):
    self.tools: Dict[str, Dict[str, Any]] = {}
  
  def registerTool(self, name: str, description: str, func: callable):
    self.tools[name] = {"description": description, "func": func}
  
  def runTool(self, name: str, param: str) -> str:
    if name in self.tools:
      func = self.tools[name]["func"]
      return func(param)
    else:
      return f"工具 '{name}' 不存在。"
  

