import os

from dotenv import load_dotenv

load_dotenv(override=False)

DEFAULT_LLM_API_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_LLM_MODEL = "deepseek-v3-2-251201"

TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def get_env(name: str, default: str = "") -> str:
  value = os.getenv(name)
  if value is None:
    return default
  value = value.strip()
  return value if value else default


def get_env_bool(name: str, default: bool = False) -> bool:
  value = os.getenv(name)
  if value is None:
    return default
  return value.strip().lower() in TRUE_VALUES


def get_first_env(names: list[str], default: str = "") -> str:
  for name in names:
    value = os.getenv(name)
    if value is None:
      continue
    value = value.strip()
    if value:
      return value
  return default


def get_llm_api_base_url() -> str:
  return get_first_env(
    ["PLANK_LLM_API_BASE_URL", "ARK_BASE_URL", "OPENAI_BASE_URL"],
    DEFAULT_LLM_API_BASE_URL,
  )


def get_llm_api_key() -> str:
  return get_first_env(["PLANK_LLM_API_KEY", "OPENAI_API_KEY"], "")


def get_llm_model() -> str:
  return get_first_env(["PLANK_LLM_MODEL", "ARK_MODEL", "OPENAI_MODEL"], DEFAULT_LLM_MODEL)


def get_embedding_model_path() -> str:
  default = os.path.join(os.path.dirname(__file__), "models", "paraphrase-multilingual-MiniLM-L12-v2")
  return get_env("PLANK_EMBEDDING_MODEL_PATH", default)


def get_embedding_device(default_device: str) -> str:
  return get_env("PLANK_EMBEDDING_DEVICE", default_device)


def get_serpapi_key() -> str:
  return get_env("SERPAPI_KEY", "")

def get_react_max_steps() -> int:
  return int(get_env("PLANK_REACT_MAX_STEPS", "5"))
  # 防止死循环
