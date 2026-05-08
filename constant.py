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


def get_context_max_history_turns() -> int:
  return int(get_env("PLANK_CONTEXT_MAX_HISTORY_TURNS", "4"))


def get_context_max_chars() -> int:
  return int(get_env("PLANK_CONTEXT_MAX_CHARS", "5000"))


def get_context_max_kb_items() -> int:
  return int(get_env("PLANK_CONTEXT_MAX_KB_ITEMS", "3"))


def get_context_max_memory_items() -> int:
  return int(get_env("PLANK_CONTEXT_MAX_MEMORY_ITEMS", "4"))


def get_memory_collection_name() -> str:
  return get_env("PLANK_MEMORY_COLLECTION", "agent_memory")


def get_memory_db_path() -> str:
  return get_env("PLANK_MEMORY_DB_PATH", "./chroma_db")


def get_memory_threshold() -> float:
  return float(get_env("PLANK_MEMORY_THRESHOLD", "0.65"))


def get_memory_top_k() -> int:
  return int(get_env("PLANK_MEMORY_TOP_K", "6"))


def get_memory_write_enabled() -> bool:
  return get_env_bool("PLANK_MEMORY_WRITE_ENABLED", True)


def get_kb_query_cache_max_size() -> int:
  return int(get_env("PLANK_KB_QUERY_CACHE_MAX_SIZE", "1024"))


def get_kb_query_cache_ttl_seconds() -> int:
  return int(get_env("PLANK_KB_QUERY_CACHE_TTL_SECONDS", "3600"))
