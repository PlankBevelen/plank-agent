import os

from dotenv import load_dotenv

load_dotenv(override=False)

DEFAULT_LLM_DEVICE = "auto"
DEFAULT_LLM_QUANTIZATION = "none"
DEFAULT_LOCAL_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct"

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


def get_llm_device() -> str:
  return get_env("PLANK_LLM_DEVICE", DEFAULT_LLM_DEVICE).lower()


def get_embedding_model_path() -> str:
  default = os.path.join(os.path.dirname(__file__), "models", "paraphrase-multilingual-MiniLM-L12-v2")
  return get_env("PLANK_EMBEDDING_MODEL_PATH", default)

def get_llm_quantization() -> str:
  return get_env("PLANK_LLM_QUANTIZATION", DEFAULT_LLM_QUANTIZATION).lower()


def get_local_model_repo() -> str:
  return get_env("PLANK_LOCAL_MODEL_REPO", DEFAULT_LOCAL_MODEL_REPO)


def get_local_model_path(default_path: str) -> str:
  return get_env("PLANK_LOCAL_MODEL_PATH", default_path)


def get_auto_download_local_model() -> bool:
  return get_env_bool("PLANK_AUTO_DOWNLOAD_LOCAL_MODEL", True)


def get_embedding_device(default_device: str) -> str:
  return get_env("PLANK_EMBEDDING_DEVICE", default_device)


def get_serpapi_key() -> str:
  return get_env("SERPAPI_KEY", "")

def get_react_max_steps() -> int:
  return int(get_env("PLANK_REACT_MAX_STEPS", "5"))
  # 防止死循环