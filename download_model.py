import os

from constant import get_local_model_path, get_local_model_repo
from huggingface_hub import snapshot_download


def main():
  repo_id = get_local_model_repo()
  default_local_path = os.path.join(os.path.dirname(__file__), "models", "Qwen2.5-1.5B-Instruct")
  local_path = get_local_model_path(default_local_path)

  os.makedirs(local_path, exist_ok=True)
  print(f"Downloading {repo_id} -> {local_path}")
  snapshot_download(
    repo_id=repo_id,
    local_dir=local_path,
  )
  print("Done.")


if __name__ == "__main__":
  main()
