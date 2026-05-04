import importlib.util
import logging
import os
from threading import Thread

from constant import (
  get_auto_download_local_model,
  get_llm_device,
  get_llm_quantization,
  get_local_model_path,
  get_local_model_repo,
)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.utils import logging as transformers_logging

try:
  from huggingface_hub import snapshot_download
except ImportError:
  snapshot_download = None

logging.getLogger("transformers").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()


class LLM:
  def __init__(self):
    self.device = get_llm_device()
    if self.device == "auto":
      self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.quantization = get_llm_quantization()
    self.local_repo_id = get_local_model_repo()
    default_local_path = os.path.join(os.path.dirname(__file__), "models", "Qwen2.5-1.5B-Instruct")
    self.local_model_path = get_local_model_path(default_local_path)
    self.auto_download_local = get_auto_download_local_model()

    self._ensure_local_model()
    self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
    self.model = self._load_model(self.local_model_path)

  def _ensure_local_model(self):
    if os.path.exists(os.path.join(self.local_model_path, "config.json")):
      return

    if not self.auto_download_local:
      raise FileNotFoundError(
        f"Local model not found at '{self.local_model_path}'. "
        "Enable PLANK_AUTO_DOWNLOAD_LOCAL_MODEL=true or run download_model.py first."
      )

    if snapshot_download is None:
      raise RuntimeError("Please install 'huggingface_hub' for automatic local model download.")

    os.makedirs(self.local_model_path, exist_ok=True)
    print(f"Downloading model {self.local_repo_id} to {self.local_model_path} ...")
    snapshot_download(
      repo_id=self.local_repo_id,
      local_dir=self.local_model_path,
    )
    print("Model download completed.")

  def _load_model(self, model_path: str):
    use_cuda = self.device == "cuda" and torch.cuda.is_available()
    base_kwargs = {
      "torch_dtype": torch.float16 if use_cuda else torch.float32,
      "trust_remote_code": True,
    }
    if importlib.util.find_spec("accelerate") is not None:
      base_kwargs["low_cpu_mem_usage"] = True

    quantized_kwargs = self._quantization_kwargs(use_cuda)
    if quantized_kwargs:
      try:
        model = AutoModelForCausalLM.from_pretrained(
          model_path,
          **base_kwargs,
          **quantized_kwargs,
        )
        model.eval()
        return model
      except Exception as e:
        print(f"Quantized load failed, fallback to normal load: {e}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **base_kwargs).to(self.device)
    model.eval()
    return model

  def _quantization_kwargs(self, use_cuda: bool) -> dict:
    if self.quantization in ("none", "", "false", "0"):
      return {}
    if self.quantization not in ("4bit", "8bit"):
      print(f"Unknown PLANK_LLM_QUANTIZATION={self.quantization}, ignored.")
      return {}
    if not use_cuda:
      print("bitsandbytes quantization requires CUDA. Fallback to normal load.")
      return {}
    if importlib.util.find_spec("bitsandbytes") is None or importlib.util.find_spec("accelerate") is None:
      print("Quantization requires bitsandbytes + accelerate. Fallback to normal load.")
      return {}

    from transformers import BitsAndBytesConfig

    if self.quantization == "4bit":
      quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
      )
    else:
      quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    return {
      "device_map": "auto",
      "quantization_config": quantization_config,
    }

  def think(self, messages: list[dict[str, str]], temperature: float = 0.7, max_new_tokens: int = 512) -> str:
    input_text = self.tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
    )
    inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

    streamer = TextIteratorStreamer(
      self.tokenizer,
      skip_prompt=True,
      skip_special_tokens=True,
    )
    generate_kwargs = dict(
      **inputs,
      streamer=streamer,
      max_new_tokens=max_new_tokens,
      do_sample=temperature > 0,
    )
    if temperature > 0:
      generate_kwargs["temperature"] = temperature

    def generate():
      with torch.inference_mode():
        self.model.generate(**generate_kwargs)

    thread = Thread(target=generate)
    thread.start()

    collected = []
    print("Assistant: ", end="", flush=True)
    for token_text in streamer:
      print(token_text, end="", flush=True)
      collected.append(token_text)
    print()

    thread.join()
    return "".join(collected)


if __name__ == "__main__":
  llm = LLM()
  messages = [
    {"role": "system", "content": "You are my assistant."},
    {"role": "user", "content": "Please introduce yourself."},
  ]
  response = llm.think(messages)
  print("LLM Response:", response)
