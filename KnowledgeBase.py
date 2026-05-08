import json
import logging
import os
from collections import OrderedDict
from threading import Lock
from time import time
from typing import Any, Optional

import chromadb
import torch
from Constant import (
  get_embedding_device,
  get_embedding_model_path,
  get_kb_query_cache_max_size,
  get_kb_query_cache_ttl_seconds,
)
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_REPO_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOKENIZER_REQUIRED_FILES = ("tokenizer_config.json",)
TOKENIZER_ANY_OF_FILES = (
  "tokenizer.json",
  "sentencepiece.bpe.model",
  "spiece.model",
  "vocab.txt",
  "merges.txt",
  "unigram.json",
)


def _has_local_tokenizer(model_dir: str) -> bool:
  if not os.path.isdir(model_dir):
    return False

  for required in TOKENIZER_REQUIRED_FILES:
    if not os.path.isfile(os.path.join(model_dir, required)):
      return False

  return any(
    os.path.isfile(os.path.join(model_dir, name))
    for name in TOKENIZER_ANY_OF_FILES
  )


def _resolve_local_model_dir(model_name_or_path: str) -> tuple[str, str]:
  model_name_or_path = (model_name_or_path or "").strip()
  project_root = os.path.dirname(__file__)

  if "/" in model_name_or_path and not os.path.isabs(model_name_or_path) and "\\" not in model_name_or_path:
    repo_id = model_name_or_path
    model_dir = os.path.join(project_root, "models", repo_id.split("/")[-1])
    return os.path.abspath(model_dir), repo_id

  repo_id = os.getenv("PLANK_EMBEDDING_MODEL_REPO", DEFAULT_EMBEDDING_REPO_ID).strip() or DEFAULT_EMBEDDING_REPO_ID
  if not model_name_or_path:
    model_name_or_path = os.path.join(project_root, "models", "paraphrase-multilingual-MiniLM-L12-v2")

  return os.path.abspath(model_name_or_path), repo_id


def _ensure_local_tokenizer(model_name_or_path: str) -> str:
  local_model_dir, repo_id = _resolve_local_model_dir(model_name_or_path)

  if _has_local_tokenizer(local_model_dir):
    logger.info("Using local tokenizer at: %s", local_model_dir)
    return local_model_dir

  os.makedirs(local_model_dir, exist_ok=True)
  logger.warning(
    "Tokenizer files not found at %s. Downloading from %s ...",
    local_model_dir,
    repo_id,
  )
  snapshot_download(
    repo_id=repo_id,
    local_dir=local_model_dir,
    local_dir_use_symlinks=False,
  )

  if not _has_local_tokenizer(local_model_dir):
    raise RuntimeError(f"Tokenizer download incomplete at {local_model_dir}")

  logger.info("Tokenizer downloaded and loaded from local path: %s", local_model_dir)
  return local_model_dir


class KnowledgeBase:
  _shared_embedder: Optional[SentenceTransformer] = None
  _shared_embedder_lock = Lock()
  _shared_embedder_path: Optional[str] = None
  _shared_embedder_device: Optional[str] = None
  _shared_clients: dict[str, chromadb.PersistentClient] = {}
  _shared_collections: dict[tuple[str, str], Any] = {}
  _shared_storage_lock = Lock()

  @classmethod
  def _get_shared_embedder(cls, embedding_model_path: str, device: str) -> SentenceTransformer:
    if (
      cls._shared_embedder is not None
      and cls._shared_embedder_path == embedding_model_path
      and cls._shared_embedder_device == device
    ):
      return cls._shared_embedder

    with cls._shared_embedder_lock:
      if (
        cls._shared_embedder is None
        or cls._shared_embedder_path != embedding_model_path
        or cls._shared_embedder_device != device
      ):
        cls._shared_embedder = SentenceTransformer(
          embedding_model_path,
          device=device,
          local_files_only=True,
        )
        cls._shared_embedder_path = embedding_model_path
        cls._shared_embedder_device = device
    return cls._shared_embedder

  @classmethod
  def _get_shared_collection(cls, db_path: str, collection_name: str):
    normalized_path = os.path.abspath(db_path)
    key = (normalized_path, collection_name)
    with cls._shared_storage_lock:
      client = cls._shared_clients.get(normalized_path)
      if client is None:
        client = chromadb.PersistentClient(path=normalized_path)
        cls._shared_clients[normalized_path] = client

      collection = cls._shared_collections.get(key)
      if collection is None:
        collection = client.get_or_create_collection(collection_name)
        cls._shared_collections[key] = collection
      return client, collection

  def __init__(self, db_path: str = "./chroma_db", collection_name: str = "plankbevelen"):
    self.client, self.collection = self._get_shared_collection(db_path, collection_name)
    self.device = get_embedding_device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model_path = _ensure_local_tokenizer(get_embedding_model_path())
    self.embedder = self._get_shared_embedder(embedding_model_path, self.device)
    self.query_embedding_cache: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
    self.query_embedding_cache_max_size = max(1, get_kb_query_cache_max_size())
    self.query_embedding_cache_ttl_seconds = max(1, get_kb_query_cache_ttl_seconds())

  def _encode(self, text: str) -> list[float]:
    return self.embedder.encode(
      text,
      show_progress_bar=False,
      convert_to_numpy=True,
    ).tolist()

  def _get_cached_query_embedding(self, query: str) -> list[float]:
    now = time()
    cached = self.query_embedding_cache.get(query)
    if cached is not None:
      cached_at, embedding = cached
      if now - cached_at <= self.query_embedding_cache_ttl_seconds:
        self.query_embedding_cache.move_to_end(query)
        return embedding
      self.query_embedding_cache.pop(query, None)

    embedding = self._encode(query)
    self.query_embedding_cache[query] = (now, embedding)
    self.query_embedding_cache.move_to_end(query)

    while len(self.query_embedding_cache) > self.query_embedding_cache_max_size:
      self.query_embedding_cache.popitem(last=False)
    return embedding

  def _existing_doc(self, doc_id: str):
    result = self.collection.get(
      ids=[doc_id],
      include=["documents", "metadatas"],
    )
    if not result.get("ids"):
      return None, None
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return (
      documents[0] if documents else None,
      metadatas[0] if metadatas else None,
    )

  def _normalize_metadata(self, metadata: dict) -> dict:
    normalized = {}
    for key, value in metadata.items():
      if value is None:
        continue
      if isinstance(value, (str, int, float, bool)):
        normalized[key] = value
      else:
        normalized[key] = json.dumps(value, ensure_ascii=False)
    return normalized

  def add(self, doc_id: str, text: str, metadata: Optional[dict] = None, force: bool = False) -> bool:
    metadata = self._normalize_metadata(metadata or {})
    if not force:
      existing_text, existing_metadata = self._existing_doc(doc_id)
      if existing_text == text and (existing_metadata or {}) == metadata:
        return False

    embedding = self._encode(text)
    self.collection.upsert(
      ids=[doc_id],
      embeddings=[embedding],
      documents=[text],
      metadatas=[metadata],
    )
    return True

  def search(self, query: str, top_k: int = 3, threshold: float = 0.5) -> list[str]:
    records = self.search_with_meta(query=query, top_k=top_k, threshold=threshold)
    return [item.get("text", "") for item in records if item.get("text")]

  def search_with_meta(
    self,
    query: str,
    top_k: int = 3,
    threshold: float = 0.5,
    where: Optional[dict[str, Any]] = None,
  ) -> list[dict[str, Any]]:
    query_embedding = self._get_cached_query_embedding(query)

    kwargs = {
      "query_embeddings": [query_embedding],
      "n_results": top_k,
      "include": ["documents", "distances", "metadatas"],
    }
    if where:
      kwargs["where"] = where

    results = self.collection.query(**kwargs)
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    records = []
    for idx, doc, dist, meta in zip(ids, docs, distances, metas):
      if dist >= threshold:
        continue
      records.append(
        {
          "id": idx,
          "text": doc,
          "distance": float(dist),
          "score": 1.0 - float(dist),
          "metadata": meta or {},
        }
      )
    return records
