import json
import logging
import os
from typing import Optional

import chromadb
import torch
from Constant import get_embedding_device, get_embedding_model_path
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

class KnowledgeBase:
  def __init__(self, db_path: str = "./chroma_db"):
    self.client = chromadb.PersistentClient(path=db_path)
    self.collection = self.client.get_or_create_collection("plankbevelen")
    self.device = get_embedding_device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model_path = get_embedding_model_path()
    self.embedder = SentenceTransformer(
      embedding_model_path,  
      device=self.device,
    )
    self.query_embedding_cache: dict[str, list[float]] = {}

  def _encode(self, text: str) -> list[float]:
    return self.embedder.encode(
      text,
      show_progress_bar=False,
      convert_to_numpy=True,
    ).tolist()

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
    if query in self.query_embedding_cache:
      query_embedding = self.query_embedding_cache[query]
    else:
      query_embedding = self._encode(query)
      self.query_embedding_cache[query] = query_embedding

    results = self.collection.query(
      query_embeddings=[query_embedding],
      n_results=top_k,
      include=["documents", "distances"],
    )
    docs = results["documents"][0] if results.get("documents") else []
    distances = results["distances"][0] if results.get("distances") else []
    filtered = [doc for doc, dist in zip(docs, distances) if dist < threshold]
    return filtered

