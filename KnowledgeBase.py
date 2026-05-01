import chromadb
from sentence_transformers import SentenceTransformer

# 一个基于ChromaDB的知识库工具，支持文本存储和语义搜索。
# RAG（Retrieval-Augmented Generation）检索增强生成
# 核心就是：
# 1. 把问题转换为向量，
# 2. 搜索向量知识库中最相关的文本，
# 3. 把找到的内容塞到prompt里让LLM生成答案。

# 这里构成为：向量数据库、Embedding模型、检索逻辑
class KnowledgeBase:
  def __init__(self, db_path="./chroma_db"):
    self.client = chromadb.PersistentClient(path=db_path) # 创建持久化客户端，数据存储在本地文件系统
    self.collection = self.client.get_or_create_collection("plankbevelen") # 创建或获取集合
    self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

  def add(self, doc_id: str, text: str, metadata: dict = {}):
    # 将文本转换为向量并存储
    embedding = self.embedder.encode(text).tolist()
    self.collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata]
    )

  def search(self, query: str, top_k: int = 3, threshold: float = 0.5) -> list[str]:
    # 将查询转换为向量并搜索最相关的文本
    query_embedding = self.embedder.encode(query).tolist()
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )
    # distance 越小越相关（0=完全一致），超过阈值的视为不相关直接过滤掉
    docs = results["documents"][0]
    distances = results["distances"][0]
    filtered = [doc for doc, dist in zip(docs, distances) if dist < threshold]
    return filtered