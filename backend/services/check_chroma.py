from collections import Counter
from backend.rag import _get_store

col = _get_store()._collection
print("total chunks:", col.count())
meta = col.get(include=["metadatas"])["metadatas"]
print("chunks per thread:", Counter(m.get("thread_id") for m in meta))
print("chunks with a user_id:", sum(1 for m in meta if m.get("user_id")))
print("sample:", meta[:2])