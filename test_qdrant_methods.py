from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
methods = [m for m in dir(client) if not m.startswith("_")]
print("Methods:", methods)
