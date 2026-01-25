from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
print(f"Module: {client.__module__}")
print(f"Class: {client.__class__.__name__}")
print(f"Has search: {hasattr(client, 'search')}")
print(f"Has query_points: {hasattr(client, 'query_points')}")
print(f"Has recommend: {hasattr(client, 'recommend')}")
