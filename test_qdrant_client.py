from qdrant_client import QdrantClient
print("Imported QdrantClient")
client = QdrantClient(host="localhost", port=6333)
print(f"Client type: {type(client)}")
print(f"Has search? {hasattr(client, 'search')}")
print(f"Dir: {dir(client)}")
