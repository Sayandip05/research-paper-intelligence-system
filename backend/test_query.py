"""Test query endpoint for unified search"""
import requests

response = requests.post(
    'http://localhost:8000/api/query',
    json={
        'question': 'What is LoRA? Show me architecture diagrams.',
        'similarity_top_k': 5
    },
    timeout=60
)

data = response.json()

print("=" * 60)
print("UNIFIED SEARCH TEST")
print("=" * 60)
print(f"\nQuestion: {data.get('question')}")
print(f"\nAnswer ({len(data.get('answer', ''))} chars):")
print("-" * 40)
print(data.get('answer', 'No answer')[:500] + "...")

print(f"\n📄 Text Sources: {len(data.get('sources', []))}")
for i, src in enumerate(data.get('sources', []), 1):
    print(f"  {i}. {src.get('paper_title', 'Unknown')[:40]} (score: {src.get('score', 0):.2f})")

print(f"\n🖼️ Related Images: {len(data.get('images', []))}")
for i, img in enumerate(data.get('images', []), 1):
    print(f"  {i}. {img.get('paper_title', 'Unknown')[:40]}")
    print(f"     Page: {img.get('page_number')}, Type: {img.get('image_type')}, Score: {img.get('score', 0):.2f}")

print("\n" + "=" * 60)
print(f"✅ RESULT: Text Sources={len(data.get('sources', []))}, Images={len(data.get('images', []))}")
