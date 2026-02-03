"""Find and optionally delete duplicate papers"""
from qdrant_client import QdrantClient
from collections import defaultdict

c = QdrantClient(host='localhost', port=6333)

# Get all points
result = c.scroll('research_papers_hybrid', limit=200, with_payload=True)
points = result[0]

print(f"Total chunks: {len(points)}")

# Group by paper title (normalized)
by_title = defaultdict(list)
for p in points:
    title = p.payload.get('paper_title', 'Unknown')[:50]
    by_title[title].append({
        'id': p.id,
        'paper_id': p.payload.get('paper_id', '?')
    })

print("\nPapers and their paper_ids:")
for title, chunks in by_title.items():
    paper_ids = set(c['paper_id'] for c in chunks)
    print(f"\n{title}:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Unique paper_ids: {len(paper_ids)}")
    for pid in paper_ids:
        count = sum(1 for c in chunks if c['paper_id'] == pid)
        print(f"    {pid[:20]}... : {count} chunks")
