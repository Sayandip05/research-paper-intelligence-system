"""Verify section distribution in Qdrant"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from qdrant_client import QdrantClient
from collections import Counter

client = QdrantClient(host="localhost", port=6333)

# Scroll through all points
sections = []
offset = None

while True:
    result = client.scroll(
        collection_name="research_papers",
        limit=100,
        offset=offset,
        with_payload=True
    )
    
    points, offset = result
    if not points:
        break
    
    for point in points:
        section = point.payload.get("section_title", "MISSING")
        sections.append(section)
    
    if offset is None:
        break

# Count sections
print("\n" + "="*60)
print("  SECTION DISTRIBUTION IN QDRANT")
print("="*60)

counter = Counter(sections)
for section, count in sorted(counter.items(), key=lambda x: -x[1]):
    print(f"  {section}: {count} chunks")

print(f"\n  Total: {len(sections)} chunks")
print("="*60)

# Check for invalid sections
VALID_SECTIONS = {
    'Abstract', 'Introduction', 'Related Work', 'Methods', 'Experiments',
    'Results', 'Discussion', 'Limitations', 'Future Work', 'Conclusion',
    'References', 'Appendix', 'Unknown'
}

invalid = [s for s in sections if s not in VALID_SECTIONS]
if invalid:
    print(f"\n⚠️  INVALID SECTIONS FOUND: {set(invalid)}")
else:
    print(f"\n✅ All sections are valid canonical names!")
