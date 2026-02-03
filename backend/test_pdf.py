"""Test PDF parsing"""
import fitz

doc = fitz.open('corpus/OpenAGI.pdf')
print(f"Total pages: {len(doc)}")

total_text = 0
total_images = 0

for i in range(len(doc)):
    page = doc[i]
    txt = page.get_text()
    imgs = len(page.get_images())
    total_text += len(txt)
    total_images += imgs
    if len(txt) > 0 or imgs > 0:
        print(f"Page {i}: text={len(txt)} chars, images={imgs}")

print(f"\nTotal: {total_text} chars, {total_images} images")

if total_text > 0:
    print("\nFirst 500 chars of all text:")
    all_text = ""
    for page in doc:
        all_text += page.get_text()
    print(all_text[:500])
