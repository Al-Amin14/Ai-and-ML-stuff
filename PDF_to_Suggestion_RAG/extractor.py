"""
extractor.py
─────────────────────────────────────────────────────────────────
Extracts raw text from every page of a PDF using PyMuPDF (fitz).
Returns a list of { page, text } dicts so we preserve page numbers
for source citations later.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
import json


def extract_pages(pdf_path: str) -> List[Dict]:
    """
    Extract text page-by-page from a PDF.

    Returns:
        List of dicts:  [{ "page": 1, "text": "..." }, ...]
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    doc = fitz.open(str(path))

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        # Skip blank pages
        if not text:
            continue

        pages.append({
            "page":  page_num + 1,   # 1-indexed for human display
            "text":  text,
        })

    doc.close()
    return pages

# data=extract_pages("./pdf/T1.2.pdf")
# stor_dic=[]
# for page in data:
#     page["text"] = page["text"].replace("\n", " ")
#     stor_dic.append(page)

# with open("data.json","w") as f:
#     json.dump(stor_dic,f)





def extract_full_text(pdf_path: str) -> str:
    """Return all pages joined as one string (used for quick summaries)."""
    data = extract_pages(pdf_path)
    stor_dic=[]
    for page in data:
        page["text"] = page["text"].replace("\n", " ")
        stor_dic.append(page)
    # return "\n\n".join(p["text"] for p in pages)
    return stor_dic


'''
data=extract_full_text("./pdf/T1.2.pdf")
print(data)

with open('data.txt','w',encoding='Utf-8') as f:
    f.write(data)

'''
'''

'''
def get_page_count(pdf_path: str) -> int:
    """Return total page count of PDF."""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


# storcount=get_page_count('./pdf/T1.2.pdf')
# print(storcount)                          