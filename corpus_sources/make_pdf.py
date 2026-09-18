"""Render everyday_knowledge.md to the PDF that goes into corpus/, then verify extraction.

One corpus file is delivered as a PDF on purpose, so the notebook's PDF import path is
actually exercised instead of being described. This Mac has no Chrome, poppler, pandoc or
LaTeX, so the PDF is produced with WebKit through `render.swift` (1 CSS px = 1 pt, fixed
612x792 px `.page` blocks).

Two things went wrong on the first attempt and are worth recording:

1. A per-page `<h2>` header extracted as text and became six junk training passages
   ("everyday knowledge — teaching sentences ( page 1 of 6 )"). The header is gone now.
2. At 46 lines per page the list overflowed the fixed 792 px page height and
   `overflow:hidden` clipped it, so **5 teaching sentences silently disappeared** from the
   PDF. Nothing warned about this: the pages all contained text, so the notebook's own
   "no text extracted" warning would never have fired. 34 lines per page fits (34 x 19 px
   = 646 px of the 680 px available) and the round-trip check below is now exact.

    python corpus_sources/make_pdf.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from make_corpus import normalized_passages  # noqa: E402

SOURCE = HERE / "everyday_knowledge.md"
HTML = HERE / "everyday_knowledge.html"
PDF = ROOT / "corpus" / "everyday_knowledge.pdf"
LINES_PER_PAGE = 34

STYLE = """<!doctype html><meta charset="utf-8"><title>Everyday knowledge</title>
<style>
 html,body{margin:0;padding:0;background:#fff;font-family:Georgia,serif;color:#111}
 .page{width:612px;height:792px;box-sizing:border-box;padding:56px 60px;overflow:hidden}
 ul{margin:0;padding:0;list-style:none}
 li{font-size:13px;line-height:19px;margin:0}
</style>"""


def main() -> None:
    lines = [line for line in SOURCE.read_text(encoding="utf-8").split("\n") if line.strip()]
    pages = [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    HTML.write_text(STYLE + "".join(
        '<div class="page"><ul>%s</ul></div>' % "".join(f"<li>{line}</li>" for line in page)
        for page in pages) + "<script>window.__report=()=>document.querySelectorAll('.page').length</script>")
    subprocess.run(["swift", str(HERE / "render.swift"), str(HTML), str(PDF), str(len(pages))],
                   cwd=HERE, check=True)
    for preview in HERE.glob("preview-*.png"):
        preview.unlink()

    from pypdf import PdfReader
    reader = PdfReader(PDF)
    page_text = [page.extract_text() or "" for page in reader.pages]
    extracted = normalized_passages("\n".join(page_text))
    source = normalized_passages(SOURCE.read_text(encoding="utf-8"))
    check = {"pdf_pages": len(reader.pages), "encrypted": reader.is_encrypted,
             "lines_per_page": LINES_PER_PAGE,
             "pages_with_no_text": [n + 1 for n, t in enumerate(page_text) if not t.strip()],
             "source_passages": len(source), "extracted_passages": len(extracted),
             "extracted_not_in_source": sorted(set(extracted) - set(source)),
             "source_not_in_extract": sorted(set(source) - set(extracted)),
             "reading_order_identical": extracted == source}
    (HERE / "pdf_extraction_check.json").write_text(json.dumps(check, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(check, indent=2, ensure_ascii=False))
    if not check["reading_order_identical"]:
        raise SystemExit("PDF extraction does not round-trip; fix the layout before training.")


if __name__ == "__main__":
    main()
