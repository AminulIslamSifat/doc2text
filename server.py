#!/usr/bin/env python3
"""FastAPI server — upload an image or PDF, get OCR results as JSON."""

import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ["FLAGS_eager_delete_tensor_gb"] = "0.0"
os.environ["FLAGS_fast_eager_deletion_mode"] = "True"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
os.environ.setdefault("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "10")

import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse

from ocr import run_ocr

app = FastAPI(title="doc2text")
STATIC_DIR = Path(__file__).parent / "static"

# Cap parallel OCR workers to avoid memory spikes
OCR_POOL = ThreadPoolExecutor(max_workers=4)


def _is_pdf(filename: str | None, content_type: str | None) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return False


def _pdf_pages_to_images(pdf_path: str, dpi: int = 200) -> list[tuple[int, str]]:
    """Convert each PDF page to a temp PNG. Returns [(page_number, temp_path), ...]."""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pages = []
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pix.save(tmp.name)
        tmp.close()
        pages.append((i + 1, tmp.name))
    doc.close()
    return pages


def _ocr_page(page_num: int, img_path: str, threshold: float, lang: str, engine: str) -> dict:
    """Run OCR on a single page image. Returns {page, results, count}."""
    try:
        results = run_ocr(img_path, confidence_threshold=threshold, lang=lang, engine=engine)
        return {"page": page_num, "results": results, "count": len(results)}
    except Exception as e:
        return {"page": page_num, "results": [], "count": 0, "error": str(e)}


@app.post("/api/ocr")
async def ocr_endpoint(
    image: UploadFile = File(...),
    threshold: float = Query(0.8),
    lang: str = Query("en"),
    engine: str = Query("paddle"),
):
    filename = image.filename or "upload"
    content_type = image.content_type or ""
    is_pdf = _is_pdf(filename, content_type)
    suffix = ".pdf" if is_pdf else (Path(filename).suffix or ".png")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if is_pdf:
            loop = asyncio.get_running_loop()
            pages = await loop.run_in_executor(OCR_POOL, _pdf_pages_to_images, tmp_path)

            tasks = [
                loop.run_in_executor(
                    OCR_POOL, _ocr_page, pg_num, pg_path, threshold, lang, engine
                )
                for pg_num, pg_path in pages
            ]
            page_results = await asyncio.gather(*tasks)

            # Clean up page temp files
            for _, pg_path in pages:
                Path(pg_path).unlink(missing_ok=True)

            # Sort by page number
            page_results.sort(key=lambda p: p["page"])
            total = sum(p["count"] for p in page_results)
            return JSONResponse(content={
                "pages": page_results,
                "total_count": total,
                "page_count": len(page_results),
                "engine": engine,
                "type": "pdf",
            })
        else:
            results = run_ocr(tmp_path, confidence_threshold=threshold, lang=lang, engine=engine)
            return JSONResponse(content={
                "results": results,
                "count": len(results),
                "engine": engine,
                "type": "image",
            })
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8765"))
    print(f"http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
