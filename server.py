#!/usr/bin/env python3
"""FastAPI server — upload an image, get OCR results as JSON."""

import os
import tempfile
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


@app.post("/api/ocr")
async def ocr_endpoint(
    image: UploadFile = File(...),
    threshold: float = Query(0.8),
    lang: str = Query("en"),
    engine: str = Query("paddle"),
):
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        results = run_ocr(tmp_path, confidence_threshold=threshold, lang=lang, engine=engine)
        return JSONResponse(content={"results": results, "count": len(results), "engine": engine})
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
