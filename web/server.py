"""FastAPI server for doc2text web UI."""

import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import AsyncGenerator

# Add project root to path so engine/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="doc2text", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WEB_DIR = Path(__file__).parent
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="doc2text_"))

app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (WEB_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/ocr/image")
async def ocr_image(file: UploadFile = File(...), lang: str = "ben"):
    """OCR a single image file. Returns JSON result. lang: ben, eng, ben+eng"""
    import traceback
    from engine.ocr import ocr_image as _ocr_image

    ext = Path(file.filename or "upload.png").suffix or ".png"
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    content = await file.read()
    tmp_path.write_bytes(content)

    # Normalize lang param
    lang_map = {"ben": "ben", "eng": "eng", "ben+eng": "ben+eng", "bn": "ben", "en": "eng"}
    tesseract_lang = lang_map.get(lang, "ben")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _ocr_image, str(tmp_path), tesseract_lang)
        return {
            "filename": file.filename,
            "text": result.text,
            "word_count": result.word_count,
            "line_count": result.line_count,
            "corrections": result.corrections,
        }
    except Exception as e:
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/ocr/pdf/stream")
async def ocr_pdf_stream(file: UploadFile = File(...), lang: str = "ben") -> StreamingResponse:
    """OCR a PDF page-by-page with SSE progress streaming. lang: ben, eng, ben+eng"""
    from engine.ocr import ocr_pdf_pages, pdf_page_count

    lang_map = {"ben": "ben", "eng": "eng", "ben+eng": "ben+eng", "bn": "ben", "en": "eng"}
    tesseract_lang = lang_map.get(lang, "ben")

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
    content = await file.read()
    tmp_path.write_bytes(content)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            loop = asyncio.get_event_loop()
            total = await loop.run_in_executor(None, pdf_page_count, str(tmp_path))
            yield f"data: {json.dumps({'type': 'start', 'total_pages': total, 'filename': file.filename})}\n\n"

            gen = ocr_pdf_pages(str(tmp_path), lang=tesseract_lang)
            for page_num, result in gen:
                page_data = {
                    "type": "page",
                    "page": page_num + 1,
                    "total_pages": total,
                    "text": result.text,
                    "word_count": result.word_count,
                    "line_count": result.line_count,
                    "corrections": result.corrections,
                }
                yield f"data: {json.dumps(page_data)}\n\n"
                await asyncio.sleep(0)  # yield to event loop

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
