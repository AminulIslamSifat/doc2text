# doc2text

Bangla OCR engine with web UI for converting images and PDFs to text.

## Features

- **Dual-pass OCR** — grayscale + adaptive threshold, merged with confidence weighting
- **SymSpell correction** — 439K word dictionary + 46K curated set, frequency-aware suggestions
- **Suffix-aware** — strips Bangla inflectional suffixes before dictionary lookup
- **PDF support** — page-by-page processing with real-time progress streaming
- **Batch upload** — multiple files at once, each processed independently
- **Save as .txt** — download extracted text per file

## Stack

| Component | Technology |
|-----------|------------|
| OCR Engine | Tesseract (`ben`) |
| Preprocessing | OpenCV (upscale, denoise, adaptive threshold) |
| Spell Correction | SymSpell + curated Bangla dictionary |
| Unicode Normalization | bnunicodenormalizer |
| PDF Rendering | PyMuPDF |
| Web Backend | FastAPI + Uvicorn |
| Web Frontend | Vanilla HTML/CSS/JS |

## Setup

```bash
# Install dependencies
uv sync

# Start web server
uv run python web/server.py
```

Open `http://localhost:8765` in your browser.

## Project Structure

```
doc2text/
├── engine/
│   ├── ocr.py          # Core OCR pipeline
│   ├── corrector.py    # SymSpell-based Bangla spell correction
│   └── data/
│       ├── bangla_words.txt      # 439K word dictionary
│       └── bangla_curated.txt    # 46K curated high-confidence words
├── web/
│   ├── server.py       # FastAPI server
│   └── static/
│       └── index.html  # Web UI
├── tests/
│   └── assets/         # Test images
└── pyproject.toml
```

## API

### `POST /api/ocr/image`

Upload a single image (PNG/JPG/JPEG). Returns JSON:

```json
{
  "filename": "page.png",
  "text": "extracted text...",
  "word_count": 244,
  "line_count": 20,
  "corrections": [{"original": "ভ্রুদ্ধ", "corrected": "ক্রুদ্ধ", "distance": 1}]
}
```

### `POST /api/ocr/pdf/stream`

Upload a PDF. Returns SSE stream with per-page progress:

```
data: {"type": "start", "total_pages": 5, "filename": "doc.pdf"}
data: {"type": "page", "page": 1, "total_pages": 5, "text": "...", ...}
data: {"type": "page", "page": 2, "total_pages": 5, "text": "...", ...}
data: {"type": "done"}
```

## Requirements

- Python 3.12+
- Tesseract OCR with Bengali language pack (`tesseract-data-ben`)
- System fonts: `gnu-free-fonts` (for PDF rendering)
