# doc2text

Simple OCR web app — upload an image, get extracted text as JSON.

Dual engine support:
- **PaddleOCR** — fast, accurate for English/Chinese/Korean/Japanese
- **EasyOCR** — supports Bangla (`bn`) and 80+ languages

## Quick Start

```bash
uv sync
uv run python server.py
# → http://localhost:8765
```

## CLI Usage

```bash
# English via PaddleOCR (default)
uv run python ocr.py image.png

# Bangla via EasyOCR
uv run python ocr.py image.png -e easyocr -l bn

# Custom threshold + output path
uv run python ocr.py image.png -t 0.7 -o result.json
```

## API

### `POST /api/ocr`

Upload an image for OCR processing.

**Parameters:**
- `file` (multipart) — image file
- `threshold` (query, optional) — confidence threshold, default `0.8`
- `lang` (query, optional) — language code, default `en`
- `engine` (query, optional) — `paddle` or `easyocr`, default `paddle`

**Response:**
```json
{
  "results": [
    {"text": "Hello", "score": 0.98, "box": [10, 20, 200, 50]}
  ],
  "count": 1
}
```

### `GET /`

Web UI with drag-drop upload, bounding box visualization, and language/engine selection.

## Project Structure

```
doc2text/
├── ocr.py           # Core OCR logic (CLI + importable)
├── server.py        # FastAPI server
├── static/
│   └── index.html   # Web UI
└── pyproject.toml   # Dependencies
```

## Supported Languages

| Engine     | Languages |
|------------|-----------|
| PaddleOCR  | en, ch, korean, japan, chinese_cht, devanagari, ta, te, ka, latin, arabic, cyrillic |
| EasyOCR    | bn, hi, en, ar, ch, korean, japan, + 80 more |

> **Note:** PaddleOCR does not support Bangla/Bengali. Use EasyOCR (`-e easyocr -l bn`) for Bangla text.
