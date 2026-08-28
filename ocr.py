#!/usr/bin/env python3
"""OCR CLI — PaddleOCR with unified output format."""

import os
import json
import argparse
import logging
from pathlib import Path

# Suppress Paddle noise before imports
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ["FLAGS_eager_delete_tensor_gb"] = "0.0"
os.environ["FLAGS_fast_eager_deletion_mode"] = "True"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
os.environ.setdefault("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "10")

logging.getLogger("ppocr").setLevel(logging.ERROR)

import cv2
import numpy as np

_paddle_engines = {}


# ── PaddleOCR ──────────────────────────────────────────────────

def _get_paddle_engine(lang: str):
    if lang not in _paddle_engines:
        import paddle
        from paddleocr import PaddleOCR
        cpu_threads = min(os.cpu_count() or 1, 4)
        paddle.set_device("cpu")
        _paddle_engines[lang] = PaddleOCR(
            use_angle_cls=False,
            lang=lang,
            use_gpu=False,
            det_limit_side_len=1024,
            cpu_threads=cpu_threads,
            ir_optim=True,
            layout=False,
            table=False,
            formula=False,
        )
    return _paddle_engines[lang]


def _run_paddle(img_path: str, threshold: float, lang: str) -> list[dict]:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not decode image: {img_path}")

    ocr = _get_paddle_engine(lang)
    output = ocr.ocr(img, cls=False)

    results = []
    if output and output[0]:
        for line in output[0]:
            if not line:
                continue
            pts = np.array(line[0])
            text = line[1][0]
            score = float(line[1][1])
            if score > threshold:
                xmin = int(pts[:, 0].min())
                ymin = int(pts[:, 1].min())
                xmax = int(pts[:, 0].max())
                ymax = int(pts[:, 1].max())
                results.append({
                    "text": text,
                    "score": round(score, 4),
                    "box": [xmin, ymin, xmax, ymax],
                })
    return results


# ── Unified entry point ───────────────────────────────────────

ENGINES = {"paddle": _run_paddle}

def run_ocr(
    img_path: str,
    confidence_threshold: float = 0.8,
    lang: str = "en",
    engine: str = "paddle",
) -> list[dict]:
    """Run OCR on an image file.

    Args:
        img_path: Path to image
        confidence_threshold: Minimum score to include result
        lang: Language code (en, hi, ch, korean, japan, etc.)
        engine: 'paddle'
    """
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    if engine not in ENGINES:
        raise ValueError(f"Unknown engine '{engine}'. Use: {list(ENGINES.keys())}")

    return ENGINES[engine](img_path, confidence_threshold, lang)


def main():
    parser = argparse.ArgumentParser(description="OCR an image to JSON")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("-t", "--threshold", type=float, default=0.8)
    parser.add_argument("-l", "--lang", default="en",
                        help="Language: en, hi, ch, korean, japan, devanagari, etc.")
    parser.add_argument("-e", "--engine", default="paddle", choices=["paddle"],
                        help="OCR engine (default: paddle)")
    args = parser.parse_args()

    results = run_ocr(args.image, args.threshold, args.lang, args.engine)

    if args.output is None:
        stem = Path(args.image).stem
        args.output = str(Path(args.image).parent / f"{stem}_ocr.json")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"OK [{args.engine}]: {len(results)} regions -> {args.output}")


if __name__ == "__main__":
    main()
