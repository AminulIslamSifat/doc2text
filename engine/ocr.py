"""Bangla OCR engine: image preprocessing, dual-pass Tesseract, word merging, text extraction."""

import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pymupdf
import pytesseract as pts
from bnunicodenormalizer import Normalizer

from engine.corrector import correct_text, BanglaCorrector

_corrector = BanglaCorrector()

_normalizer = Normalizer()


def preprocess(img_cv: np.ndarray, scale: int = 4) -> tuple[np.ndarray, np.ndarray, int]:
    """Upscale, grayscale, denoise. Returns (grayscale_denoised, adaptive_binary, scale)."""
    img_up = cv2.resize(img_cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    return denoised, adaptive, scale


def extract_words(image: np.ndarray, lang: str, scale: int, min_conf: int = 25, min_h: int = 18) -> list[dict]:
    """Run OCR and return filtered word list with coordinates mapped to original space."""
    data = pts.image_to_data(image, lang=lang, output_type=pts.Output.DICT, config="--psm 3")
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        conf = data["conf"][i]
        h = data["height"][i]
        if conf < min_conf or h < min_h:
            continue
        r = _normalizer(text)
        normalized = (r["normalized"] if r else None) or text
        words.append({
            "text": normalized,
            "left": data["left"][i] // scale,
            "top": data["top"][i] // scale,
            "width": data["width"][i] // scale,
            "height": h // scale,
            "conf": conf,
        })
    return words


def merge_words(words_a: list[dict], words_b: list[dict], overlap_thresh: int = 8) -> list[dict]:
    """Merge two word lists. When overlapping, prefer dictionary-valid word.
    
    Priority: 1) dictionary validity  2) confidence tiebreaker.
    Tesseract's grayscale pass often hallucinates extra characters (e.g. হইরান vs ইরান)
    with higher confidence. Dictionary check catches this.
    """
    merged = list(words_a)
    for wb in words_b:
        dominated = False
        for wa in merged:
            y_close = abs(wa["top"] - wb["top"]) <= overlap_thresh
            x_overlap = not (wa["left"] + wa["width"] < wb["left"] or
                            wb["left"] + wb["width"] < wa["left"])
            if y_close and x_overlap:
                a_valid = _corrector.is_valid(wa["text"])
                b_valid = _corrector.is_valid(wb["text"])
                if b_valid and not a_valid:
                    wa.update(wb)
                elif a_valid and not b_valid:
                    pass  # keep wa
                elif wb["conf"] > wa["conf"]:
                    wa.update(wb)
                dominated = True
                break
        if not dominated:
            merged.append(wb)
    return merged


def _words_to_text(words: list[dict]) -> str:
    """Convert sorted word list to text lines with spacing."""
    words.sort(key=lambda w: (w["top"], w["left"]))

    # Deduplicate
    deduped: list[dict] = []
    for w in words:
        is_dup = False
        for j, existing in enumerate(deduped):
            if (w["text"] == existing["text"]
                    and abs(w["top"] - existing["top"]) <= 10
                    and abs(w["left"] - existing["left"]) <= 10):
                if w["conf"] > existing["conf"]:
                    deduped[j] = w
                is_dup = True
                break
        if not is_dup:
            deduped.append(w)

    # Group into lines
    lines: list[list[dict]] = []
    for w in deduped:
        placed = False
        for line in lines:
            if abs(w["top"] - line[0]["top"]) <= 10:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    # Build text with spacing
    result_lines = []
    for line in lines:
        line.sort(key=lambda w: w["left"])
        parts = []
        for i, w in enumerate(line):
            if i > 0:
                gap = w["left"] - (line[i-1]["left"] + line[i-1]["width"])
                avg_char_w = max(w["width"] / max(len(w["text"]), 1), 4)
                spaces = max(1, round(gap / avg_char_w))
                parts.append(" " * spaces)
            parts.append(w["text"])
        result_lines.append("".join(parts))

    # Post-processing
    cleaned = []
    for line in result_lines:
        line = line.replace("|", "।")
        line = re.sub(r'(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])', '', line)
        line = re.sub(r' {3,}', '  ', line)
        cleaned.append(line.strip())

    return "\n".join(cleaned)


class OcrResult:
    """Result of an OCR operation."""
    def __init__(self, text: str, corrections: list[dict], word_count: int, line_count: int):
        self.text = text
        self.corrections = corrections
        self.word_count = word_count
        self.line_count = line_count


def ocr_image(image_path: str | Path, lang: str = "ben") -> OcrResult:
    """Full pipeline for a single image file."""
    img_cv = cv2.imread(str(image_path))
    if img_cv is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray_denoised, adaptive_bin, scale = preprocess(img_cv)
    words_gray = extract_words(gray_denoised, lang, scale)
    words_adapt = extract_words(adaptive_bin, lang, scale)
    words = merge_words(words_gray, words_adapt)

    raw_text = _words_to_text(words)
    corrected_text, corrections = correct_text(raw_text)

    lines = [l for l in corrected_text.split("\n") if l.strip()]
    return OcrResult(
        text=corrected_text,
        corrections=corrections,
        word_count=len(words),
        line_count=len(lines),
    )


def ocr_pdf_pages(pdf_path: str | Path, lang: str = "ben", dpi: int = 300) -> Iterator[tuple[int, OcrResult]]:
    """OCR each page of a PDF. Yields (page_number, OcrResult) tuples."""
    doc = pymupdf.open(str(pdf_path))
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=dpi)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

            gray_denoised, adaptive_bin, scale = preprocess(img_array)
            words_gray = extract_words(gray_denoised, lang, scale)
            words_adapt = extract_words(adaptive_bin, lang, scale)
            words = merge_words(words_gray, words_adapt)

            raw_text = _words_to_text(words)
            corrected_text, corrections = correct_text(raw_text)

            lines = [l for l in corrected_text.split("\n") if l.strip()]
            yield page_num, OcrResult(
                text=corrected_text,
                corrections=corrections,
                word_count=len(words),
                line_count=len(lines),
            )
    finally:
        doc.close()


def pdf_page_count(pdf_path: str | Path) -> int:
    """Return number of pages in a PDF."""
    doc = pymupdf.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
