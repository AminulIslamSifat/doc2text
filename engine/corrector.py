"""Bangla OCR post-processing corrector using SymSpell + curated dictionary."""

import re
from pathlib import Path

_DICT_PATH = Path(__file__).parent / "data" / "bangla_words.txt"
_CURATED_PATH = Path(__file__).parent / "data" / "bangla_curated.txt"
_BANGLA_RANGE = re.compile(r'[\u0980-\u09FF]')
_PUNCT = set('।॥,.:;!?()[]{}\'"-/—–…·')
_BANGLA_DIGIT = re.compile(r'[\u09E6-\u09EF]')

# Known false corrections to block (dictionary has wrong entries)
_BLOCK_CORRECTIONS = {
    'পঙ্চবিংশতি',  # wrong; correct is পঞ্চবিংশতি
}

# Common Bangla inflectional suffixes to strip before dictionary lookup
_SUFFIXES = [
    'দের', 'দেরকে', 'দেরও',
    'কে', 'কেও',
    'রা', 'রাকে',
    'তে', 'তেই', 'তেও',
    'য়ে', 'য়েই', 'য়েও',
    'লে', 'লেই', 'লেও',
    'ার', 'ের', 'ির', 'ুর', 'োর',
    'ায়', 'েয়', 'িয়',
    'টি', 'টা', 'গুলি', 'গুলো',
    'ই', 'ও', 'ইতো',
    'র', 'য়',
]


class BanglaCorrector:
    def __init__(self, max_edit_distance: int = 1, min_word_len: int = 2):
        self.max_edit = max_edit_distance
        self.min_word_len = min_word_len
        self._curated: set[str] = set()
        self._ss = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        from symspellpy import SymSpell
        self._ss = SymSpell(max_dictionary_edit_distance=self.max_edit, prefix_length=5)

        # Load full word list into SymSpell
        if _DICT_PATH.exists():
            with open(_DICT_PATH, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    w = line.strip()
                    if w and _BANGLA_RANGE.search(w):
                        self._ss.create_dictionary_entry(w, 1)

        # Load curated set for filtering
        if _CURATED_PATH.exists():
            with open(_CURATED_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    w = line.strip()
                    if w:
                        self._curated.add(w)

        self._loaded = True

    @property
    def dict_size(self) -> int:
        self._load()
        return len(self._ss.words) if self._ss else 0

    def is_valid(self, word: str) -> bool:
        self._load()
        return word in self._ss.words if self._ss else False

    def _strip_suffixes(self, word: str) -> list[str]:
        stems = [word]
        for suffix in _SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix):
                stems.append(word[:-len(suffix)])
        return stems

    def suggest(self, word: str) -> str | None:
        """Find best correction using SymSpell, filtered by curated set."""
        self._load()
        from symspellpy import Verbosity

        if not self._ss or word in self._ss.words:
            return None  # already valid

        # Skip words with Bangla digits
        if _BANGLA_DIGIT.search(word):
            return None

        # Skip very short words
        bangla_chars = _BANGLA_RANGE.findall(word)
        if len(bangla_chars) < self.min_word_len:
            return None

        # Check if stem is valid (inflected form)
        for stem in self._strip_suffixes(word):
            if stem in self._ss.words:
                return None  # stem is valid, keep original

        # Get all closest suggestions
        suggestions = self._ss.lookup(word, Verbosity.CLOSEST, max_edit_distance=self.max_edit)
        if not suggestions:
            return None

        # Single candidate = unique match, high confidence even if not curated
        if len(suggestions) == 1:
            term = suggestions[0].term
            if term not in _BLOCK_CORRECTIONS:
                return term
            return None

        # Multiple candidates — only accept if one is in curated set
        curated_matches = [s.term for s in suggestions if s.term in self._curated]
        if curated_matches:
            return curated_matches[0]

        # No curated match among multiple candidates — too ambiguous
        return None

    def correct_text(self, text: str) -> tuple[str, list[dict]]:
        corrections: list[dict] = []
        tokens = re.split(r'(\s+)', text)
        result = []

        for token in tokens:
            if not token or token.isspace():
                result.append(token)
                continue

            stripped = token.strip(''.join(_PUNCT))
            prefix = token[:len(token) - len(token.lstrip(''.join(_PUNCT)))]
            suffix = token[len(token.rstrip(''.join(_PUNCT))):]

            if not stripped or not _BANGLA_RANGE.search(stripped):
                result.append(token)
                continue

            suggestion = self.suggest(stripped)
            if suggestion:
                dist = sum(1 for a, b in zip(stripped, suggestion) if a != b)
                corrections.append({
                    'original': stripped,
                    'corrected': suggestion,
                    'distance': dist,
                })
                result.append(prefix + suggestion + suffix)
            else:
                result.append(token)

        return ''.join(result), corrections


_corrector: BanglaCorrector | None = None


def get_corrector() -> BanglaCorrector:
    global _corrector
    if _corrector is None:
        _corrector = BanglaCorrector(max_edit_distance=1)
    return _corrector


def correct_text(text: str) -> tuple[str, list[dict]]:
    return get_corrector().correct_text(text)
