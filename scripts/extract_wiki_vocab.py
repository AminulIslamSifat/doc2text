"""Extract unique Bangla words from Wikipedia XML dump.

Usage:
    python scripts/extract_wiki_vocab.py /tmp/bnwiki-dump.xml.bz2

Outputs:
    engine/data/bangla_curated_wiki.txt  — new curated words found in Wikipedia
    Stats printed to stdout.
"""

import bz2
import re
import sys
from pathlib import Path
from xml.etree.ElementTree import iterparse

# Only keep Bangla characters, digits, and common punctuation
BANGLA_WORD = re.compile(r'[\u0980-\u09FF][\u0980-\u09FF\u09E6-\u09EF\u09CD\u200C\u200D]*')
MIN_LEN = 2

def extract_words_from_dump(dump_path: str) -> set[str]:
    """Parse Wikipedia XML dump and extract all unique Bangla words."""
    words: set[str] = set()
    ns = '{http://www.mediawiki.org/xml/export-0.11/}'
    
    # Try multiple namespace versions
    for ns_prefix in [
        '{http://www.mediawiki.org/xml/export-0.11/}',
        '{http://www.mediawiki.org/xml/export-0.10/}',
        '{http://www.mediawiki.org/xml/export-0.9/}',
        '',
    ]:
        try:
            opener = bz2.open if dump_path.endswith('.bz2') else open
            with opener(dump_path, 'rt', encoding='utf-8') as f:
                for event, elem in iterparse(f, events=('end',)):
                    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                    if tag == 'text' and elem.text:
                        for match in BANGLA_WORD.finditer(elem.text):
                            w = match.group()
                            if len(w) >= MIN_LEN:
                                words.add(w)
                        elem.clear()
            break
        except Exception:
            continue
    
    return words


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_wiki_vocab.py <dump.xml.bz2>")
        sys.exit(1)
    
    dump_path = sys.argv[1]
    data_dir = Path(__file__).parent.parent / "engine" / "data"
    
    print(f"Loading existing word list...")
    existing_words: set[str] = set()
    with open(data_dir / "bangla_words.txt", 'r', encoding='utf-8-sig') as f:
        for line in f:
            w = line.strip()
            if w:
                existing_words.add(w)
    print(f"  Existing word list: {len(existing_words):,} words")
    
    print(f"Loading existing curated set...")
    existing_curated: set[str] = set()
    with open(data_dir / "bangla_curated.txt", 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            if w:
                existing_curated.add(w)
    print(f"  Existing curated: {len(existing_curated):,} words")
    
    print(f"\nExtracting words from Wikipedia dump...")
    wiki_words = extract_words_from_dump(dump_path)
    print(f"  Unique Bangla words in Wikipedia: {len(wiki_words):,}")
    
    # Words in Wikipedia AND in our existing word list = verified real Bangla
    verified = wiki_words & existing_words
    print(f"  Verified (in both Wikipedia + word list): {len(verified):,}")
    
    # New curated = verified words NOT already in curated set
    new_curated = verified - existing_curated
    print(f"  NEW curated candidates: {len(new_curated):,}")
    
    # Write new curated words
    out_path = data_dir / "bangla_curated_wiki.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        for w in sorted(new_curated):
            f.write(w + '\n')
    print(f"\nWrote {len(new_curated):,} new curated words to {out_path}")
    
    # Also create merged curated set
    merged = existing_curated | new_curated
    merged_path = data_dir / "bangla_curated_merged.txt"
    with open(merged_path, 'w', encoding='utf-8') as f:
        for w in sorted(merged):
            f.write(w + '\n')
    print(f"Wrote merged curated set ({len(merged):,} words) to {merged_path}")
    
    # Check some test words
    test_words = ['ভ্রান্তিবিলাস', 'মাসকাল', 'পঞ্চবিংশতি', 'এতদ্বিষয়ক', 'রুদ্ধ', 'করিয়া']
    print("\nTest word coverage:")
    for w in test_words:
        in_wiki = w in wiki_words
        in_list = w in existing_words
        in_new = w in new_curated
        print(f"  {w}: wiki={in_wiki}, wordlist={in_list}, new_curated={in_new}")


if __name__ == '__main__':
    main()
