"""Assamese sentence splitter.

Assamese uses the dari '।' (U+0964) as its full stop; '?' and '!' also terminate.
The Latin '.' appears mostly inside abbreviations, initials, decimals and dates,
so it only terminates a sentence under restrictive conditions.
"""
import re

# U+0964 DEVANAGARI DANDA is the standard Assamese full stop.
# U+09F7 BENGALI CURRENCY NUMERATOR FOUR (৷) is visually identical and is very
# widely misused as a danda in Assamese/Bengali Wikipedia text, so it must count
# as a terminator too -- otherwise sentences silently merge across it.
# U+0965 (॥) appears in verse and quotations.
DARI = "।৷"
DOUBLE_DARI = "॥"

TERMINATORS = DARI + DOUBLE_DARI + "?!"
CLOSERS = "\"'”’)]}»"

# Assamese/Bengali script range, incl. the Assamese-specific ৰ (09F0) and ৱ (09F1)
RE_AS_CHAR = re.compile(r"[ঀ-৿]")
RE_DIGIT = re.compile(r"[0-9০-৯]")

# '.' preceded by a single letter (initials: A. K. Barua) or by a digit
RE_INITIAL = re.compile(r"(?:^|[\s(])[A-Za-zঀ-৿]\.$")
RE_DECIMAL_LEFT = re.compile(r"[0-9০-৯]$")

ABBREV = {
    "ড", "ডঃ", "শ্ৰী", "শ্ৰীমতী", "মহ", "ইং", "খ্ৰী", "খ্ৰীঃ", "অৰ্থাৎ",
    "dr", "mr", "mrs", "ms", "prof", "st", "no", "vs", "etc", "i.e", "e.g",
    "jr", "sr", "fig", "eq", "vol", "pp", "ed", "eds", "approx",
}


def _is_hard_terminator(text, i, allow_period=True):
    """Is text[i] a sentence-ending punctuation mark?"""
    ch = text[i]
    if ch in DARI + DOUBLE_DARI:
        return True
    if ch in "?!":
        return True
    if ch == ".":
        if not allow_period:
            return False
        left = text[:i]
        if RE_DECIMAL_LEFT.search(left):          # 3.14, 1947.
            nxt = text[i + 1 : i + 2]
            if RE_DIGIT.match(nxt or ""):
                return False
        if RE_INITIAL.search(left):                # A. K.
            return False
        tok = re.split(r"[\s(\[]", left)[-1].rstrip(".").lower()
        if tok in ABBREV:
            return False
        return True
    return False


def split_sentences(paragraph, allow_period=None):
    """Split one paragraph into sentences. Returns a list of strings.

    In Assamese-script prose the sentence terminator is the dari '।'; a Latin
    '.' there is nearly always an abbreviation, an initial or a decimal
    (পি.এইচ.ডি., ড., ১৯২৬.৫). So by default '.' only terminates sentences in
    paragraphs that are not predominantly Assamese script.
    """
    text = paragraph.strip()
    if not text:
        return []
    if allow_period is None:
        allow_period = assamese_ratio(text) < 0.5
    sents, start = [], 0
    i, n = 0, len(text)
    while i < n:
        if _is_hard_terminator(text, i, allow_period):
            j = i + 1
            # absorb repeated terminators and any closing quotes/brackets
            while j < n and (text[j] in TERMINATORS or text[j] in CLOSERS):
                j += 1
            if j >= n or text[j].isspace():
                piece = text[start:j].strip()
                if piece:
                    sents.append(piece)
                start = j
                i = j
                continue
        i += 1
    tail = text[start:].strip()
    if tail:
        sents.append(tail)
    return sents


def assamese_ratio(text):
    """Fraction of letter characters that are in the Assamese/Bengali block."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if RE_AS_CHAR.match(c)) / len(letters)


def word_count(text):
    return len([w for w in re.split(r"\s+", text.strip()) if w])
