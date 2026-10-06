"""
Wikitext cleaner that PRESERVES paragraph boundaries.

Standard extractors (wikiextractor, mwparserfromhell's strip_code) are tuned for
bag-of-text pretraining and are careless about blank lines. For a paragraph
segmentation dataset the blank line IS the label, so this module keeps it.
"""
import re
import unicodedata

# ---------------------------------------------------------------- balanced removal

def _remove_balanced(text, open_tok, close_tok):
    """Remove balanced {{...}} / {|...|} / [[...]] constructs, handling nesting."""
    out = []
    i = 0
    n = len(text)
    lo, lc = len(open_tok), len(close_tok)
    while i < n:
        if text.startswith(open_tok, i):
            depth = 1
            j = i + lo
            while j < n and depth > 0:
                if text.startswith(open_tok, j):
                    depth += 1
                    j += lo
                elif text.startswith(close_tok, j):
                    depth -= 1
                    j += lc
                else:
                    j += 1
            if depth == 0:
                i = j            # drop the whole construct
                continue
            else:
                # unbalanced: drop to end of line and carry on
                nl = text.find("\n", i)
                i = nl if nl != -1 else n
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


# File/image links may nest ([[File:x|thumb|[[link]] caption]]), so handle them
# with a depth-aware scan rather than a regex.
FILE_PREFIXES = (
    "file:", "image:", "চিত্ৰ:", "চিত্র:", "ছবি:", "মিডিয়া:", "media:",
)
CATEGORY_PREFIXES = ("category:", "শ্ৰেণী:", "শ্রেণী:", "বিষয়শ্ৰেণী:")


def _strip_wikilinks(text):
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("[[", i):
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if text.startswith("[[", j):
                    depth += 1
                    j += 2
                elif text.startswith("]]", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            inner = text[i + 2 : j - 2] if depth == 0 else text[i + 2 : j]
            low = inner.lstrip().lower()
            if low.startswith(FILE_PREFIXES) or low.startswith(CATEGORY_PREFIXES):
                pass  # drop entirely, caption and all
            else:
                # [[target|label]] -> label ; [[target]] -> target
                parts = inner.split("|")
                out.append(parts[-1] if len(parts) > 1 else parts[0])
            i = j
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


# ---------------------------------------------------------------- regex passes

RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_REF_PAIR = re.compile(r"<ref[^>/]*>.*?</ref>", re.S | re.I)
RE_REF_SELF = re.compile(r"<ref[^>]*/\s*>", re.I)
RE_DROP_BLOCK = re.compile(
    r"<(gallery|table|math|score|timeline|imagemap|syntaxhighlight|source|pre|poem)[^>]*>.*?</\1>",
    re.S | re.I,
)
RE_TAG = re.compile(r"<[^>]{0,200}>")
RE_EXTLINK_LABEL = re.compile(r"\[(?:https?:|//)[^\s\]]+\s+([^\]]*)\]")
RE_EXTLINK_BARE = re.compile(r"\[(?:https?:|//)[^\]]*\]")
RE_BOLDIT = re.compile(r"'{2,5}")
RE_HEADING = re.compile(r"^\s*(={2,6})\s*(.+?)\s*\1\s*$")
RE_MAGIC = re.compile(r"^__[A-Z]+__$")
RE_WS = re.compile(r"[ \t ]+")

# Lines that are structural rather than prose.
NON_PROSE_PREFIX = ("*", "#", ":", ";", "|", "!", "{", "}", "=")

# Sections that are reference apparatus, not prose.
BOILERPLATE_HEADINGS = (
    "তথ্যসূত্ৰ", "তথ্যসূত্র", "সন্দৰ্ভ", "সন্দর্ভ", "প্ৰসংগ", "প্রসঙ্গ",
    "বাহ্যিক সংযোগ", "বাহ্যিক সূত্ৰ", "বাহ্যিক", "লগতে চাওক", "আৰু চাওক",
    "গ্ৰন্থপঞ্জী", "গ্রন্থপঞ্জি", "পাদটীকা", "টোকা", "আৰু পঢ়ক",
    "চিত্ৰশালা", "চিত্ৰাগাৰ", "গেলাৰী",
    "references", "external links", "see also", "further reading",
    "bibliography", "notes", "gallery", "footnotes",
)

HEADING_MARK = "\x00HEADING\x00"


def is_boilerplate_heading(title):
    t = title.strip().lower()
    return any(b in t for b in BOILERPLATE_HEADINGS)


def clean_wikitext(text):
    """Return cleaned text where '\n\n' marks a paragraph break and
    HEADING_MARK<level>|<title> lines mark section starts."""
    t = text
    t = RE_COMMENT.sub("", t)
    t = RE_REF_PAIR.sub("", t)
    t = RE_REF_SELF.sub("", t)
    t = RE_DROP_BLOCK.sub("", t)

    # tables first (they contain templates), then templates
    t = _remove_balanced(t, "{|", "|}")
    t = _remove_balanced(t, "{{", "}}")

    t = _strip_wikilinks(t)

    t = RE_EXTLINK_LABEL.sub(r"\1", t)
    t = RE_EXTLINK_BARE.sub("", t)
    t = RE_TAG.sub("", t)
    t = RE_BOLDIT.sub("", t)

    lines = []
    for raw in t.split("\n"):
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        m = RE_HEADING.match(line)
        if m:
            lines.append("")
            lines.append(f"{HEADING_MARK}{len(m.group(1))}|{m.group(2)}")
            lines.append("")
            continue
        if RE_MAGIC.match(line):
            continue
        if line.startswith(NON_PROSE_PREFIX):
            # structural line: treat as a hard break so prose either side is
            # not silently glued into one paragraph
            lines.append("")
            continue
        lines.append(RE_WS.sub(" ", line))

    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return unicodedata.normalize("NFC", t).strip()


def parse_sections(cleaned, lead_title="__LEAD__"):
    """cleaned text -> [(heading_title, level, [paragraph, ...]), ...]"""
    sections = []
    cur_title, cur_level, buf = lead_title, 1, []

    def flush():
        paras = [p.strip() for p in "\n".join(buf).split("\n\n")]
        paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in paras if p.strip()]
        if paras:
            sections.append((cur_title, cur_level, paras))

    for block in cleaned.split("\n"):
        if block.startswith(HEADING_MARK):
            flush()
            lvl, _, title = block[len(HEADING_MARK) :].partition("|")
            cur_title, cur_level, buf = title, int(lvl), []
        else:
            buf.append(block)
    flush()
    return sections
