"""Text normalisation shared by quote verification and scoring: quote lookup with character offsets, token F1,
and normalised dates / durations / governing law / party names."""
import bisect
import re
import unicodedata
from datetime import date

_CHAR_MAP = {"‘": "'", "’": "'", "‚": "'", "′": "'", "`": "'", "“": '"', "”": '"',
             "„": '"', "″": '"', "‐": "-", "‑": "-", "‒": "-", "–": "-",
             "—": "-", "−": "-", " ": " ", "­": ""}


def normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Lower-case, straight quotes / dashes, every whitespace run -> one space. Returns the normalised text and,
    for each of its characters, the offset of the original character (so a match maps back to the source)."""
    out, where, space = [], [], False
    for i, ch in enumerate(text):
        ch = _CHAR_MAP.get(ch, ch)
        if not ch:
            continue
        if ch.isspace():
            if not space and out:
                out.append(" ")
                where.append(i)
            space = True
            continue
        space = False
        out.append(ch.lower())
        where.append(i)
    if out and out[-1] == " ":
        out.pop()
        where.pop()
    return "".join(out), where


def normalise(text: str) -> str:
    return normalise_with_map(text)[0]


MIN_QUOTE_CHARS = 8  # "yes" or "the" would "verify" anywhere
_BULLETS = set("•▪●◦‣∙·■□➢➤►")
_QUOTE_CHARS = set("'\"")


def compact_with_map(text: str, drop_quotes: bool = False) -> tuple[str, list[int]]:
    """Like normalise_with_map, but for finding quotes: no whitespace at all ("only,to" == "only, to"), no bullets
    or private-use symbols (PDF bullets such as U+F0B7), ligatures expanded (ﬁ -> fi), every quote mark -> ' (or
    dropped with drop_quotes). Formatting differences only: a changed word still doesn't match."""
    out, where = [], []
    for i, ch in enumerate(text):
        if ch.isspace() or ch in _BULLETS or unicodedata.category(ch) == "Co":
            continue
        for c in unicodedata.normalize("NFKC", _CHAR_MAP.get(ch, ch)).lower():
            c = _CHAR_MAP.get(c, c)
            if c == '"':
                c = "'"
            if not c or c.isspace() or (drop_quotes and c in _QUOTE_CHARS):
                continue
            out.append(c)
            where.append(i)
    return "".join(out), where


class QuoteFinder:
    """Find quotes in one document; the compact texts are built once."""

    def __init__(self, text: str):
        self.text = text
        self.views = [compact_with_map(text), compact_with_map(text, drop_quotes=True)]

    def find(self, quote: str, near: int = 0) -> tuple[int, int] | None:
        """Original (start, end) of `quote`, ignoring case, whitespace, bullets, ligatures and quote/dash styles;
        None if absent. Quotes shorter than MIN_QUOTE_CHARS never verify. A quote with "..." must have each part,
        in order (within 2,000 characters of each other). A colon counts as such a gap too: models quote a list's
        lead-in and one item of it ("... shall not apply to information that: (d) is independently developed"),
        skipping the items in between. The model often drops the quote marks around a defined term
        ("Confidential Information" means ...), so a second pass ignores quote marks altogether.
        `near` = prefer a match at or after this original offset (the chunk the answer came from)."""
        pieces = re.split(r"\.\.\.|\u2026|\[\.\.\.\]|(?<=:)", quote or "")
        for drop_quotes, (norm, where) in zip((False, True), self.views):
            raw = [compact_with_map(p, drop_quotes)[0] for p in pieces]
            n_near = bisect.bisect_left(where, near) if near else 0
            for parts in (raw, [p.strip(".,;:()[]'") for p in raw]):  # exact first, then without edge punctuation
                parts = [p for p in parts if len(p) >= 3]
                if sum(map(len, parts)) >= MIN_QUOTE_CHARS and (span := self._search(norm, where, parts, n_near)):
                    return span
        return None

    @staticmethod
    def _search(norm: str, where: list[int], parts: list[str], n_near: int) -> tuple[int, int] | None:
        for start_at in (n_near, 0):
            pos = norm.find(parts[0], start_at)
            while pos != -1:
                end, ok = pos + len(parts[0]), True
                for p in parts[1:]:
                    nxt = norm.find(p, end)
                    if nxt == -1 or nxt - end > 2000:
                        ok = False
                        break
                    end = nxt + len(p)
                if ok:
                    return where[pos], where[end - 1] + 1
                pos = norm.find(parts[0], pos + 1)
        return None


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalise(text or ""))


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token overlap F1."""
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    hit = 0
    for t in g:
        if common.get(t, 0) > 0:
            common[t] -= 1
            hit += 1
    if hit == 0:
        return 0.0
    precision, recall = hit / len(p), hit / len(g)
    return 2 * precision * recall / (precision + recall)


# --- values ---------------------------------------------------------------------------------------------------

_MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                           "nov", "dec"])}
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty-five": 45,
            "forty": 40, "sixty": 60, "ninety": 90, "a": 1, "an": 1}
_UNITS = {"day": 1, "business day": 1, "week": 7, "month": 30, "year": 365}
_OPEN_ENDED = ("perpetual", "indefinite", "evergreen", "until terminated")


def _year(y: str) -> int:
    n = int(y)
    return n if len(y) == 4 else (2000 + n if n <= 30 else 1900 + n)


def norm_date(value) -> str | None:
    """-> 'YYYY-MM-DD' (or 'perpetual'); also accepts M/D/YY as in CUAD's master_clauses.csv. None if unparseable."""
    if not value:
        return None
    s = normalise(str(value))
    if any(w in s for w in _OPEN_ENDED):
        return "perpetual"
    try:
        if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s):
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        if m := re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\b", s):  # US style M/D/Y
            return date(_year(m[3]), int(m[1]), int(m[2])).isoformat()
        s2 = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s).replace(" day of ", " ")
        if m := re.search(r"\b([a-z]{3})[a-z]*\.? (\d{1,2}),? (\d{4})\b", s2):  # September 7, 1999
            if m[1] in _MONTHS:
                return date(int(m[3]), _MONTHS[m[1]], int(m[2])).isoformat()
        if m := re.search(r"\b(\d{1,2}) ([a-z]{3})[a-z]*,? (\d{4})\b", s2):  # 7 September 1999
            if m[2] in _MONTHS:
                return date(int(m[3]), _MONTHS[m[2]], int(m[1])).isoformat()
    except ValueError:
        return None
    return None


def norm_duration(value) -> str | None:
    """-> days as a string ('365', '30'), or 'perpetual'. '1 year' == '12 months' == '365 days'."""
    if not value:
        return None
    s = normalise(str(value))
    if any(w in s for w in _OPEN_ENDED):
        return "perpetual"
    for m in re.finditer(r"(\d+|[a-z-]+)\)?\s*(?:\((\d+)\)\s*)?(business day|day|week|month|year)s?\b", s):
        # "thirty (30) days": prefer the number in brackets
        n = int(m[2]) if m[2] else int(m[1]) if m[1].isdigit() else _NUMBERS.get(m[1])
        if n is None:
            continue
        if m[3] == "month" and n % 12 == 0:
            return str(n // 12 * 365)
        return str(n * _UNITS[m[3]])
    return None


_LAW_NOISE = re.compile(r"\b(the|laws?|of|state|commonwealth|province|republic|united states|usa|u\.s\.a?|us|"
                        r"america|governed|by|and|federal)\b")


def norm_law(value) -> str | None:
    """'the laws of the State of New York, USA' -> 'new york'. Only the first place before a comma is kept,
    so 'Ontario, Canada' == 'Ontario'."""
    if not value:
        return None
    s = normalise(str(value)).split(",")[0].split(";")[0]
    s = re.sub(r"\s+", " ", _LAW_NOISE.sub(" ", re.sub(r"[^a-z ,.]", " ", s))).strip(" .")
    return s or None


_COMPANY_NOISE = re.compile(r"\b(inc|incorporated|llc|l l c|ltd|limited|corp|corporation|co|company|plc|lp|llp|"
                            r"s a|sa|gmbh|ag|the)\b")


def norm_party(name: str) -> str:
    """Company name without defined terms ("Licensor"), punctuation or legal suffixes."""
    s = re.sub(r"\([^)]*\)", " ", name or "")
    s = re.sub(r"[^a-z0-9 ]", " ", normalise(s))
    return re.sub(r"\s+", " ", _COMPANY_NOISE.sub(" ", s)).strip()


def split_gold_parties(answer: str) -> list[str]:
    """CUAD master_clauses 'A Inc. ("Company"); B Ltd. ("Seller")' -> ['A Inc.', 'B Ltd.']."""
    return [p.strip() for p in re.split(r";|\n", answer or "") if norm_party(p)]


def parties_match(pred: list[str], gold: list[str]) -> bool:
    """Every gold party is named (one name containing the other counts) by some predicted party."""
    p = [norm_party(x) for x in pred if len(norm_party(x)) >= 3]
    g = [norm_party(x) for x in gold if len(norm_party(x)) >= 3]
    return bool(g) and all(any(a == b or a in b or b in a for b in p) for a in g)
