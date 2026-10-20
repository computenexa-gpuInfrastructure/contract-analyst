"""Read a contract (PDF or text) and cut it into chunks that fit the model's context, keeping character offsets."""
from dataclasses import dataclass
from pathlib import Path

# A vLLM server started without --max-model-len (as ComputeNexa's Model API is) serves the model's own maximum
# context: 32,768 tokens for Qwen 2.5 7B / 14B Instruct. We ask the server (/v1/models reports max_model_len) and
# fall back to this for servers that don't say (e.g. hosted APIs).
DEFAULT_CONTEXT_TOKENS = 32768
CHARS_PER_TOKEN = 3.0  # conservative for English legal text (usually ~4), so a chunk never overflows
RESERVED_TOKENS = 6000  # instructions + field list (~2,000) + the JSON answer (max_tokens 3,000) + margin
MAX_CHUNK_CHARS = 40000  # ~10k tokens: big enough that most contracts take 1-3 calls, small enough to stay sharp
OVERLAP_CHARS = 1500  # a clause cut at a chunk edge is still whole in the next chunk


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int  # character offsets into the full text
    end: int
    text: str


def read_text(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # only needed for PDFs

        pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
        text = "\n\n".join(pages)
        if not text.strip():
            raise ValueError(f"{path.name}: no text found in the PDF (scanned? run OCR first)")
        return text
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_chars_for(context_tokens: int | None, prompt_tokens: int = RESERVED_TOKENS) -> int:
    """How many characters of contract fit next to the prompt and the answer."""
    budget = (context_tokens or DEFAULT_CONTEXT_TOKENS) - prompt_tokens
    return max(2000, min(MAX_CHUNK_CHARS, int(budget * CHARS_PER_TOKEN)))


def chunk(text: str, size: int, overlap: int = OVERLAP_CHARS) -> list[Chunk]:
    """Overlapping chunks of at most `size` characters, cut at a paragraph / line / sentence break when one is
    near the end. chunk.text == text[chunk.start:chunk.end] always."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = min(overlap, size // 4)
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            window = text[start + size // 2:end]  # look for a break in the last half
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep)
                if cut != -1:
                    end = start + size // 2 + cut + len(sep)
                    break
        chunks.append(Chunk(len(chunks), start, end, text[start:end]))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
