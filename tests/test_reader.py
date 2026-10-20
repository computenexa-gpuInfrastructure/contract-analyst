import pytest
from conftest import FIXTURES

from contract_analyst import reader


def test_chunks_keep_offsets_cover_text_and_overlap():
    text = "".join(f"Clause {i}. " + "word " * 40 + "\n\n" for i in range(60))
    chunks = reader.chunk(text, 1000, overlap=200)
    assert len(chunks) > 5
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for i, c in enumerate(chunks):
        assert c.index == i
        assert c.text == text[c.start:c.end]
        assert len(c.text) <= 1000
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # overlap: nothing falls between two chunks
        assert a.end - b.start <= 200


def test_chunks_cut_at_paragraph_breaks():
    text = ("A" * 600 + "\n\n") * 5
    for c in reader.chunk(text, 1000, overlap=0)[:-1]:
        assert c.text.endswith("\n\n")


def test_short_text_is_one_chunk_and_text_without_breaks_still_progresses():
    assert [c.text for c in reader.chunk("short contract", 1000)] == ["short contract"]
    chunks = reader.chunk("x" * 2500, 1000, overlap=100)
    assert chunks[-1].end == 2500 and all(len(c.text) <= 1000 for c in chunks)


def test_chunk_size_follows_served_context():
    # Qwen 2.5 on the Model API serves 32,768 tokens: capped at MAX_CHUNK_CHARS
    assert reader.chunk_chars_for(32768) == reader.MAX_CHUNK_CHARS
    assert reader.chunk_chars_for(None) == reader.MAX_CHUNK_CHARS
    small = reader.chunk_chars_for(8192)
    assert small == int((8192 - reader.RESERVED_TOKENS) * reader.CHARS_PER_TOKEN)
    assert reader.chunk_chars_for(4000) == 2000  # never below the floor


def _tiny_pdf(text: str) -> bytes:
    """A one-page PDF with `text` in Helvetica (hand-built; no PDF library needed)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               b"/Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_read_text_and_pdf(tmp_path):
    assert "MASTER SUPPLY AGREEMENT" in reader.read_text(FIXTURES / "sample_contract.txt")
    pytest.importorskip("pypdf")
    pdf = tmp_path / "c.pdf"
    pdf.write_bytes(_tiny_pdf("This Agreement is governed by the laws of Texas."))
    assert "governed by the laws of Texas" in reader.read_text(pdf)
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(_tiny_pdf(""))
    with pytest.raises(ValueError, match="no text"):
        reader.read_text(empty)
