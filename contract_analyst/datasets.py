"""Download CUAD and ContractNLI into a cache folder and turn them into one gold format.

Both datasets are CC BY 4.0. They are downloaded on first use (never committed): on ComputeNexa the cache is
~/work/.cache/contract-analyst, so it's saved with your project and not downloaded again.

  CUAD v1 (The Atticus Project): 510 commercial contracts, 41 clause categories.
    - data.zip from the CUAD GitHub repo: CUADv1.json (all contracts, SQuAD format) + test.json, the OFFICIAL test
      split (102 contracts) used in the CUAD paper; train = the other 408.
    - CUAD_v1.zip from Zenodo: only master_clauses.csv is read, for normalised gold values (dates, durations,
      governing law, parties).
  ContractNLI (Koreeda & Manning 2021): 607 NDAs with official train / dev / test splits (423 / 61 / 123).
"""
import csv
import hashlib
import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .fields import NLI_LABELS, fields_for
from .textnorm import split_gold_parties


@dataclass(frozen=True)
class Source:
    url: str
    filename: str
    size: int  # bytes; checked so a wrong or truncated file is never used
    sha256: str


SOURCES = {
    "cuad-data": Source("https://github.com/TheAtticusProject/cuad/raw/main/data.zip", "cuad-data.zip", 18309308,
                        "f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a"),
    "cuad-v1": Source("https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1", "CUAD_v1.zip", 105883672,
                      "88b694d99007d39777fa44cd72daf8297773d285dc3eab0091ba32078888d18e"),
    "contractnli": Source("https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip",
                          "contract-nli.zip", 65362913,
                          "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"),
}
SPLITS = {"cuad": ("train", "test"), "contractnli": ("train", "dev", "test")}


@dataclass
class GoldField:
    label: str  # yes / no (CUAD: is the clause there?), or yes / no / not_mentioned (ContractNLI)
    value: object = None  # normalised value from master_clauses.csv (CUAD value fields only)
    spans: list = field(default_factory=list)  # [(start, end)] character offsets of the gold evidence


@dataclass
class GoldDoc:
    dataset: str
    id: str
    title: str
    doc_type: str
    text: str
    fields: dict  # field id -> GoldField

    def span_texts(self, field_id: str) -> list[str]:
        return [self.text[s:e] for s, e in self.fields[field_id].spans]

    def to_dict(self, with_text=False) -> dict:
        d = asdict(self)
        if not with_text:
            d.pop("text")
        return d


# --- download ---------------------------------------------------------------------------------------------------

def cache_dir() -> Path:
    """CNX_CONTRACT_CACHE, else ~/work/.cache/contract-analyst on ComputeNexa (saved with the project), else
    ~/.cache/contract-analyst."""
    if os.environ.get("CNX_CONTRACT_CACHE"):
        return Path(os.environ["CNX_CONTRACT_CACHE"]).expanduser()
    work = Path.home() / "work"
    return (work if work.is_dir() else Path.home()) / ".cache" / "contract-analyst"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(source: Source, folder: Path | None = None, quiet=False) -> Path:
    """The file in the cache, downloaded first if missing. Size and SHA-256 must match."""
    folder = folder or cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / source.filename
    if path.exists() and path.stat().st_size == source.size:
        return path
    tmp = path.with_name(path.name + ".part")
    if not quiet:
        print(f"Downloading {source.url} ({source.size / 1e6:.0f} MB) to {folder}", file=sys.stderr)
    req = urllib.request.Request(source.url, headers={"User-Agent": "contract-analyst"})
    got = 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        length = resp.headers.get("Content-Length")
        if length and int(length) != source.size:
            raise RuntimeError(f"{source.filename}: server says {length} bytes, expected {source.size}")
        while block := resp.read(1 << 20):
            got += len(block)
            if got > source.size:
                break
            out.write(block)
    if got != source.size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{source.filename}: got {got} bytes, expected {source.size} (incomplete or changed "
                           "upstream)")
    if _sha256(tmp) != source.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{source.filename}: checksum mismatch (file changed upstream?)")
    tmp.replace(path)
    return path


def _zip_member(zf: zipfile.ZipFile, basename: str) -> str:
    for name in zf.namelist():
        if name.rsplit("/", 1)[-1] == basename and "__MACOSX" not in name:
            return name
    raise KeyError(f"{basename} not found in {zf.filename}")


def _read_json(zip_path: Path, basename: str):
    with zipfile.ZipFile(zip_path) as zf, zf.open(_zip_member(zf, basename)) as f:
        return json.load(f)


# --- CUAD -------------------------------------------------------------------------------------------------------

def _title_key(name: str) -> str:
    """master_clauses.csv file names and CUAD titles differ in case, '.pdf' and punctuation."""
    name = re.sub(r"\.pdf'?$", "", name.strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parse_cuad(squad: dict, master_rows: list[dict] | None = None) -> list[GoldDoc]:
    """CUAD in SQuAD format (+ optional master_clauses.csv rows) -> gold documents with our commercial fields."""
    fields = fields_for("commercial")
    by_category = {f.source: f for f in fields}
    master = {_title_key(r.get("Filename", "")): r for r in master_rows or []}
    docs = []
    for item in squad["data"]:
        texts, spans_by_cat, offset = [], {}, 0
        for para in item["paragraphs"]:  # CUAD has one paragraph (the whole contract) per document
            ctx = para["context"]
            for qa in para["qas"]:
                category = qa["id"].split("__")[-1]
                for a in qa.get("answers", []):
                    start = a["answer_start"]
                    if ctx[start:start + len(a["text"])] != a["text"]:  # rare off-by-n offsets: search nearby
                        found = ctx.find(a["text"], max(0, start - 50))
                        start = found if found != -1 else ctx.find(a["text"])
                        if start == -1:
                            continue
                    spans_by_cat.setdefault(category, []).append((offset + start, offset + start + len(a["text"])))
            texts.append(ctx)
            offset += len(ctx) + 2
        text = "\n\n".join(texts)
        row = master.get(_title_key(item["title"]))
        gold = {}
        for cat, f in by_category.items():
            spans = sorted(set(spans_by_cat.get(cat, [])))
            value = None
            if f.value_column and row:
                raw = (row.get(f.value_column) or "").strip()
                if raw:
                    value = split_gold_parties(raw) if f.id == "parties" else raw
            gold[f.id] = GoldField("yes" if spans else "no", value, spans)
        docs.append(GoldDoc("cuad", item["title"], item["title"], "commercial", text, gold))
    return docs


def load_cuad(split="test", folder: Path | None = None) -> list[GoldDoc]:
    if split not in SPLITS["cuad"]:
        raise ValueError(f"CUAD splits: {', '.join(SPLITS['cuad'])}")
    data_zip = download(SOURCES["cuad-data"], folder)
    v1_zip = download(SOURCES["cuad-v1"], folder)
    squad = _read_json(data_zip, "CUADv1.json")
    test_titles = {d["title"] for d in _read_json(data_zip, "test.json")["data"]}
    with zipfile.ZipFile(v1_zip) as zf, zf.open(_zip_member(zf, "master_clauses.csv")) as f:
        rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
    squad["data"] = [d for d in squad["data"] if (d["title"] in test_titles) == (split == "test")]
    return sorted(parse_cuad(squad, rows), key=lambda d: d.id)


# --- ContractNLI ------------------------------------------------------------------------------------------------

def parse_contractnli(data: dict) -> list[GoldDoc]:
    """ContractNLI JSON (documents + labels) -> gold documents with our NDA fields (evidence spans = the
    annotated sentence spans)."""
    fields = [f for f in fields_for("nda") if f.evaluated]
    docs = []
    for doc in data["documents"]:
        annotations = doc["annotation_sets"][0]["annotations"]
        gold = {}
        for f in fields:
            ann = annotations.get(f.source)
            if not ann:
                continue
            spans = [tuple(doc["spans"][i]) for i in ann.get("spans", [])]
            gold[f.id] = GoldField(NLI_LABELS[ann["choice"]], None, spans)
        docs.append(GoldDoc("contractnli", f"cnli-{doc['id']}", doc.get("file_name", "").strip() or str(doc["id"]),
                            "nda", doc["text"], gold))
    return docs


def load_contractnli(split="test", folder: Path | None = None) -> list[GoldDoc]:
    if split not in SPLITS["contractnli"]:
        raise ValueError(f"ContractNLI splits: {', '.join(SPLITS['contractnli'])}")
    path = download(SOURCES["contractnli"], folder)
    docs = parse_contractnli(_read_json(path, f"{split}.json"))
    return sorted(docs, key=lambda d: (len(d.id), d.id))  # cnli-2 before cnli-10


def load(dataset: str, split="test", folder: Path | None = None) -> list[GoldDoc]:
    if dataset == "cuad":
        return load_cuad(split, folder)
    if dataset == "contractnli":
        return load_contractnli(split, folder)
    raise ValueError("dataset must be cuad or contractnli")
