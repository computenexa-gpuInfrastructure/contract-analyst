import csv
import hashlib
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import FIXTURES

from contract_analyst import datasets


def load_fixture_cuad():
    squad = json.loads((FIXTURES / "cuad_mini.json").read_text())
    rows = list(csv.DictReader(open(FIXTURES / "master_clauses_mini.csv")))
    return {d.id: d for d in datasets.parse_cuad(squad, rows)}


def test_parse_cuad_labels_spans_and_values():
    docs = load_fixture_cuad()
    acme = docs["AcmeWidgets_MSA"]
    assert acme.doc_type == "commercial" and acme.text.startswith("MASTER SUPPLY AGREEMENT")
    assert acme.fields["exclusivity"].label == "yes"
    assert acme.span_texts("exclusivity") == ["Buyer shall purchase the Products exclusively from Supplier during "
                                              "the Term."]
    assert acme.fields["insurance"].label == "no" and acme.fields["insurance"].spans == []
    assert acme.fields["governing_law"].value == "New York"
    assert acme.fields["parties"].value == ['Acme Widgets, Inc. ("Supplier")', 'Blue River Foods LLC ("Buyer")']
    assert acme.fields["renewal_term"].value == "successive 1 year"
    assert acme.fields["cap_on_liability"].value is None  # yes/no fields have no value
    assert "document_name" not in acme.fields  # categories outside our field list are ignored
    north = docs["Northwind_Services"]  # master_clauses name differs in case and ends in .PDF'
    assert north.fields["governing_law"].value == "Ontario, Canada"
    assert north.fields["effective_date"].value is None and north.fields["effective_date"].label == "no"


def test_parse_contractnli_labels_and_evidence():
    data = json.loads((FIXTURES / "contractnli_mini.json").read_text())
    docs = {d.id: d for d in datasets.parse_contractnli(data)}
    one, two = docs["cnli-1"], docs["cnli-2"]
    assert one.doc_type == "nda" and one.title == "nda-1.pdf"
    assert len(one.fields) == 17  # synthetic-later fields are not in the gold
    assert one.fields["verbal_info_included"].label == "yes"
    assert one.span_texts("verbal_info_included") == ["Confidential Information includes information disclosed "
                                                      "orally or in writing."]
    assert one.fields["non_solicit"].label == "not_mentioned" and one.fields["non_solicit"].spans == []
    assert two.fields["verbal_info_included"].label == "no"
    assert two.fields["must_be_marked"].label == "yes"


def test_load_uses_the_official_splits(fixture_cache):
    assert [d.id for d in datasets.load("cuad", "test")] == ["Northwind_Services"]
    assert [d.id for d in datasets.load("cuad", "train")] == ["AcmeWidgets_MSA"]
    assert [d.id for d in datasets.load("contractnli", "test")] == ["cnli-1", "cnli-2"]
    with pytest.raises(ValueError):
        datasets.load("cuad", "dev")


@pytest.fixture
def file_server(tmp_path):
    (tmp_path / "data.zip").write_bytes(b"x" * 1000)
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(tmp_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/data.zip"
    httpd.shutdown()


def test_download_checks_size_and_checksum(file_server, tmp_path):
    good = datasets.Source(file_server, "d.zip", 1000, hashlib.sha256(b"x" * 1000).hexdigest())
    cache = tmp_path / "cache"
    path = datasets.download(good, cache, quiet=True)
    assert path.read_bytes() == b"x" * 1000
    assert datasets.download(good, cache, quiet=True) == path  # cached: no second download

    with pytest.raises(RuntimeError, match="expected 999"):
        datasets.download(datasets.Source(file_server, "small.zip", 999, good.sha256), cache, quiet=True)
    with pytest.raises(RuntimeError, match="checksum"):
        datasets.download(datasets.Source(file_server, "bad.zip", 1000, "0" * 64), cache, quiet=True)
    assert not (cache / "bad.zip").exists() and not (cache / "bad.zip.part").exists()


def test_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CNX_CONTRACT_CACHE", str(tmp_path / "c"))
    assert datasets.cache_dir() == tmp_path / "c"
    monkeypatch.delenv("CNX_CONTRACT_CACHE")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert datasets.cache_dir() == tmp_path / ".cache" / "contract-analyst"
    (tmp_path / "work").mkdir()  # on ComputeNexa: inside the project folder
    assert datasets.cache_dir() == tmp_path / "work" / ".cache" / "contract-analyst"
