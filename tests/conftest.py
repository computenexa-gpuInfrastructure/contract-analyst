import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))

from contract_analyst import datasets  # noqa: E402
from contract_analyst.fake_model import FakeModelServer  # noqa: E402


@pytest.fixture
def fake_server():
    """fake_server(responder) -> a running FakeModelServer (stopped after the test)."""
    servers = []

    def start(responder=None, **kwargs):
        srv = FakeModelServer(**({"responder": responder} if responder else {}), **kwargs).start()
        servers.append(srv)
        return srv

    yield start
    for s in servers:
        s.stop()


@pytest.fixture
def model_env(monkeypatch):
    def use(srv, key="test-key"):
        monkeypatch.setenv("CNX_API_BASE", srv.base_url)
        monkeypatch.setenv("CNX_API_KEY", key)
        monkeypatch.setenv("CNX_MODEL", srv.model)
    return use


def _zip(path: Path, members: dict) -> datasets.Source:
    with zipfile.ZipFile(path, "w") as zf:
        for name, src in members.items():
            zf.write(src, name)
    data = path.read_bytes()
    return datasets.Source(f"http://unused.invalid/{path.name}", path.name, len(data),
                           hashlib.sha256(data).hexdigest())


@pytest.fixture
def fixture_cache(tmp_path, monkeypatch):
    """A cache folder holding zips built from the fixtures, laid out like the real downloads."""
    cache = tmp_path / "cache"
    cache.mkdir()
    sources = {
        "cuad-data": _zip(cache / "cuad-data.zip", {"CUADv1.json": FIXTURES / "cuad_mini.json",
                                                    "test.json": FIXTURES / "cuad_test_mini.json"}),
        "cuad-v1": _zip(cache / "CUAD_v1.zip", {"CUAD_v1/master_clauses.csv": FIXTURES / "master_clauses_mini.csv"}),
        "contractnli": _zip(cache / "contract-nli.zip", {"contract-nli/test.json": FIXTURES / "contractnli_mini.json",
                                                         "contract-nli/train.json": FIXTURES /
                                                         "contractnli_mini.json"}),
    }
    monkeypatch.setattr(datasets, "SOURCES", sources)
    monkeypatch.setenv("CNX_CONTRACT_CACHE", str(cache))
    return cache
