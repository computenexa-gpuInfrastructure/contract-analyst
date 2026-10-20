"""End to end: the CLI against the fake OpenAI-compatible server."""
import csv
import json
import os
import subprocess
import sys

import pytest

from conftest import FIXTURES, ROOT

from contract_analyst import cli
from contract_analyst.fake_model import keyword_responder, parse_prompt


def test_extract_end_to_end_as_a_subprocess(fake_server, tmp_path):
    srv = fake_server(api_key="secret-key")  # default keyword responder
    env = dict(os.environ, CNX_API_BASE=srv.base_url, CNX_API_KEY="secret-key", CNX_MODEL=srv.model,
               PYTHONPATH=str(ROOT))
    html = tmp_path / "out.html"
    proc = subprocess.run([sys.executable, "-m", "contract_analyst", "extract", str(FIXTURES / "sample_contract.txt"),
                           "--type", "commercial", "--html", str(html)], env=env, capture_output=True, text=True,
                          timeout=60, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    f = out["fields"]
    assert out["doc_type"] == "commercial" and out["source"] == "sample_contract.txt"
    assert f["governing_law"]["value"] == "New York" and f["governing_law"]["status"] == "found"
    assert f["exclusivity"]["value"] == "yes" and f["insurance"]["value"] == "no"
    assert f["agreement_date"]["value"] == "March 3, 2021"
    assert "not legal advice" in out["disclaimer"]
    page = html.read_text()
    assert "Governing law" in page and "not legal advice" in page and "CC BY 4.0" in page
    assert "The Atticus Project" in page and "Koreeda" in page


def test_extract_with_a_wrong_key_reports_failure(fake_server, tmp_path, model_env, capsys):
    srv = fake_server(api_key="right")
    model_env(srv, key="wrong")
    code = cli.main(["extract", str(FIXTURES / "sample_contract.txt"), "--html", str(tmp_path / "x.html")])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["failed"]


def test_evaluate_end_to_end(fake_server, fixture_cache, model_env, tmp_path, capsys):
    srv = fake_server()
    model_env(srv)
    out = tmp_path / "results"
    assert cli.main(["evaluate", "--dataset", "cuad", "--split", "test", "--limit", "5", "--out", str(out),
                     "--price-per-hour", "1.12", "--workers", "2"]) == 0
    res = json.loads((out / "results.json").read_text())
    assert res["meta"]["dataset"] == "cuad" and res["summary"]["documents"] == 1
    assert res["summary"]["cost_per_document"] is not None and "not legal advice" in res["meta"]["disclaimer"]
    gl = next(m for m in res["per_field"] if m["field"] == "governing_law")
    assert gl["found"]["f1"] == 1.0 and gl["value_accuracy"] == 1.0  # "Ontario" == "Ontario, Canada"
    ins = next(m for m in res["per_field"] if m["field"] == "insurance")
    assert ins["accuracy"] == 1.0 and ins["quote_f1"] > 0.9
    rows = list(csv.DictReader(open(out / "per_field.csv")))
    assert len(rows) == 19
    assert (out / "predictions.csv").exists() and (out / "extractions.json").exists()
    page = (out / "report.html").read_text()
    assert "Per document" in page and "Northwind_Services" in page and "CC BY 4.0" in page
    assert "MACRO" in capsys.readouterr().out

    again = cli.main(["rescore", str(out), "--out", str(tmp_path / "again")])
    assert again == 0 and "Before -> after" in capsys.readouterr().out
    res_again = json.loads((tmp_path / "again" / "results.json").read_text())
    assert res_again["summary"]["accuracy_macro"] == res["summary"]["accuracy_macro"]  # same answers, same score
    assert res_again["summary"]["cost_per_document"] == pytest.approx(res["summary"]["cost_per_document"])
    assert res_again["meta"]["answers_created"] == res["meta"]["created"]
    assert "candidates" in json.loads((out / "extractions.json").read_text())["Northwind_Services"]

    out2 = tmp_path / "nli"
    assert cli.main(["evaluate", "--dataset", "contractnli", "--out", str(out2)]) == 0
    res2 = json.loads((out2 / "results.json").read_text())
    assert res2["summary"]["documents"] == 2 and len(res2["per_field"]) == 17
    assert next(m for m in res2["per_field"] if m["field"] == "no_reverse_engineering")["accuracy"] == 1.0


def test_evaluate_survives_a_document_that_breaks(fake_server, fixture_cache, model_env, tmp_path):
    def breaks_on_nda_2(body):
        if "reverse engineer" in parse_prompt(body["messages"])[1]:
            return 500, {"error": {"message": "GPU on fire"}}
        return keyword_responder(body)

    srv = fake_server(breaks_on_nda_2)
    model_env(srv)
    assert cli.main(["evaluate", "--dataset", "contractnli", "--out", str(tmp_path / "r"), "--timeout", "5",
                     "--retries", "0"]) == 0
    res = json.loads((tmp_path / "r" / "results.json").read_text())
    assert res["summary"]["documents"] == 2 and res["summary"]["failed_documents"] == 1
    assert next(m for m in res["per_field"] if m["field"] == "limited_use")["found"]["tp"] == 1  # NDA 1 scored


def test_evaluate_guided_with_skip_and_rescore(fake_server, fixture_cache, model_env, tmp_path, capsys):
    srv = fake_server()
    model_env(srv)
    out = tmp_path / "guided"
    assert cli.main(["evaluate", "--dataset", "contractnli", "--skip", "1", "--instructions", "guided",
                     "--out", str(out)]) == 0
    res = json.loads((out / "results.json").read_text())
    assert res["meta"]["instructions"] == "guided" and res["meta"]["skip"] == 1
    assert res["summary"]["documents"] == 1 and {r["doc_id"] for r in res["rows"]} == {"cnli-2"}
    assert "guided instructions" in (out / "report.html").read_text()
    assert json.loads((out / "extractions.json").read_text())["cnli-2"]["instructions"] == "guided"
    assert cli.main(["rescore", str(out)]) == 0  # same documents: the skip is kept
    again = json.loads((out / "rescored" / "results.json").read_text())
    assert again["summary"]["documents"] == 1 and again["summary"]["accuracy_macro"] == res["summary"]["accuracy_macro"]


def test_retry_reanalyses_only_failed_documents(fake_server, fixture_cache, model_env, tmp_path, capsys):
    def breaks_on_nda_2(body):
        if "reverse engineer" in parse_prompt(body["messages"])[1]:
            return 500, {"error": {"message": "GPU on fire"}}
        return keyword_responder(body)

    out = tmp_path / "r"
    model_env(fake_server(breaks_on_nda_2))
    assert cli.main(["evaluate", "--dataset", "contractnli", "--out", str(out), "--timeout", "5", "--retries", "0"]) == 0
    first = json.loads((out / "results.json").read_text())
    assert first["summary"]["failed_documents"] == 1

    healed = fake_server()  # the same model name, now answering
    model_env(healed)
    assert cli.main(["retry", str(out), "--retries", "0"]) == 0
    again = json.loads((out / "results.json").read_text())
    assert again["summary"]["failed_documents"] == 0 and again["meta"]["retried"] == ["cnli-2"]
    assert again["summary"]["wall_seconds"] == first["summary"]["wall_seconds"]  # a repair, not a new measurement
    assert "failed documents 1 -> 0" in capsys.readouterr().out
    assert cli.main(["retry", str(out)]) == 0  # nothing left
