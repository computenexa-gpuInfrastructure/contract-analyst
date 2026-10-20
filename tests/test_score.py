import pytest

from contract_analyst.datasets import GoldDoc, GoldField
from contract_analyst.fields import fields_for
from contract_analyst.score import prf, quote_f1, score, value_match
from contract_analyst.textnorm import norm_date, norm_duration, norm_law, parties_match, token_f1

F = {f.id: f for f in fields_for("commercial")}


def test_prf_maths():
    m = prf(tp=3, fp=1, fn=2)
    assert m["precision"] == 0.75 and m["recall"] == 0.6
    assert m["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert prf(0, 0, 4)["f1"] == 0.0 and prf(0, 0, 4)["precision"] is None
    assert prf(0, 0, 0)["f1"] is None


def test_token_f1_and_quote_f1():
    assert token_f1("the cat sat", "the cat sat") == 1.0
    assert token_f1("the cat", "the cat sat down") == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert token_f1("dog", "cat") == 0.0
    # best gold span wins; evidence split over two sentences also counts when joined
    assert quote_f1("governed by New York law", ["unrelated text", "governed by New York law"]) == 1.0
    assert quote_f1("a b c d", ["a b", "c d"]) == 1.0
    assert quote_f1(None, ["x"]) is None and quote_f1("x", []) is None


def test_value_normalisation():
    assert norm_date("5/8/14") == norm_date("May 8, 2014") == norm_date("2014-05-08") == "2014-05-08"
    assert norm_date("8th day of May, 1999") == "1999-05-08"
    assert norm_duration("twelve (12) months") == norm_duration("1 year") == "365"
    assert norm_duration("sixty (60) days") == norm_duration("60 days") == "60"
    assert norm_law("the laws of the State of New York") == norm_law("New York") == "new york"
    assert norm_law("Ontario, Canada") == norm_law("Province of Ontario") == "ontario"
    assert parties_match(["ACME WIDGETS INC", "Blue River Foods"], ['Acme Widgets, Inc. ("Supplier")',
                                                                     'Blue River Foods LLC ("Buyer")'])
    assert not parties_match(["Acme Widgets"], ["Acme Widgets, Inc.", "Blue River Foods LLC"])


def test_value_match_by_type():
    assert value_match(F["agreement_date"], "2021-03-03", "3/3/21")
    assert not value_match(F["agreement_date"], "2021-03-04", "3/3/21")
    assert value_match(F["renewal_term"], "1 year", "successive 1 year")
    assert value_match(F["governing_law"], "New York", "New York")
    assert value_match(F["expiration_date"], "perpetual", "Perpetual")
    assert not value_match(F["governing_law"], None, "New York")


def gold(**fields):
    base = {f.id: GoldField("no") for f in fields_for("commercial")}
    base.update(fields)
    return GoldDoc("cuad", "d", "d", "commercial", "TEXT governed by New York law. Licensee shall carry insurance.",
                   base)


def pred(**fields):
    out = {}
    for fid, (value, status, quote) in fields.items():
        out[fid] = {"value": value, "status": status, "found": status == "found", "quote": quote}
    return {"fields": out, "seconds": 2.0, "chunks": 1, "errors": []}


def test_score_known_case():
    g1 = gold(governing_law=GoldField("yes", "New York", [(5, 30)]), insurance=GoldField("yes", None, [(31, 63)]))
    g2 = gold(governing_law=GoldField("yes", "Texas", [(5, 30)]))
    g2.id = "e"
    preds = {
        "d": pred(governing_law=("New York", "found", "governed by New York law."),
                  insurance=("yes", "found", "Licensee shall carry insurance."),
                  exclusivity=("yes", "found", "TEXT")),  # false positive
        "e": pred(governing_law=("Texas", "unverified", "invented quote")),  # unverified: not found
    }
    r = score([g1, g2], preds, wall_seconds=10.0, price_per_hour=1.8)
    pf = {m["field"]: m for m in r["per_field"]}
    gl = pf["governing_law"]
    assert gl["found"]["tp"] == 1 and gl["found"]["fn"] == 1 and gl["found"]["recall"] == 0.5
    assert gl["value_accuracy"] == 0.5 and gl["value_n"] == 2
    assert gl["quote_f1"] == 1.0 and gl["quote_n"] == 1 and gl["unverified"] == 1
    ins = pf["insurance"]
    assert ins["accuracy"] == 1.0 and ins["per_class"]["yes"]["f1"] == 1.0 and ins["per_class"]["no"]["f1"] == 1.0
    ex = pf["exclusivity"]
    assert ex["accuracy"] == 0.5 and ex["found"]["precision"] == 0.0 and ex["per_class"]["no"]["recall"] == 0.5
    s = r["summary"]
    assert s["documents"] == 2 and s["seconds_per_document"] == 2.0 and s["wall_seconds_per_document"] == 5.0
    assert s["cost_per_document"] == pytest.approx(1.8 * 10 / 3600 / 2)
    assert s["unverified_answers"] == 1


def test_score_nli_classes_and_failed_documents():
    nda = [f for f in fields_for("nda") if f.evaluated]
    golds = []
    for i, label in enumerate(["yes", "no", "not_mentioned", "yes"]):
        fields = {f.id: GoldField("not_mentioned") for f in nda}
        fields["limited_use"] = GoldField(label, None, [(0, 4)] if label != "not_mentioned" else [])
        golds.append(GoldDoc("contractnli", f"n{i}", "", "nda", "text of the nda", fields))
    preds = {"n0": {"fields": {"limited_use": {"value": "yes", "found": True, "status": "found", "quote": "text"}},
                    "seconds": 1, "chunks": 1, "errors": []},
             "n1": {"fields": {"limited_use": {"value": "yes", "found": True, "status": "found", "quote": "text"}},
                    "seconds": 1, "chunks": 1, "errors": []},
             "n2": {"fields": {"limited_use": {"value": "not_mentioned", "found": False, "status": "not_found"}},
                    "seconds": 1, "chunks": 1, "errors": []},
             "n3": None}  # this document failed
    r = score(golds, preds)
    lu = next(m for m in r["per_field"] if m["field"] == "limited_use")
    assert lu["accuracy"] == 0.5
    assert lu["per_class"]["yes"]["precision"] == 0.5 and lu["per_class"]["yes"]["recall"] == 0.5
    assert lu["per_class"]["no"]["f1"] == 0.0
    assert lu["per_class"]["not_mentioned"]["precision"] == 0.5  # n2 right, failed n3 counted as not_mentioned
    assert r["summary"]["failed_documents"] == 1 and r["summary"]["cost_per_document"] is None
