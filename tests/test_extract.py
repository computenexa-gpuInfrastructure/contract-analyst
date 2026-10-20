import json

import pytest

from contract_analyst import extract, reader
from contract_analyst.extract import Analyst, merge, parse_json
from contract_analyst.fake_model import parse_prompt
from contract_analyst.fields import fields_for
from contract_analyst.textnorm import QuoteFinder

FIELDS = {f.id: f for fs in ("commercial", "nda") for f in fields_for(fs)}
TEXT = ('This Agreement is governed by the laws of the\n   State of “New York”.\n\n'
        "Neither party may assign this Agreement without consent. Payment is due in 30 days.")


# --- quote verification ---

def test_quote_found_with_offsets():
    finder = QuoteFinder(TEXT)
    start, end = finder.find("Neither party may assign this Agreement without consent.")
    assert TEXT[start:end] == "Neither party may assign this Agreement without consent."


def test_quote_found_despite_whitespace_case_and_quote_styles():
    finder = QuoteFinder(TEXT)
    span = finder.find('governed by the laws of the State of "new york"')
    assert span and TEXT[span[0]:span[1]].startswith("governed by") and TEXT[span[1] - 1] == "”"


def test_quote_with_ellipsis_needs_each_part_in_order():
    finder = QuoteFinder(TEXT)
    assert finder.find("This Agreement is governed ... Payment is due in 30 days")
    assert finder.find("Payment is due ... This Agreement is governed") is None


def test_quote_ignores_formatting_but_not_changed_words():
    text = ("Recipient shall have no obligation with respect to information that:\n(a) is public;\n(b) is known;\n"
            "\uf0b7 (c) is independently developed by Recipient; or\nlimited to its employees only,to those who "
            "need it. The term \u201cCon\ufb01dential Information\u201d means all data.")
    finder = QuoteFinder(text)
    assert finder.find("limited to its employees only, to those who need it")  # spacing
    assert finder.find("The term 'Confidential Information' means all data")  # ligature, quote style
    assert finder.find("The term Confidential Information means all data")  # quote marks dropped
    span = finder.find("information that: (c) is independently developed by Recipient")  # lead-in + one item
    assert span and text[span[0]:span[1]].startswith("information that:") and text[span[1] - 1] == "t"
    assert finder.find("information that: (c) was independently developed by Recipient") is None  # changed word
    assert finder.find("(c) is independently developed by Recipient: information that") is None  # wrong order


def test_quote_not_found():
    finder = QuoteFinder(TEXT)
    assert finder.find("governed by the laws of California") is None
    assert finder.find("") is None and finder.find("..") is None


def test_unverified_answer_is_not_counted_as_found():
    finder = QuoteFinder(TEXT)
    part = reader.chunk(TEXT, 10000)[0]
    f = FIELDS["anti_assignment"]
    good = extract.make_candidate(f, {"answer": "Yes", "quote": "Neither party may  assign this Agreement"},
                                  finder, part)
    bad = extract.make_candidate(f, {"answer": "yes", "quote": "Assignment is freely allowed."}, finder, part)
    assert good["verified"] and good["value"] == "yes"
    assert not bad["verified"]
    out = merge(f, [bad])
    assert out["status"] == "unverified" and not out["found"] and out["value"] == "no"
    assert out["model_answer"] == "yes"


# --- merging across chunks ---

def cand(value, verified, chunk, quote="q"):
    return {"value": value, "quote": quote, "start": 0 if verified else None, "end": 1 if verified else None,
            "verified": verified, "chunk": chunk}


def test_merge_prefers_verified_answers_over_earlier_unverified_ones():
    f = FIELDS["governing_law"]
    out = merge(f, [cand("Texas", False, 0), cand(None, False, 1), cand("New York", True, 2)])
    assert out["value"] == "New York" and out["status"] == "found" and out["chunk"] == 2


def test_merge_takes_the_earliest_verified_value_and_not_found_when_nothing():
    f = FIELDS["agreement_date"]
    assert merge(f, [cand("2021-03-03", True, 0), cand("2022-01-01", True, 1)])["value"] == "2021-03-03"
    none = merge(f, [cand(None, False, 0), cand(None, False, 1)])
    assert none["status"] == "not_found" and none["value"] is None and not none["found"]
    assert merge(FIELDS["exclusivity"], [])["value"] == "no"


def test_merge_nli_majority_and_parties_union():
    f = FIELDS["limited_use"]
    out = merge(f, [cand("no", True, 0), cand("yes", True, 1), cand("yes", True, 2), cand("not_mentioned", False, 3)])
    assert out["value"] == "yes" and out["chunk"] == 1
    p = merge(FIELDS["parties"], [cand(["Acme Widgets, Inc."], True, 0),
                                  cand(["ACME WIDGETS INC", "Blue River Foods LLC"], True, 1)])
    assert p["value"] == ["Acme Widgets, Inc.", "Blue River Foods LLC"]


# --- JSON parsing ---

def test_parse_json_is_robust():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": {"answer": "yes",},}\n```') == {"a": {"answer": "yes"}}
    assert parse_json('Here you go: {"a": null} Hope it helps') == {"a": None}
    with pytest.raises(ValueError):
        parse_json("no json here")
    with pytest.raises(ValueError):
        parse_json("[1, 2]")


def test_clean_values():
    assert extract.clean_value(FIELDS["limited_use"], "Entailment") == "yes"
    assert extract.clean_value(FIELDS["limited_use"], "Not mentioned") == "not_mentioned"
    assert extract.clean_value(FIELDS["limited_use"], "contradiction") == "no"
    assert extract.clean_value(FIELDS["insurance"], True) == "yes"
    assert extract.clean_value(FIELDS["insurance"], None) == "no"
    assert extract.clean_value(FIELDS["governing_law"], "N/A") is None
    assert extract.clean_value(FIELDS["parties"], "A Inc; B Ltd") == ["A Inc", "B Ltd"]


# --- against the fake server ---

LONG = ("RECITALS. " + "The parties have agreed as follows. " * 60 + "\n\n"
        + "Supplier shall maintain general liability insurance. " + "Filler sentence here. " * 60 + "\n\n"
        + "This Agreement is governed by the laws of the State of Delaware.\n")


def per_chunk_responder(body):
    """Answers only from what is in this chunk; the insurance quote is reformatted (extra spaces)."""
    ids, text = parse_prompt(body["messages"])
    out = {fid: {"answer": None, "quote": None} for fid in ids}
    if "insurance" in text:
        out["insurance"] = {"answer": "yes", "quote": "Supplier  shall maintain\ngeneral liability insurance"}
    if "Delaware" in text:
        out["governing_law"] = {"answer": "Delaware", "quote": "governed by the laws of the State of Delaware"}
    out["audit_rights"] = {"answer": "yes", "quote": "Supplier may audit Buyer at any time."}  # invented quote
    return out


def test_analyse_merges_chunks_and_verifies_quotes(fake_server):
    srv = fake_server(per_chunk_responder, api_key="k")
    analyst = Analyst(base_url=srv.base_url, api_key="k", model=srv.model, chunk_chars=1200)
    res = analyst.analyse(LONG, "commercial")
    assert res["chunks"] >= 3 and not res["errors"] and not res["failed"]
    f = res["fields"]
    assert f["insurance"]["status"] == "found" and f["insurance"]["value"] == "yes"
    assert LONG[f["insurance"]["start"]:f["insurance"]["end"]] == "Supplier shall maintain general liability insurance"
    assert f["governing_law"]["value"] == "Delaware" and f["governing_law"]["chunk"] == res["chunks"] - 1
    assert f["audit_rights"]["status"] == "unverified" and f["audit_rights"]["value"] == "no"
    assert f["exclusivity"]["status"] == "not_found"
    assert res["disclaimer"].startswith("Demo / case study output, not legal advice")
    req = srv.requests[0]
    assert req["response_format"] == {"type": "json_object"} and req["temperature"] == 0


def test_chunk_size_comes_from_served_context(fake_server):
    srv = fake_server(per_chunk_responder, max_model_len=10000)
    analyst = Analyst(base_url=srv.base_url, api_key="k", model=srv.model)
    assert analyst.chunk_chars == reader.chunk_chars_for(10000) and analyst.context_tokens == 10000
    srv2 = fake_server(per_chunk_responder, max_model_len=None)
    assert Analyst(base_url=srv2.base_url, api_key="k", model=srv2.model).chunk_chars == reader.MAX_CHUNK_CHARS


def test_bad_json_is_retried_and_one_failing_chunk_does_not_lose_the_rest(fake_server):
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        text = parse_prompt(body["messages"])[1]
        if "Delaware" in text and "insurance" not in text:
            return 500, {"error": {"message": "boom"}}  # this chunk always fails
        if calls["n"] == 1:
            return "Sorry, here is the answer: not JSON"
        return per_chunk_responder(body)

    srv = fake_server(flaky)
    analyst = Analyst(base_url=srv.base_url, api_key="k", model=srv.model, chunk_chars=1200, retries=1)
    res = analyst.analyse(LONG, "commercial")
    assert res["errors"] and not res["failed"]
    assert res["fields"]["insurance"]["status"] == "found"
    assert any(m["role"] == "assistant" for m in srv.requests[1]["messages"])  # the retry shows the bad reply


def test_falls_back_without_json_mode_and_splits_a_chunk_that_is_too_long(fake_server):
    def picky(body):
        if "response_format" in body:
            return 400, {"error": {"message": "response_format json_object is not supported"}}
        text = parse_prompt(body["messages"])[1]
        if len(text) > 700:
            return 400, {"error": {"message": "This model's maximum context length is 512 tokens"}}
        return "```json\n" + json.dumps(per_chunk_responder(body)) + "\n```"

    srv = fake_server(picky)
    analyst = Analyst(base_url=srv.base_url, api_key="k", model=srv.model, chunk_chars=1200, retries=0)
    res = analyst.analyse(LONG, "commercial")
    assert not analyst.json_mode and not res["errors"]
    assert res["fields"]["governing_law"]["value"] == "Delaware"


def test_models_that_refuse_temperature_are_asked_without_it(fake_server):
    seen = []

    def no_temperature(body):
        seen.append("temperature" in body)
        if "temperature" in body:
            return 400, {"error": {"message": "`temperature` is deprecated for this model."}}
        return json.dumps(per_chunk_responder(body))

    srv = fake_server(no_temperature)
    analyst = Analyst(base_url=srv.base_url, api_key="k", model=srv.model, chunk_chars=100000, retries=0)
    res = analyst.analyse(LONG, "commercial")
    assert not res["errors"] and res["fields"]["governing_law"]["value"] == "Delaware"
    assert analyst.temperature is False and seen == [True, False]  # asked once more without it, then remembered
    analyst.analyse(LONG, "commercial")
    assert seen[2:] == [False]


def test_settings_switched_off_by_another_thread_dont_fail_a_request_in_flight(fake_server):
    import threading

    gate = threading.Barrier(2)

    def picky(body):
        if "temperature" in body or "response_format" in body:
            gate.wait(timeout=5)  # both first requests are refused at the same time
            return 400, {"error": {"message": "`temperature` is deprecated for this model."
                                   if "temperature" in body else "response_format.type: Input should be 'json_schema'"}}
        return json.dumps(per_chunk_responder(body))

    srv = fake_server(picky)
    analyst = Analyst(base_url=srv.base_url, api_key="k", model=srv.model, chunk_chars=100000, retries=0)
    out = [None, None]
    threads = [threading.Thread(target=lambda n=n: out.__setitem__(n, analyst.analyse(LONG, "commercial"))) for n in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(not r["errors"] for r in out), [r["errors"] for r in out]
    assert analyst.temperature is False and analyst.json_mode is False


def test_everything_failing_marks_the_document_failed(fake_server):
    srv = fake_server(lambda body: (500, {"error": {"message": "down"}}))
    res = Analyst(base_url=srv.base_url, api_key="k", model=srv.model, retries=0).analyse("A short NDA.", "nda")
    assert res["failed"] and len(res["errors"]) == 1
    assert res["fields"]["limited_use"]["value"] == "not_mentioned"


def test_guess_doc_type():
    assert extract.guess_doc_type("MUTUAL NON-DISCLOSURE AGREEMENT\n...") == "nda"
    assert extract.guess_doc_type("MASTER SUPPLY AGREEMENT") == "commercial"


def test_null_quote_text_is_no_quote():
    finder = QuoteFinder(TEXT)
    part = reader.chunk(TEXT, 10000)[0]
    c = extract.make_candidate(FIELDS["limited_use"], {"answer": "no", "quote": "null"}, finder, part)
    assert c["quote"] is None and not c["verified"]


def test_reverify_with_and_without_saved_candidates():
    text = "Recipient will use the information only for the Purpose."
    f = FIELDS["limited_use"]
    good = {"value": "yes", "quote": "will use the information only for the Purpose", "chunk": 0, "near": 0,
            "verified": False}
    saved = {"doc_type": "nda", "fields": {f.id: merge(f, [good])}, "candidates": {f.id: [good]}}
    assert saved["fields"][f.id]["status"] == "unverified"
    assert extract.reverify(saved, text)["fields"][f.id]["status"] == "found"

    old = {"doc_type": "nda", "fields": {f.id: merge(f, [good])}}  # saved before candidates were kept
    r = extract.reverify(old, text)["fields"][f.id]
    assert r["status"] == "found" and r["value"] == "yes"
    other = {"doc_type": "nda", "fields": {f.id: merge(f, [dict(good, quote="use it for anything")])}}
    assert extract.reverify(other, text)["fields"][f.id]["status"] == "unverified"
    assert extract.reverify({"failed": True, "fields": {}}, text) == {"failed": True, "fields": {}}


def test_guided_instructions_add_the_reading_rules():
    nda = fields_for("nda")
    part = reader.chunk("Some NDA text.", 10000)[0]
    plain = extract.build_messages(nda, "nda", part, 1)
    guided = extract.build_messages(nda, "nda", part, 1, "guided")
    hint = FIELDS["return_or_destroy"].hint
    assert hint and hint not in plain[1]["content"] and "How to decide" not in plain[1]["content"]
    assert hint in guided[1]["content"] and extract.GUIDED_NLI in guided[0]["content"]
    assert extract.GUIDED_QUOTES in guided[0]["content"] and extract.GUIDED_QUOTES not in plain[0]["content"]
    assert extract.GUIDED_NLI not in extract.build_messages(fields_for("commercial"), "commercial", part, 1,
                                                            "guided")[0]["content"]
    assert all(f.hint for f in nda if f.evaluated)
    with pytest.raises(ValueError):
        Analyst(client=object(), model="m", instructions="clever")
