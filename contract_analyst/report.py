"""Write results as JSON, CSV and a standalone HTML page (no scripts, no external files)."""
import csv
import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from .extract import DISCLAIMER
from .fields import fields_for

CREDITS = [
    ("CUAD v1", "The Atticus Project", "https://www.atticusprojectai.org/cuad",
     "Hendrycks, Burns, Chen and Ball, \"CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review\", "
     "NeurIPS 2021 Datasets and Benchmarks"),
    ("ContractNLI", "Yuta Koreeda and Christopher D. Manning", "https://stanfordnlp.github.io/contract-nli/",
     "Koreeda and Manning, \"ContractNLI: A Dataset for Document-level Natural Language Inference for "
     "Contracts\", Findings of EMNLP 2021"),
]

CSS = """
:root { --bg:#ffffff; --fg:#1d2330; --muted:#5b6475; --line:#e3e6ec; --card:#f6f7f9;
  --good:#e3f4e8; --good-fg:#17663a; --mid:#fdf3dc; --mid-fg:#8a5a00; --bad:#fbe5e5; --bad-fg:#a12828; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171d; --fg:#e6e9ef; --muted:#9aa3b2; --line:#2a2f39;
  --card:#1c2028; --good:#173823; --good-fg:#8fdba8; --mid:#3a2f12; --mid-fg:#f0c66b; --bad:#3d1c1c;
  --bad-fg:#f2a0a0; } }
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1100px; margin:0 auto; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:28px 0 8px; }
.muted { color:var(--muted); } .small { font-size:13px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:16px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.card b { display:block; font-size:20px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:600; }
td.num { text-align:right; white-space:nowrap; }
.good { background:var(--good); color:var(--good-fg); } .mid { background:var(--mid); color:var(--mid-fg); }
.bad { background:var(--bad); color:var(--bad-fg); }
.badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; white-space:nowrap; }
q, .quote { font-style:italic; } blockquote { margin:4px 0; padding-left:8px; border-left:3px solid var(--line); }
details { border:1px solid var(--line); border-radius:8px; margin:6px 0; padding:6px 10px; }
summary { cursor:pointer; }
footer { margin-top:32px; padding-top:12px; border-top:1px solid var(--line); color:var(--muted); font-size:13px; }
"""


def _fmt(x, pct=True):
    if x is None:
        return "–"
    return f"{x * 100:.0f}%" if pct else f"{x:.2f}"


def _cls(x):
    if x is None:
        return ""
    return "good" if x >= 0.8 else "mid" if x >= 0.5 else "bad"


def _cell(x):
    return f'<td class="num {_cls(x)}">{_fmt(x)}</td>'


def _value(v) -> str:
    if v in (None, "", []):
        return '<span class="muted">–</span>'
    if isinstance(v, list):
        return escape("; ".join(map(str, v)))
    return escape(str(v))


def _page(title: str, body: str) -> str:
    credits = "".join(f'<li><a href="{u}">{escape(n)}</a> by {escape(a)}, licensed CC BY 4.0. {escape(c)}.</li>'
                      for n, a, u, c in CREDITS)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(title)}</title>'
            f"<style>{CSS}</style></head><body><main>{body}<footer><p><b>{escape(DISCLAIMER)}</b></p>"
            f"<p>Private Contract Analyst: an open template that runs on your own ComputeNexa GPU. "
            f"Evaluation data:</p><ul>{credits}</ul></footer></main></body></html>")


STATUS = {"found": ("good", "verified"), "unverified": ("mid", "quote not found"), "not_found": ("", "not found")}


def extraction_html(result: dict, source_name: str) -> str:
    """One contract: every field with its answer, status and quote."""
    rows = []
    for f in fields_for(result["doc_type"]):
        r = result["fields"][f.id]
        css, text = STATUS.get(r["status"], ("bad", r["status"]))
        answer = _value(r["value"])
        if r["status"] == "unverified":
            answer = f'<span class="muted">model said: {_value(r.get("model_answer"))}</span>'
        quote = f'<blockquote class="quote">{escape(r["quote"])}</blockquote>' if r.get("quote") else ""
        note = f'<br><span class="muted small">{escape(f.note)}</span>' if f.note else ""
        rows.append(f"<tr><td>{escape(f.label)}{note}</td><td>{answer}</td>"
                    f'<td><span class="badge {css}">{text}</span></td><td>{quote}</td></tr>')
    errors = "".join(f"<li>{escape(e)}</li>" for e in result.get("errors", []))
    body = (f"<h1>{escape(source_name)}</h1><p class='muted'>{escape(result['doc_type'])} contract · model "
            f"{escape(result.get('model') or '?')} · {result['chunks']} part(s) · {result['seconds']:.1f} s</p>"
            + (f"<p class='bad'>Some parts failed:</p><ul>{errors}</ul>" if errors else "")
            + "<div class='scroll'><table><tr><th>Field</th><th>Answer</th><th>Status</th><th>Quote</th></tr>"
            + "".join(rows) + "</table></div>")
    return _page(f"Contract analysis: {source_name}", body)


def evaluation_html(report: dict, meta: dict) -> str:
    s = report["summary"]
    cost = f"${s['cost_per_document']:.4f}" if s.get("cost_per_document") is not None else "–"
    cards = [("Found F1", _fmt(s["found_f1_macro"])), ("Accuracy (yes/no)", _fmt(s["accuracy_macro"])),
             ("Value match", _fmt(s["value_accuracy_macro"])), ("Quote F1", _fmt(s["quote_f1_macro"])),
             ("Seconds / document", _fmt(s["wall_seconds_per_document"], pct=False)), ("Cost / document", cost)]
    head = (f"<h1>Contract Analyst evaluation</h1><p class='muted'>{escape(meta.get('dataset', ''))} "
            f"{escape(meta.get('split', ''))} · {s['documents']} documents ({s['failed_documents']} failed) · model "
            f"{escape(meta.get('model') or '?')} · {escape(meta.get('instructions') or 'plain')} instructions · "
            f"{escape(meta.get('created', ''))}</p>"
            "<div class='cards'>" + "".join(f"<div class='card'><span class='muted small'>{k}</span><b>{v}</b>"
                                            f"</div>" for k, v in cards) + "</div>")
    ft = ["<h2>Per field</h2><div class='scroll'><table><tr><th>Field</th><th>Type</th><th>Docs</th>"
          "<th>Gold found</th><th>Found P</th><th>Found R</th><th>Found F1</th><th>Accuracy</th>"
          "<th>Value match</th><th>Quote F1</th><th>Unverified</th></tr>"]
    for m in report["per_field"]:
        fd = m["found"]
        ft.append(f"<tr><td>{escape(m['label'])}</td><td>{m['type']}</td><td class='num'>{m['n']}</td>"
                  f"<td class='num'>{m['gold_found']}</td>{_cell(fd['precision'])}{_cell(fd['recall'])}"
                  f"{_cell(fd['f1'])}{_cell(m.get('accuracy'))}{_cell(m.get('value_accuracy'))}"
                  f"{_cell(m['quote_f1'])}<td class='num'>{m['unverified']}</td></tr>")
    ft.append("</table></div>")
    docs, by_doc = [], {}
    for r in report["rows"]:
        by_doc.setdefault(r["doc_id"], []).append(r)
    for doc_id, rows in by_doc.items():
        ok = sum(r["correct"] for r in rows)
        trs = []
        for r in rows:
            css = "good" if r["correct"] and r["value_correct"] is not False else "bad"
            gold = escape(r["gold_label"]) + (f"<br>{_value(r['gold_value'])}" if r["gold_value"] else "")
            pred = escape(r["pred_label"])
            if r["pred_value"] not in (None, "", [], "yes", "no", "not_mentioned"):
                pred += f"<br>{_value(r['pred_value'])}"
            if r["status"] == "unverified":
                pred += f"<br><span class='muted small'>model said {_value(r['model_answer'])}; quote not found"
                pred += "</span>"
            quote = f"<blockquote>{escape(r['quote'])}</blockquote>" if r["quote"] else ""
            if r["quote_f1"] is not None:
                quote += f"<span class='muted small'>quote F1 {_fmt(r['quote_f1'])}</span>"
            ev = "".join(f"<blockquote>{escape(t[:400])}{'…' if len(t) > 400 else ''}</blockquote>"
                         for t in r["gold_spans"][:3])
            trs.append(f"<tr><td>{escape(r['field'])}</td><td class='{css}'>{gold}</td><td class='{css}'>{pred}"
                       f"</td><td>{quote}</td><td>{ev}</td></tr>")
        docs.append(f"<details><summary>{escape(doc_id)} <span class='muted small'>{ok}/{len(rows)} "
                    f"correct</span></summary><div class='scroll'><table><tr><th>Field</th><th>Gold</th>"
                    f"<th>Prediction</th><th>Our quote</th><th>Gold evidence</th></tr>{''.join(trs)}</table>"
                    f"</div></details>")
    body = head + "".join(ft) + "<h2>Per document</h2>" + "".join(docs)
    return _page("Contract Analyst evaluation", body)


PER_FIELD_COLUMNS = ["field", "type", "n", "gold_found", "found_precision", "found_recall", "found_f1", "accuracy",
                     "value_accuracy", "value_n", "quote_f1", "quote_n", "unverified"]
ROW_COLUMNS = ["doc_id", "field", "gold_label", "pred_label", "correct", "status", "gold_value", "pred_value",
               "value_correct", "quote_f1", "quote"]


def _csv_value(v):
    if isinstance(v, list):
        return "; ".join(map(str, v))
    return "" if v is None else v


def write_evaluation(report: dict, meta: dict, out: Path) -> dict:
    """results.json, per_field.csv, predictions.csv and report.html in `out`."""
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, created=meta.get("created") or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
                disclaimer=DISCLAIMER)
    (out / "results.json").write_text(json.dumps({"meta": meta, **report}, indent=1, default=str))
    with open(out / "per_field.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PER_FIELD_COLUMNS)
        for m in report["per_field"]:
            fd = m["found"]
            w.writerow([m["field"], m["type"], m["n"], m["gold_found"], fd["precision"], fd["recall"], fd["f1"],
                        m.get("accuracy"), m.get("value_accuracy"), m.get("value_n"), m["quote_f1"], m["quote_n"],
                        m["unverified"]])
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ROW_COLUMNS)
        for r in report["rows"]:
            w.writerow([_csv_value(r[c]) for c in ROW_COLUMNS])
    (out / "report.html").write_text(evaluation_html(report, meta))
    return meta
