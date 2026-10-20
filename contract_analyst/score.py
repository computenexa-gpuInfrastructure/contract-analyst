"""Score predictions against gold documents.

Per field:
  - found: precision / recall / F1 of "the clause is there" (a value, a "yes", or for NDAs yes / no).
  - yes/no and NDA fields: accuracy and F1 per class (yes / no / not_mentioned).
  - values (dates, durations, governing law, parties): normalised exact match, over documents with a gold value.
  - quotes: token-overlap F1 between our quote and the best-matching gold span, over documents where both exist.
Plus timings (seconds per document) and an estimated cost per document = price per hour x wall time / documents.
"""
from .datasets import GoldDoc
from .fields import Field, fields_for, is_found
from .textnorm import norm_date, norm_duration, norm_law, normalise, parties_match, token_f1

CLASSES = {"yesno": ("yes", "no"), "yesno_quote": ("yes", "no"), "nli": ("yes", "no", "not_mentioned")}


def value_match(field: Field, pred, gold) -> bool:
    """Normalised exact match of a value (see textnorm for the rules)."""
    if gold in (None, "", []) or pred in (None, "", []):
        return False
    if field.id == "parties":
        return parties_match(pred if isinstance(pred, list) else [str(pred)], gold)
    norm = {"date": norm_date, "duration": norm_duration}.get(field.type)
    if field.id == "governing_law":
        norm = norm_law
    if norm:
        p, g = norm(pred), norm(gold)
        if p is not None and g is not None:
            return p == g
    return normalise(str(pred)).strip(" .") == normalise(str(gold)).strip(" .")


def quote_f1(quote: str | None, gold_spans: list[str]) -> float | None:
    """Best token F1 against any gold span (or all spans together, for evidence split over sentences)."""
    if not quote or not gold_spans:
        return None
    candidates = gold_spans + ([" ".join(gold_spans)] if len(gold_spans) > 1 else [])
    return max(token_f1(quote, g) for g in candidates)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None  # None = nothing to find and nothing found
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def score(golds: list[GoldDoc], preds: dict, wall_seconds: float | None = None,
          price_per_hour: float | None = None) -> dict:
    """golds + {doc id: extraction result (from Analyst.analyse) or None if it failed} -> report dict."""
    if not golds:
        raise ValueError("nothing to score")
    doc_type = golds[0].doc_type
    fields = [f for f in fields_for(doc_type) if f.evaluated]
    rows = []
    for g in golds:
        pred = preds.get(g.id) or {}
        for f in fields:
            gf = g.fields.get(f.id)
            if gf is None:
                continue
            p = (pred.get("fields") or {}).get(f.id) or {}
            value = p.get("value")
            pred_found = bool(p.get("found")) and is_found(f, value)  # unverified answers never count
            gold_found = gf.label in ("yes", "no") if f.type == "nli" else gf.label == "yes"
            if f.type in CLASSES:
                label = value if p.get("found") else ("not_mentioned" if f.type == "nli" else "no")
                correct = label == gf.label
            else:
                label, correct = ("yes" if pred_found else "no"), pred_found == gold_found
            spans = g.span_texts(f.id)
            rows.append({
                "doc_id": g.id, "field": f.id, "type": f.type,
                "gold_label": gf.label, "gold_value": gf.value, "gold_spans": spans,
                "pred_label": label, "pred_value": value, "status": p.get("status", "error"),
                "model_answer": p.get("model_answer"), "quote": p.get("quote"),
                "gold_found": gold_found, "pred_found": pred_found, "correct": correct,
                # value accuracy only where the gold has a value; a missing answer is wrong
                "value_correct": None if gf.value in (None, "", []) else pred_found and value_match(f, value,
                                                                                                    gf.value),
                "quote_f1": quote_f1(p.get("quote"), spans) if pred_found and gold_found else None,
            })

    per_field = []
    for f in fields:
        fr = [r for r in rows if r["field"] == f.id]
        if not fr:
            continue
        tp = sum(r["pred_found"] and r["gold_found"] for r in fr)
        fp = sum(r["pred_found"] and not r["gold_found"] for r in fr)
        fn = sum(not r["pred_found"] and r["gold_found"] for r in fr)
        m = {"field": f.id, "label": f.label, "type": f.type, "n": len(fr), "gold_found": tp + fn,
             "found": prf(tp, fp, fn), "unverified": sum(r["status"] == "unverified" for r in fr)}
        if f.type in CLASSES:
            m["accuracy"] = sum(r["correct"] for r in fr) / len(fr)
            m["per_class"] = {c: prf(sum(r["pred_label"] == c and r["gold_label"] == c for r in fr),
                                     sum(r["pred_label"] == c and r["gold_label"] != c for r in fr),
                                     sum(r["pred_label"] != c and r["gold_label"] == c for r in fr))
                              for c in CLASSES[f.type]}
        vals = [r["value_correct"] for r in fr if r["value_correct"] is not None]
        if f.value_column and vals:
            m["value_accuracy"], m["value_n"] = sum(vals) / len(vals), len(vals)
        qs = [r["quote_f1"] for r in fr if r["quote_f1"] is not None]
        m["quote_f1"], m["quote_n"] = (_mean(qs), len(qs))
        per_field.append(m)

    n = len(golds)
    results = [preds.get(g.id) for g in golds]
    latencies = [r["seconds"] for r in results if r and r.get("seconds") is not None]
    wall = wall_seconds if wall_seconds is not None else sum(latencies)
    summary = {
        "documents": n,
        "failed_documents": sum(1 for r in results if not r or r.get("failed")),
        "documents_with_errors": sum(1 for r in results if not r or r.get("errors")),
        "found_f1_macro": _mean(m["found"]["f1"] for m in per_field),
        "accuracy_macro": _mean(m.get("accuracy") for m in per_field),
        "value_accuracy_macro": _mean(m.get("value_accuracy") for m in per_field),
        "quote_f1_macro": _mean(m["quote_f1"] for m in per_field),
        "unverified_answers": sum(m["unverified"] for m in per_field),
        "seconds_per_document": _mean(latencies),  # latency of one document
        "wall_seconds": round(wall, 3),
        "wall_seconds_per_document": wall / n,  # throughput (several documents may run at once)
        "price_per_hour": price_per_hour,
        "cost_per_document": price_per_hour * wall / 3600 / n if price_per_hour is not None else None,
        "chunks_per_document": _mean(r.get("chunks") for r in results if r),
    }
    return {"summary": summary, "per_field": per_field, "rows": rows}
