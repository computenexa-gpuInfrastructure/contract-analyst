"""Command line: extract one contract, evaluate on CUAD / ContractNLI, re-score saved answers, download data, or
run the fake server."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import datasets, report, score
from .extract import INSTRUCTIONS, Analyst, guess_doc_type, reverify
from .fields import DATASET_DOC_TYPE, fields_for
from .reader import read_text


def cmd_extract(args) -> int:
    path = Path(args.file)
    text = read_text(path)
    doc_type = args.type or guess_doc_type(text)
    analyst = Analyst(chunk_chars=args.chunk_chars, timeout=args.timeout, retries=args.retries,
                      instructions=args.instructions)
    result = analyst.analyse(text, doc_type)
    result["source"] = path.name
    result["field_labels"] = {f.id: f.label for f in fields_for(doc_type)}
    html_path = Path(args.html or f"{path.stem}-analysis.html")
    html_path.write_text(report.extraction_html(result, path.name))
    print(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"\nHTML report: {html_path}", file=sys.stderr)
    return 1 if result["failed"] else 0


def run_batch(analyst: Analyst, docs, workers: int) -> tuple[dict, float]:
    """Analyse every document; one bad document is recorded as failed, never stops the batch."""
    preds, t0 = {}, time.monotonic()

    def one(doc):
        try:
            return analyst.analyse(doc.text, doc.doc_type)
        except Exception as e:
            return {"failed": True, "errors": [f"{type(e).__name__}: {str(e)[:300]}"], "fields": {}, "chunks": 0,
                    "seconds": None}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, d): d for d in docs}
        for i, fut in enumerate(as_completed(futures), 1):
            doc, res = futures[fut], fut.result()
            preds[doc.id] = res
            state = "FAILED " + "; ".join(res["errors"])[:200] if res.get("failed") else (
                f"{res['seconds']:.1f}s, {res['chunks']} part(s)" + (" (some parts failed)" if res["errors"] else ""))
            print(f"[{i}/{len(docs)}] {doc.id[:70]}: {state}", file=sys.stderr)
    return preds, time.monotonic() - t0


def select(docs, skip=0, limit=None):
    """Documents skip .. skip+limit of a split (in the split's fixed order)."""
    docs = docs[skip or 0:]
    return docs[:limit] if limit else docs


def cmd_evaluate(args) -> int:
    docs = select(datasets.load(args.dataset, args.split), args.skip, args.limit)
    analyst = Analyst(chunk_chars=args.chunk_chars, timeout=args.timeout, retries=args.retries,
                      instructions=args.instructions)
    print(f"{len(docs)} {args.dataset} {args.split} documents, model {analyst.model}, {args.instructions} "
          f"instructions, chunks of {analyst.chunk_chars} characters, {args.workers} at a time", file=sys.stderr)
    preds, wall = run_batch(analyst, docs, args.workers)
    result = score.score(docs, preds, wall_seconds=wall, price_per_hour=args.price_per_hour)
    out = Path(args.out)
    meta = report.write_evaluation(result, {
        "dataset": args.dataset, "split": args.split, "skip": args.skip, "limit": args.limit, "model": analyst.model,
        "instructions": args.instructions,
        "doc_type": DATASET_DOC_TYPE[args.dataset], "chunk_chars": analyst.chunk_chars,
        "context_tokens": analyst.context_tokens, "workers": args.workers}, out)
    (out / "extractions.json").write_text(json.dumps(preds, indent=1, ensure_ascii=False))
    print_summary(result)
    print(f"\nReport: {out / 'report.html'} (also results.json, per_field.csv, predictions.csv)\n{meta['disclaimer']}")
    return 0


def cmd_rescore(args) -> int:
    """Score saved answers again (current quote check and scoring), without a GPU: before / after comparisons must
    use the same scorer."""
    src = Path(args.results)
    saved = json.loads((src / "results.json").read_text())
    meta, before = saved["meta"], saved["summary"]
    preds = json.loads((src / "extractions.json").read_text())
    docs = select(datasets.load(meta["dataset"], meta["split"]), meta.get("skip"), meta.get("limit"))
    by_id = {d.id: d for d in docs}
    missing = [i for i in preds if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} saved answers are not in {meta['dataset']} {meta['split']} "
                         f"(e.g. {missing[0]}): different dataset version?")
    preds = {i: reverify(p, by_id[i].text) for i, p in preds.items()}
    wall = before["wall_seconds_per_document"] * before["documents"]  # unrounded; wall_seconds is rounded
    result = score.score(docs, preds, wall_seconds=wall, price_per_hour=before["price_per_hour"])
    out = Path(args.out) if args.out else src / "rescored"
    meta = dict(meta, created=None, answers_created=meta.get("answers_created") or meta.get("created"),
                rescored_from=str(src))
    report.write_evaluation(result, meta, out)
    (out / "extractions.json").write_text(json.dumps(preds, indent=1, ensure_ascii=False))
    print_summary(result)
    after = result["summary"]
    print(f"\nBefore -> after: accuracy {_f(before['accuracy_macro']).strip()} -> {_f(after['accuracy_macro']).strip()}"
          f", found F1 {_f(before['found_f1_macro']).strip()} -> {_f(after['found_f1_macro']).strip()}, "
          f"unverified answers {before['unverified_answers']} -> {after['unverified_answers']}")
    print(f"Report: {out / 'report.html'}")
    return 0


def _f(x):
    return "   -" if x is None else f"{x * 100:4.0f}"


def print_summary(result: dict) -> None:
    s = result["summary"]
    print(f"\n{'field':32} {'foundF1':>7} {'acc':>5} {'value':>5} {'quote':>5}")
    for m in result["per_field"]:
        print(f"{m['field']:32} {_f(m['found']['f1']):>7} {_f(m.get('accuracy')):>5} "
              f"{_f(m.get('value_accuracy')):>5} {_f(m['quote_f1']):>5}")
    print(f"{'MACRO':32} {_f(s['found_f1_macro']):>7} {_f(s['accuracy_macro']):>5} "
          f"{_f(s['value_accuracy_macro']):>5} {_f(s['quote_f1_macro']):>5}")
    cost = f", ~${s['cost_per_document']:.4f} per document" if s["cost_per_document"] is not None else ""
    print(f"{s['documents']} documents ({s['failed_documents']} failed), {s['wall_seconds_per_document']:.2f} s "
          f"per document (wall){cost}")


def cmd_download(args) -> int:
    names = {"cuad": ["cuad-data", "cuad-v1"], "contractnli": ["contractnli"]}
    for ds in (["cuad", "contractnli"] if args.dataset == "all" else [args.dataset]):
        for name in names[ds]:
            print(datasets.download(datasets.SOURCES[name]))
    return 0


def cmd_fake_server(args) -> int:
    from .fake_model import FakeModelServer

    server = FakeModelServer(host=args.host, port=args.port)
    print(f"Fake model (keyword rules, NOT a real model) at {server.base_url}, model name {server.model}. "
          "Ctrl+C to stop.", file=sys.stderr)
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m contract_analyst",
                                description="Private Contract Analyst (demo, not legal advice).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--chunk-chars", type=int, help="characters per chunk (default: sized to the served "
                                                        "context)")
        sp.add_argument("--timeout", type=float, default=180.0, help="seconds per model call (default 180)")
        sp.add_argument("--retries", type=int, default=2, help="retries per model call (default 2)")
        sp.add_argument("--instructions", choices=INSTRUCTIONS, default="plain",
                        help="plain = the field questions only (default); guided = plus a reading rule per field")

    e = sub.add_parser("extract", help="extract the fields from one PDF or text file")
    e.add_argument("file")
    e.add_argument("--type", choices=["commercial", "nda"], help="contract type (default: guessed)")
    e.add_argument("--html", help="where to write the HTML report (default <file>-analysis.html)")
    common(e)
    e.set_defaults(func=cmd_extract)

    v = sub.add_parser("evaluate", help="score the model on CUAD or ContractNLI")
    v.add_argument("--dataset", choices=["cuad", "contractnli"], required=True)
    v.add_argument("--split", default="test")
    v.add_argument("--skip", type=int, default=0, help="skip the first N documents of the split")
    v.add_argument("--limit", type=int, help="at most N documents (after --skip)")
    v.add_argument("--out", default="results")
    v.add_argument("--price-per-hour", type=float, help="your GPU's price in $/hour, for the cost per document")
    v.add_argument("--workers", type=int, default=4, help="documents at a time (vLLM batches them; default 4)")
    common(v)
    v.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("rescore", help="score saved answers again with the current checks (no GPU needed)")
    r.add_argument("results", help="a folder written by evaluate (results.json + extractions.json)")
    r.add_argument("--out", help="where to write the new report (default <results>/rescored)")
    r.set_defaults(func=cmd_rescore)

    d = sub.add_parser("download", help="download the datasets into the cache")
    d.add_argument("--dataset", choices=["cuad", "contractnli", "all"], default="all")
    d.set_defaults(func=cmd_download)

    f = sub.add_parser("fake-server", help="run a fake OpenAI-compatible server (keyword rules) for dry runs")
    f.add_argument("--host", default="127.0.0.1")
    f.add_argument("--port", type=int, default=8099)
    f.set_defaults(func=cmd_fake_server)

    args = p.parse_args(argv)
    return args.func(args)
