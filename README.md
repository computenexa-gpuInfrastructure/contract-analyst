# Private Contract Analyst

**A blueprint for building private document AI.** This open template pulls the key facts and risk flags out of
commercial contracts and NDAs with an **open model on a GPU you control**, so the documents never go to a
third-party AI service. Every answer comes with a quote from the contract, and the quote is checked against the text.

It's for **builders**: startups, agencies and in-house teams making a document-review feature whose customers ask
*"does our data leave our control?"*. It comes with:
- **working code:** plain Python and the `openai` client against **any OpenAI-compatible endpoint** (vLLM, a
  ComputeNexa Model API, or a hosted API for comparison);
- **an honest evaluation:** two public, lawyer-labelled benchmarks (CUAD and ContractNLI), with accuracy, speed and
  cost per document;
- **the improvement loop:** plain questions, then guided instructions, then fine-tuning (coming in Part 2).

Read the case study: [Building a private contract analyst on one GPU](https://computenexa.com/case-studies/private-contract-analyst/). MIT licence; the datasets keep their own CC BY 4.0 licences (see Credits).

> **Not legal advice.** This is a demo of what a private model can do. Answers can be wrong or incomplete. Check
> every answer against the contract and ask a lawyer before you rely on it. Every output says this too.

This repository is the engine and the scoring. A web UI and a fine-tuning notebook will come later. To review other
document types, add fields in [`contract_analyst/fields.py`](contract_analyst/fields.py): the prompt, scoring and
reports all read from there.

## Results so far

Qwen 2.5 14B Instruct, 4-bit (AWQ), served by vLLM on **one RTX 5090** (32 GB), 4 documents at a time:

| NDAs (ContractNLI test, 123) | Private, plain | Private, guided | Claude Sonnet 5.5, hosted (reference) |
| --- | --- | --- | --- |
| Accuracy | 67.1% | 71.8% | 86.3% |
| Found the clause (F1) | 72.2% | 79.1% | 90.3% |
| Cost per 1,000 NDAs | ~$1.20 | ~$1.70 | ~$15 |

| Commercial contracts (CUAD test, 102) | Private, plain | Claude Sonnet 5.5, hosted (reference) |
| --- | --- | --- |
| Risk flags and other yes/no fields right | 88.0% | 90.4% |
| Dates, durations, governing law exactly right | 76.3% | 83.5% |
| Cost per 1,000 contracts | ~$2.30 | ~$63 |

Private costs are GPU time (one RTX 5090 at $1.49 an hour, billed by the minute); hosted costs are from the API
provider's usage page. On the dev set (61 NDAs), guided instructions moved the private model from 66.9% to 74.2%.
The hosted model was given the same plain questions; it doesn't accept temperature 0, so its answers can vary
slightly between runs.

## What it extracts

**Commercial contracts** (scored on [CUAD](https://www.atticusprojectai.org/cuad)):

| Field | Answer |
| --- | --- |
| Parties, agreement date, effective date, expiration date | names / dates |
| Renewal term, notice to stop renewal | durations ("1 year", "60 days") |
| Governing law | place ("New York") |
| Termination for convenience | yes / no |
| Cap on liability | yes / no + quote |
| Risk flags: uncapped liability, anti-assignment, audit rights, exclusivity, insurance, minimum commitment, IP ownership assignment, change of control, non-compete, liquidated damages | yes / no + quote |

**NDAs** (scored on [ContractNLI](https://stanfordnlp.github.io/contract-nli/)): 17 statements, each answered
**yes** (the NDA says it), **no** (it says the opposite) or **not mentioned**, with a quote: limited use, sharing with
employees, sharing with third parties, notice on compelled disclosure, verbal information included, independent
development allowed, third-party acquisition allowed, no licence, obligations survive, return or destroy, may keep
copies after return, may make copies, must be marked, the NDA itself is confidential, non-solicit, no reverse
engineering, technical information only.

Also extracted for NDAs but **not scored yet** (they aren't in ContractNLI; we'll score them with synthetic data
later): mutual or one-way, term, parties, governing law.

All fields live in one place: [`contract_analyst/fields.py`](contract_analyst/fields.py) (id, label, type, the
description used in the prompt, and the dataset mapping). Version 1 only uses CUAD categories with at least 50
examples.

## How it works

1. **Read** the PDF (`pypdf`) or text file.
2. **Chunk** it to fit the model. The Model API serves Qwen 2.5 with its full 32,768-token context (vLLM reports
   `max_model_len` at `/v1/models`, and we read it). We leave room for the instructions and the answer, and cap
   chunks at 40,000 characters (about 10,000 tokens) so the model stays sharp: most contracts take 1 to 3 calls.
   Chunks overlap by 1,500 characters so a clause cut at the edge is whole in the next chunk.
3. **Ask** the model, per chunk, for strict JSON: every field with an answer and an exact quote (JSON mode when the
   server supports it, robust parsing otherwise; timeouts and retries; a chunk that is too long is split).
4. **Verify** every quote: it must be in the contract, word for word. Formatting is ignored (case, spaces, bullets,
   ligatures, curly vs straight quotes, quote marks around a defined term), and "..." or a colon may skip text, so a
   list's lead-in plus one of its items counts. A changed word doesn't. An answer whose quote can't be found is
   shown as "quote not found" and is **not counted as found**.
5. **Merge** chunks: verified answers win; for NDA statements the most common verified label wins.

One bad document never stops a batch; it's recorded as failed.

## Run it

It works with **any OpenAI-compatible endpoint**: set `CNX_API_BASE`, `CNX_API_KEY` and `CNX_MODEL` (below), whether
that's your own vLLM server, a ComputeNexa Model API, or a hosted API you want to compare against.

### On ComputeNexa

1. **Launch a Model API.** On https://app.computenexa.com, pick a GPU and the **Model API** app:
   - **Qwen 2.5 14B Instruct (4-bit)** on a 24–32 GB GPU (e.g. RTX 4090, RTX 5090): the results above; or
   - **Qwen 2.5 14B Instruct** on a 48 GB GPU (e.g. RTX A6000): full precision.

   Choose a **Project** so the model and the datasets are saved between launches.
2. **Copy the three values** from the Model API box on the instance card (**Show API key**):

   ```bash
   export CNX_API_BASE="https://...-api.computenexa-apps.com/v1"
   export CNX_API_KEY="cnx-..."                      # keep it secret
   export CNX_MODEL="Qwen/Qwen2.5-14B-Instruct"       # the model you launched
   ```

3. **Install and run** (Python 3.11+, on your laptop or anywhere that can reach the URL):

   ```bash
   git clone https://github.com/computenexa-gpuInfrastructure/contract-analyst && cd contract-analyst
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt

   # one contract: prints JSON and writes my-contract-analysis.html
   python -m contract_analyst extract my-contract.pdf             # --type commercial|nda (default: guessed)

   # score the model on the official test splits (downloads the data the first time)
   python -m contract_analyst evaluate --dataset cuad --split test --limit 20 --out results/cuad --price-per-hour 1.12
   python -m contract_analyst evaluate --dataset contractnli --split test --out results/nda --price-per-hour 1.12
   ```

   `evaluate` writes `results.json`, `per_field.csv`, `predictions.csv`, `extractions.json` and `report.html` (a
   standalone page: per-field table and a drill-down per document with our quote next to the gold evidence).
   `--workers 4` (default) sends 4 documents at a time; vLLM batches them, so throughput goes up for free.

   **Some documents failed** (a rate limit, a timeout)? Re-run only those, then everything is scored again:

   ```bash
   python -m contract_analyst retry results/nda
   ```

   **Changed the checks or the scoring?** Score the saved answers again, no GPU needed (before / after comparisons
   must use the same scorer):

   ```bash
   python -m contract_analyst rescore results/nda    # writes results/nda/rescored/
   ```

**No GPU yet?** Try the plumbing with the fake server (keyword rules, NOT a model):

```bash
python -m contract_analyst fake-server --port 8099 &
CNX_API_BASE=http://127.0.0.1:8099/v1 CNX_API_KEY=fake CNX_MODEL=fake/keywords \
  python -m contract_analyst evaluate --dataset contractnli --limit 5 --out results/dry-run
```

## Cost

You pay for the GPU by the minute, not per document or per token. `evaluate` measures the wall time and, with
`--price-per-hour` (your GPU's price on the GPUs page), reports the **cost per document** = price per hour × wall
time ÷ documents.

Measured (see "Results so far"): on one RTX 5090 at $1.49 per hour, an NDA takes 3–4 seconds of wall time with 4
at a time, about $0.0012–0.0017 each, so roughly **$1.50 per 1,000 NDAs**. Commercial contracts are longer (1 to 3
model calls each). Stop the machine when you're done.

## Data and scoring

The datasets are downloaded on first use into `~/work/.cache/contract-analyst` on ComputeNexa (saved with your
project), or `~/.cache/contract-analyst` elsewhere (`CNX_CONTRACT_CACHE` overrides). Sizes and SHA-256 checksums are
checked. **They are never committed to this repo.**

- **CUAD v1** (510 contracts): `data.zip` from the [CUAD GitHub repo](https://github.com/TheAtticusProject/cuad)
  (18 MB; the **official** test split of 102 contracts, the other 408 are train) and `CUAD_v1.zip` from
  [Zenodo](https://zenodo.org/record/4595826) (106 MB; we only read `master_clauses.csv` for normalised dates,
  durations, governing law and parties).
- **ContractNLI** (607 NDAs, official train / dev / test = 423 / 61 / 123):
  [contract-nli.zip](https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip) (65 MB).

Metrics per field (`contract_analyst/score.py`):

- **Found**: precision / recall / F1 of "the clause is there" (a value, a yes, or for NDAs yes / no).
- **Yes / no and NDA statements**: accuracy and F1 per class.
- **Values** (dates, durations, governing law, parties): normalised exact match ("5/8/14" = "May 8, 2014";
  "twelve (12) months" = "1 year"; "the laws of the State of New York" = "New York"; every gold party named).
- **Quotes**: token-overlap F1 between our quote and the best-matching gold span.
- **Speed and cost**: seconds per document (latency and wall time) and cost per document.

## Plain vs guided instructions

`--instructions plain` (default) asks each NDA question exactly as ContractNLI states it. The base model then
misreads some of the dataset's conventions, for example:

- "shall not disclose except to its advisers" means sharing **is** allowed;
- a return of information only on request, not when the agreement ends, counts as "not mentioned".

`--instructions guided` adds one reading rule per NDA question (`NDA_HINTS` in `contract_analyst/fields.py`). The
rules were checked against the **train** split's labels. Guided mode also insists on word-for-word quotes (keep
the contract's own names, write "..." for skipped words): with the rules alone, the model started rewriting quotes
("BROOKS" -> "Receiving Party"), and those right answers failed the quote check.

To keep the comparison honest:

1. Compare the two only on the **dev** split (61 NDAs):

   ```bash
   python -m contract_analyst evaluate --dataset contractnli --split dev --out results/dev-plain --price-per-hour 1.37
   python -m contract_analyst evaluate --dataset contractnli --split dev --instructions guided --out results/dev-guided --price-per-hour 1.37
   ```

2. Run the **test** split once at the end, for the published numbers. The first 30 test NDAs were looked at while
   writing the rules, so also report the 93 never looked at (`--skip 30`). `rescore` keeps `--skip` and `--limit`.

## Limits

- Commercial contracts and NDAs only, in English.
- Long contracts are chunked; a clause that only makes sense together with another part far away can be missed.
- Scanned PDFs need OCR first (we read the PDF's text layer).
- CUAD and ContractNLI are US-centric public filings and NDAs; your contracts may look different.
- Not legal advice (see above).

## Credits

- **CUAD v1** by [The Atticus Project](https://www.atticusprojectai.org/cuad), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Dan Hendrycks, Collin Burns, Anya Chen and Spencer
  Ball, "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review", NeurIPS 2021 Datasets and Benchmarks.
- **ContractNLI** by Yuta Koreeda and Christopher D. Manning, licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "ContractNLI: A Dataset for Document-level Natural
  Language Inference for Contracts", Findings of EMNLP 2021.

We use the datasets unchanged for evaluation; this project is not endorsed by their authors.

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest tests          # fake OpenAI-compatible server, tiny hand-made fixtures; no GPU, no downloads
```

The fixtures in `tests/fixtures/` are invented contracts in the real CUAD / ContractNLI formats
(`python tests/fixtures/make_fixtures.py` regenerates them). CI runs these tests
(`.github/workflows/tests.yml`).

## License

MIT (see [LICENSE](LICENSE)) for the code. The datasets are not included; they're downloaded from their authors and
keep their own CC BY 4.0 licences. Model licences are between you and the model's maker.
