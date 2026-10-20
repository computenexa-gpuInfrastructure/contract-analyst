"""Extract the fields from one contract with the Model API (OpenAI-compatible), chunk by chunk.

Every answer must come with a quote that is really in the contract. We check each quote against the source text
(ignoring case, whitespace and quote styles); an answer whose quote can't be found is kept as "unverified" and is
NOT counted as found. Answers from several chunks are merged, preferring verified ones.
"""
import json
import os
import re
import time
from collections import Counter

from . import reader
from .fields import Field, fields_for, is_found, not_found_value
from .textnorm import QuoteFinder, norm_party

DISCLAIMER = ("Demo / case study output, not legal advice. Answers can be wrong or incomplete: check every answer "
              "against the contract and ask a lawyer before relying on it.")
MAX_TOKENS = 3000  # the JSON answer for ~20 fields with quotes
MAX_QUOTE_CHARS = 600

SYSTEM = (
    "You are a careful contract analyst. You read part of a contract and fill in fields as one JSON object. "
    "Use only the contract text. For every answer you give, copy a short quote (one or two sentences, at most 300 "
    "characters) from the contract that supports it, exactly as written. If this part of the contract does not "
    "cover a field, answer null (or \"no\" / \"not_mentioned\" as described) with quote null. Never invent a quote."
)

FORMATS = {
    "text": "text or null",
    "date": "date as YYYY-MM-DD, or short text such as \"perpetual\", or null",
    "duration": "duration such as \"1 year\" or \"30 days\", or null",
    "yesno": "\"yes\" or \"no\"",
    "yesno_quote": "\"yes\" or \"no\"",
    "nli": "\"yes\" if the contract says this, \"no\" if it says the opposite, or \"not_mentioned\"",
}
INSTRUCTIONS = ("plain", "guided")  # guided = add each field's reading rule (Field.hint)
GUIDED_NLI = (" For NDA statements: \"yes\" when the contract says it, even only in some cases, with conditions or "
              "as an exception to a ban; \"no\" when it says the opposite; \"not_mentioned\" when it doesn't "
              "address it. Read the definition of Confidential Information and the list of what is not "
              "confidential: many answers are there.")
GUIDED_QUOTES = (" Copy quotes character for character: keep the contract's own names and terms (if it says "
                 "\"BROOKS\" or \"Recipient\", don't write \"Receiving Party\"), don't change any word or tense, and "
                 "write \"...\" wherever you leave words out (for example between a list's opening words and one "
                 "of its items).")
DOC_NAMES = {"commercial": "commercial contract", "nda": "non-disclosure agreement (NDA)"}


class ModelError(RuntimeError):
    pass


def build_messages(fields: list[Field], doc_type: str, part: reader.Chunk, n_parts: int,
                   instructions: str = "plain") -> list[dict]:
    guided = instructions == "guided"
    lines = []
    for f in fields:
        fmt = "list of names, or []" if f.id == "parties" else FORMATS[f.type]
        desc = f'Statement: "{f.description}"' if f.type == "nli" else f.description
        if guided and f.hint:
            desc += f" How to decide: {f.hint}"
        lines.append(f"- {f.id} ({fmt}): {desc}")
    where = (f"This is part {part.index + 1} of {n_parts} (characters {part.start}-{part.end}); a field may be in "
             "another part." if n_parts > 1 else "This is the whole contract.")
    user = (f"Contract type: {DOC_NAMES[doc_type]}. {where}\n\n"
            "Fields (id (answer format): what to find):\n" + "\n".join(lines) + "\n\n"
            'Reply with one JSON object with every field id: {"<field id>": {"answer": ..., "quote": "..." or '
            "null}, ...}\n\n<contract>\n" + part.text + "\n</contract>")
    system = SYSTEM + (GUIDED_QUOTES if guided else "") + (
        GUIDED_NLI if guided and any(f.type == "nli" for f in fields) else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_json(text: str) -> dict:
    """The model's reply -> dict. Copes with ```json fences, text around the object, trailing commas, and a
    reasoning model's <think>...</think> block before the answer."""
    text = (text or "").strip()
    if "</think>" in text:
        text = THINK.sub("", text).split("</think>")[-1].strip()
    elif text.startswith("<think>"):
        raise ValueError("the model was still thinking when the answer ran out (thinking should be off)")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in the reply") from None
        body = re.sub(r",\s*([}\]])", r"\1", text[start:end + 1])
        data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    return data


def clean_value(field: Field, raw):
    """Model answer -> our value for the field's type."""
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.lower() in ("", "null", "none", "n/a", "na", "not found", "unknown", "not specified"):
            raw = None
    if field.type in ("yesno", "yesno_quote"):
        return "yes" if str(raw).strip().lower() in ("yes", "y", "true") or raw is True else "no"
    if field.type == "nli":
        s = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
        if s in ("yes", "entailment", "entailed", "true"):
            return "yes"
        if s in ("no", "contradiction", "contradicted", "false"):
            return "no"
        return "not_mentioned"
    if field.id == "parties":
        items = raw if isinstance(raw, list) else re.split(r";|\n", raw) if isinstance(raw, str) else []
        return [str(x).strip() for x in items if x and str(x).strip()]
    if raw is None or isinstance(raw, (dict, list)):
        return None
    return str(raw)


def make_candidate(field: Field, raw, finder: QuoteFinder, part: reader.Chunk) -> dict:
    """One chunk's answer for one field, with its quote checked against the full text."""
    if not isinstance(raw, dict):
        raw = {"answer": raw, "quote": None}
    value = clean_value(field, raw.get("answer", raw.get("value")))
    quote = raw.get("quote")
    if isinstance(quote, list):
        quote = " ... ".join(str(q) for q in quote if q)
    quote = str(quote).strip()[:MAX_QUOTE_CHARS] if quote else None
    if quote and quote.lower() in NO_QUOTE:
        quote = None
    return verify({"value": value, "quote": quote, "chunk": part.index, "near": part.start}, finder)


NO_QUOTE = ("null", "none", "n/a", "na", "\"null\"")


def verify(candidate: dict, finder: QuoteFinder) -> dict:
    """(Re)check a candidate's quote against the contract; also used by `rescore` with an updated checker."""
    quote = candidate.get("quote")
    if quote and quote.lower() in NO_QUOTE:
        quote = None
    span = finder.find(quote, near=candidate.get("near") or 0) if quote else None
    return dict(candidate, quote=quote, start=span[0] if span else None, end=span[1] if span else None,
                verified=span is not None)


def merge(field: Field, candidates: list[dict]) -> dict:
    """Combine one field's answers from all chunks. Order of preference: verified positive answers (for NDA
    statements the most common label wins, ties go to the earliest chunk; parties are combined), then unverified
    positive answers (reported, but not counted as found), then "not found"."""
    positive = [c for c in candidates if is_found(field, c["value"])]
    verified = [c for c in positive if c["verified"]]
    if verified:
        best = verified[0]
        if field.type == "nli":
            label = Counter(c["value"] for c in verified).most_common(1)[0][0]  # ties: first seen
            best = next(c for c in verified if c["value"] == label)
        out = dict(best, status="found", found=True)
        if field.id == "parties":
            names, seen = [], set()
            for c in verified:
                for name in c["value"]:
                    if norm_party(name) not in seen:
                        seen.add(norm_party(name))
                        names.append(name)
            out["value"] = names
        return out
    if positive:  # the model answered, but its quote isn't in the contract
        best = positive[0]
        return dict(best, value=not_found_value(field) if field.id != "parties" else [], model_answer=best["value"],
                    status="unverified", found=False)
    return {"value": not_found_value(field) if field.id != "parties" else [], "quote": None, "start": None,
            "end": None, "verified": False, "chunk": None, "status": "not_found", "found": False}


def reverify(result: dict, text: str) -> dict:
    """Redo the quote checks of a saved `analyse` result with the current checker (no model calls). Results saved
    with "candidates" are re-merged exactly; older ones only kept the merged answer, so there an unverified answer
    becomes found when its quote now verifies (other chunks' unverified answers weren't saved)."""
    if result.get("failed") or not result.get("fields"):
        return result
    finder = QuoteFinder(text)
    fields = fields_for(result["doc_type"])
    if "candidates" in result:
        candidates = {fid: [verify(c, finder) for c in cs] for fid, cs in result["candidates"].items()}
        return dict(result, fields={f.id: merge(f, candidates.get(f.id, [])) for f in fields},
                    candidates=candidates)
    out = {}
    for f in fields:
        r = result["fields"].get(f.id)
        if r and r.get("status") == "unverified":
            r = merge(f, [verify({"value": r.get("model_answer"), "quote": r.get("quote"), "chunk": r.get("chunk")},
                                 finder)])
        out[f.id] = r
    return dict(result, fields=out)


class Analyst:
    """Talks to the Model API. Settings come from CNX_API_BASE / CNX_API_KEY / CNX_MODEL unless given."""

    def __init__(self, base_url=None, api_key=None, model=None, timeout=180.0, retries=2, chunk_chars=None,
                 client=None, instructions="plain"):
        if instructions not in INSTRUCTIONS:
            raise ValueError(f"instructions must be one of {INSTRUCTIONS}")
        self.instructions = instructions
        self.model = model or os.environ.get("CNX_MODEL", "").strip()
        if client is None:
            from openai import OpenAI

            base_url = base_url or os.environ.get("CNX_API_BASE", "").strip()
            api_key = api_key or os.environ.get("CNX_API_KEY", "").strip()
            if not base_url or not api_key or not self.model:
                raise SystemExit("Set CNX_API_BASE, CNX_API_KEY and CNX_MODEL (from the Model API box on the "
                                 "Instances page).")
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=retries)
        self.client = client
        self.retries = retries
        self.json_mode = True  # response_format={"type": "json_object"}; switched off if the server refuses it
        self.temperature = True  # temperature=0 for repeatable answers; dropped for models that refuse the setting
        # Reasoning models (e.g. Qwen3.x) think out loud by default. On vLLM we ask for thinking off per request
        # (chat_template_kwargs, a vLLM extension; hosted APIs don't get it); dropped if a server refuses it.
        self.no_thinking = True
        self._vllm: bool | None = None
        self._chunk_chars = chunk_chars
        self.context_tokens = None

    @property
    def chunk_chars(self) -> int:
        if self._chunk_chars is None:
            self.context_tokens = self._served_context()
            self._chunk_chars = reader.chunk_chars_for(self.context_tokens)
        return self._chunk_chars

    def is_vllm(self) -> bool:
        """vLLM's /v1/models says owned_by "vllm" and reports max_model_len; hosted APIs don't."""
        if self._vllm is None:
            self._vllm = False
            try:
                for m in self.client.models.list().data:
                    extra = getattr(m, "model_extra", None) or {}
                    if getattr(m, "owned_by", None) == "vllm" or extra.get("max_model_len"):
                        self._vllm = True
                        break
            except Exception:
                pass
        return self._vllm

    def _served_context(self) -> int | None:
        """vLLM's /v1/models reports max_model_len for each served model."""
        try:
            for m in self.client.models.list().data:
                extra = getattr(m, "model_extra", None) or {}
                if m.id == self.model and extra.get("max_model_len"):
                    return int(extra["max_model_len"])
        except Exception:  # not fatal: use the default for the Qwen models we serve
            pass
        return None

    def _ask(self, messages: list[dict]) -> dict:
        """One chat call -> parsed JSON, retrying a reply that isn't valid JSON."""
        from openai import BadRequestError

        last, attempt = None, 0
        while attempt <= self.retries:
            kwargs = {"response_format": {"type": "json_object"}} if self.json_mode else {}
            if self.temperature:
                kwargs["temperature"] = 0
            if self.no_thinking and self.is_vllm():
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                           max_tokens=MAX_TOKENS, **kwargs)
            except BadRequestError as e:
                msg = str(e).lower()
                # Decide on what THIS request sent: with several documents at a time, another thread may already
                # have switched the setting off while this request was in flight.
                if "extra_body" in kwargs and ("chat_template_kwargs" in msg or "enable_thinking" in msg):
                    self.no_thinking = False  # a server that doesn't take chat-template options
                    continue
                if "temperature" in kwargs and "temperature" in msg:
                    self.temperature = False  # e.g. some hosted models only allow their default sampling
                    continue
                if "response_format" in kwargs and ("response_format" in msg or "json" in msg) and "context" not in msg:
                    self.json_mode = False  # server without JSON mode: rely on robust parsing (no try used)
                    continue
                raise
            attempt += 1
            content = resp.choices[0].message.content or ""
            try:
                return parse_json(content)
            except (ValueError, json.JSONDecodeError) as e:
                last = e
                messages = messages + [{"role": "assistant", "content": content},
                                       {"role": "user", "content": "That was not valid JSON. Reply with the JSON "
                                                                   "object only."}]
        raise ModelError(f"no valid JSON after {attempt} tries: {last}")

    def _ask_chunk(self, fields, doc_type, part, n_parts, depth=0) -> list[tuple[reader.Chunk, dict]]:
        """Answers for one chunk; a chunk the server says is too long is split in two (twice at most)."""
        from openai import BadRequestError

        try:
            return [(part, self._ask(build_messages(fields, doc_type, part, n_parts, self.instructions)))]
        except BadRequestError as e:
            if depth >= 2 or not re.search(r"context|maximum|too long|tokens", str(e).lower()):
                raise
        eighth = len(part.text) // 8
        halves = reader.chunk(part.text, len(part.text) // 2 + eighth, overlap=eighth)
        out = []
        for h in halves:
            sub = reader.Chunk(part.index, part.start + h.start, part.start + h.end, h.text)
            out += self._ask_chunk(fields, doc_type, sub, n_parts, depth + 1)
        return out

    def analyse(self, text: str, doc_type: str = "commercial") -> dict:
        """Extract every field of `doc_type` from `text`. Never raises for a model problem: failed chunks are
        listed in "errors" (and "failed" is true when no chunk worked)."""
        fields = fields_for(doc_type)
        t0 = time.monotonic()
        parts = reader.chunk(text, self.chunk_chars)
        finder = QuoteFinder(text)
        candidates = {f.id: [] for f in fields}
        errors = []
        for part in parts:
            try:
                answers = self._ask_chunk(fields, doc_type, part, len(parts))
            except Exception as e:  # one bad chunk must not lose the rest
                errors.append(f"part {part.index + 1}: {type(e).__name__}: {str(e)[:300]}")
                continue
            for sub, data in answers:
                for f in fields:
                    if f.id in data:
                        candidates[f.id].append(make_candidate(f, data[f.id], finder, sub))
        return {
            "doc_type": doc_type,
            "model": self.model,
            "instructions": self.instructions,
            "fields": {f.id: merge(f, candidates[f.id]) for f in fields},
            "candidates": candidates,  # every chunk's raw answer, so `rescore` can redo the checks exactly
            "chunks": len(parts),
            "chunk_chars": self.chunk_chars,
            "characters": len(text),
            "seconds": round(time.monotonic() - t0, 3),
            "errors": errors,
            "failed": bool(parts) and len(errors) == len(parts),
            "disclaimer": DISCLAIMER,
        }


def guess_doc_type(text: str) -> str:
    """'nda' when the start of the document says it's a non-disclosure / confidentiality agreement."""
    head = text[:3000].lower()
    if re.search(r"non[- ]?disclosure|confidentiality agreement|secrecy agreement|\bnda\b", head):
        return "nda"
    return "commercial"
