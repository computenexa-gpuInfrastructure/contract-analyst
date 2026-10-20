"""A FAKE OpenAI-compatible server (stdlib only) for tests and dry runs without a GPU.

  python -m contract_analyst fake-server --port 8099
  CNX_API_BASE=http://127.0.0.1:8099/v1 CNX_API_KEY=fake CNX_MODEL=fake/keywords python -m contract_analyst ...

It is NOT a model: the default responder answers with keyword rules and quotes the matching sentence, which is
enough to exercise chunking, quote verification, merging and scoring on real files. Tests pass their own
responder (messages -> dict / str / (status, dict)) to return canned JSON.
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .fields import FIELDS

KEYWORDS = {
    # commercial
    "cap_on_liability": r"in no event shall .{0,80}liab|liability .{0,80}(shall not|will not) exceed|limitation of "
                        r"liability",
    "uncapped_liability": r"(not apply|exclud\w*) .{0,80}(gross negligence|wil(l)?ful misconduct|indemnif)",
    "anti_assignment": r"\bassign\w*\b.{0,80}\b(consent|without)\b",
    "audit_rights": r"\baudit",
    "exclusivity": r"\bexclusive(ly)?\b",
    "insurance": r"\binsurance\b",
    "minimum_commitment": r"\bminimum (purchase|order|quantit|annual|commitment)",
    "ip_ownership_assignment": r"hereby assigns?|shall be the (sole and )?exclusive property",
    "change_of_control": r"change (of|in) control",
    "non_compete": r"\bshall not .{0,60}\bcompet",
    "liquidated_damages": r"liquidated damages|termination fee",
    "termination_for_convenience": r"terminate .{0,80}(for convenience|without cause|for any reason)",
    "governing_law": r"governed by|construed .{0,60}laws? of|laws of the state of",
    "effective_date": r"effective date",
    "expiration_date": r"\bexpir",
    "renewal_term": r"\brenew",
    "notice_to_terminate_renewal": r"non-renewal|notice .{0,80}not to renew",
    # NDA
    "limited_use": r"solely for the purpose|only for the purpose",
    "share_with_employees": r"\bemployees\b",
    "share_with_third_parties": r"\b(advisors|consultants|agents)\b",
    "notice_on_compelled_disclosure": r"required by law|court order|subpoena",
    "verbal_info_included": r"\b(oral|orally|verbal)",
    "independent_development_allowed": r"independently developed",
    "third_party_acquisition_allowed": r"(received|obtained) from a third party",
    "no_licence": r"no licen[cs]e|not .{0,40}grant.{0,40}licen",
    "survival_of_obligations": r"surviv",
    "return_or_destroy": r"return or destroy|destroy or return",
    "keep_copies_after_return": r"retain .{0,40}cop|archival",
    "may_copy": r"\b(copy|copies|reproduce)\b",
    "must_be_marked": r"marked .{0,20}confidential|designated as confidential",
    "nda_is_confidential": r"(terms|existence) of this agreement",
    "non_solicit": r"\bsolicit",
    "no_reverse_engineering": r"reverse engineer",
    "technical_info_only": r"\btechnical information\b",
    "mutual_or_one_way": r"\bmutual\b",
    "term": r"period of .{0,30}(year|month)",
}
FIELD_TYPES = {f.id: f.type for fs in FIELDS.values() for f in fs}
_DATE = re.compile(r"[A-Z][a-z]+ \d{1,2}, \d{4}|\d{1,2}/\d{1,2}/\d{2,4}")
_DURATION = re.compile(r"\b\d+\)? (?:days?|months?|years?)\b|\b(?:one|two|three|thirty|sixty|ninety) (?:\(\d+\) )?"
                       r"(?:days?|months?|years?)\b", re.I)


def parse_prompt(messages: list[dict]) -> tuple[list[str], str]:
    """(field ids, contract text) from our extraction prompt."""
    user = next(m["content"] for m in reversed(messages) if m["role"] == "user" and "<contract>" in m["content"])
    ids = re.findall(r"^- ([a-z_]+) \(", user, re.M)
    text = user.split("<contract>\n", 1)[1].rsplit("\n</contract>", 1)[0]
    return ids, text


def keyword_responder(body: dict) -> dict:
    ids, text = parse_prompt(body["messages"])
    sentences = [s for s in re.split(r"(?<=[.;:])\s+|\n\s*\n", text) if s.strip()]
    out = {}
    for fid in ids:
        kind = FIELD_TYPES.get(fid, "text")
        pattern = KEYWORDS.get(fid)
        hit = next((s for s in sentences if pattern and re.search(pattern, s, re.I | re.S)), None)
        if fid == "agreement_date" or fid == "parties":
            hit = next((s for s in sentences[:15] if (_DATE if fid == "agreement_date" else
                                                       re.compile(r"\bbetween\b", re.I)).search(s)), None)
        # a model would copy the sentence with tidied whitespace; that must still verify
        quote = " ".join(hit.split())[:300] if hit else None
        if kind in ("yesno", "yesno_quote"):
            out[fid] = {"answer": "yes" if hit else "no", "quote": quote}
        elif kind == "nli":
            out[fid] = {"answer": "yes" if hit else "not_mentioned", "quote": quote}
        elif not hit:
            out[fid] = {"answer": None, "quote": None}
        elif fid == "parties":
            m = re.search(r"between\s+(.{3,120}?)\s+and\s+(.{3,120}?)(?:[,.(]|$)", quote, re.I)
            out[fid] = {"answer": [m[1], m[2]] if m else [], "quote": quote}
        elif fid == "governing_law":
            m = re.search(r"laws? of (?:the )?(?:State of |Commonwealth of |Province of )?([A-Z][A-Za-z]+(?: [A-Z]"
                          r"[a-z]+)?)", hit)
            out[fid] = {"answer": m[1] if m else None, "quote": quote}
        elif kind == "date":
            m = _DATE.search(hit)
            out[fid] = {"answer": m[0] if m else None, "quote": quote}
        elif kind == "duration":
            m = _DURATION.search(hit)
            out[fid] = {"answer": m[0] if m else None, "quote": quote}
        else:
            out[fid] = {"answer": "mutual" if fid == "mutual_or_one_way" else hit[:80], "quote": quote}
    return out


class FakeModelServer:
    """Runs in a background thread. `responder(body) -> dict | str | (status, dict)`."""

    def __init__(self, responder=keyword_responder, model="fake/keywords", max_model_len=32768, api_key=None,
                 host="127.0.0.1", port=0):
        self.responder, self.model, self.max_model_len, self.api_key = responder, model, max_model_len, api_key
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authorised(self):
                if server.api_key and self.headers.get("Authorization") != f"Bearer {server.api_key}":
                    self._send(401, {"error": {"message": "Invalid or missing API key", "type": "auth"}})
                    return False
                return True

            def do_GET(self):
                if not self._authorised():
                    return
                if self.path.rstrip("/").endswith("/models"):
                    model = {"id": server.model, "object": "model", "owned_by": "fake"}
                    if server.max_model_len:
                        model["max_model_len"] = server.max_model_len
                    return self._send(200, {"object": "list", "data": [model]})
                self._send(404, {"error": {"message": "not found"}})

            def do_POST(self):
                if not self._authorised():
                    return
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                server.requests.append(body)
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    return self._send(404, {"error": {"message": "not found"}})
                try:
                    reply = server.responder(body)
                except Exception as e:  # a broken responder shows up as a server error, like a real one
                    return self._send(500, {"error": {"message": f"fake responder failed: {e}"}})
                if isinstance(reply, tuple):
                    return self._send(*reply)
                content = reply if isinstance(reply, str) else json.dumps(reply)
                self._send(200, {"id": "fake-1", "object": "chat.completion", "created": 0, "model": server.model,
                                 "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": content}}],
                                 "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.base_url = f"http://{host}:{self.httpd.server_address[1]}/v1"
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
