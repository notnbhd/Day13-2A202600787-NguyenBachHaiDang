"""Mitigation + observability layer wrapped around the opaque REAL-LLM agent.

The agent is silent and run_output.json is lean, so EVERYTHING we can see about
latency / tokens / cost / tool usage / drift / PII is logged from here via the
Day-13 telemetry toolkit. On top of observability we apply legal mitigations:

  - prompt routing : force our rewritten system prompt + few-shot on every call
  - input sanitize : neutralise instructions injected into order notes (GHI CHU)
  - cache          : memoise identical questions (thread-safe, shared dict)
  - retry/backoff  : re-issue on error/empty status with exponential backoff
  - output redact  : strip any email/phone the model still echoes (defence in depth)

Only the Python stdlib + the bundled telemetry/ package are imported.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata

try:
    from telemetry.logger import logger, new_correlation_id, set_correlation_id
    from telemetry.cost import cost_from_usage
    from telemetry.redact import redact
except Exception:  # telemetry optional -- wrapper still runs without it
    logger = None

    def cost_from_usage(model, usage):
        return 0.0

    def redact(s):
        return (s, 0)

    def new_correlation_id():
        return None

    def set_correlation_id(_cid):
        return None

_HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path, default=""):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return default


# Load our rewritten prompt + few-shot once, so we can route them on every request
# regardless of how the binary loads its (deliberately bad) shipped prompt.
SYSTEM_PROMPT = _read(os.path.join(_HERE, "prompt.txt")).strip()
try:
    EXAMPLES = json.loads(_read(os.path.join(_HERE, "examples.json"), "{}")).get("examples", [])
except Exception:
    EXAMPLES = []

# Markers that introduce an injected instruction/price inside order data.
_NOTE_MARKERS = re.compile(
    r"(?is)\b(ghi\s*ch[uú]|l[uư]u\s*[yý]|note|system|instruction|h[eệ]\s*th[oố]ng)\b\s*[:\-]?.*$"
)


def _sanitize(question: str) -> str:
    """Strip trailing injected 'notes' that try to override prices/behaviour.

    Conservative: only removes a note segment when it ALSO contains an imperative
    or a price-like token, so legitimate destinations/quantities are preserved.
    """
    if not isinstance(question, str):
        return question
    lines = re.split(r"[\n;]", question)
    cleaned = []
    for ln in lines:
        m = _NOTE_MARKERS.search(ln)
        if m:
            seg = m.group(0).lower()
            if re.search(r"\d{4,}|gi[aá]|price|set|d[uù]ng|ignore|b[oỏ]\s*qua|override|thay", seg):
                ln = ln[: m.start()]
        cleaned.append(ln)
    return " ".join(p.strip() for p in cleaned if p.strip()).strip()


def _key(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def _ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()

_TOTAL_LABELS = ("tong cong", "tong thanh toan", "tong chi phi", "tong so tien", "tong tien",
                 "final total", "total cost", "grand total", "total")
# Genuine stock / shipping failures -> order is unfulfillable, never fabricate a total.
# (Deliberately NOT matching coupon invalidity like "khong con hieu luc".)
_REFUSE_KW = ("het hang", "khong co trong kho", "khong con hang", "khong co san",
              "het san pham", "out of stock", "khong tim thay", "not found",
              "khong the van chuyen", "khong van chuyen", "khong giao den", "khong ship",
              "not served", "cannot ship", "can not ship", "do not ship", "currently ship",
              "unable to ship", "khong ho tro giao", "khong phuc vu")


def _bignums(s):
    out = []
    for n in re.findall(r"\d[\d.,]*\d|\d", s):
        d = re.sub(r"\D", "", n)
        if d and int(d) >= 1000:
            out.append(int(d))
    return out


def _finalize(answer):
    """Append a single canonical 'Tong cong: <int> VND' line so the scorer can parse the
    exact integer. Locates the LAST total label in the text (the conclusion) and takes the
    computed result after it (number after the last '=', else the first large number that
    follows -- works across markdown line breaks). Never removes the model's text; never
    adds a total to a refusal / unfulfillable order."""
    if not isinstance(answer, str) or not answer.strip():
        return answer
    low = _ascii(answer)
    if any(k in low for k in _REFUSE_KW):
        return answer
    pos = max((low.rfind(lab) for lab in _TOTAL_LABELS), default=-1)
    if pos < 0:
        return answer
    seg = low[pos:]
    after = seg.rsplit("=", 1)[1] if "=" in seg else seg
    nums = _bignums(after) or _bignums(seg)
    if not nums:
        return answer
    canon = "Tong cong: %d VND" % nums[0]
    if answer.rstrip().endswith(canon):
        return answer
    return answer.rstrip() + "\n" + canon


def mitigate(call_next, question, config, context):
    qid = context.get("qid")
    cid = new_correlation_id()
    if cid:
        set_correlation_id(cid)

    safe_q = _sanitize(question)

    # ---- cache (thread-safe; the run is concurrent) ------------------------
    cache = context.get("cache")
    lock = context.get("cache_lock")
    ckey = _key(safe_q)
    if config.get("cache", {}).get("enabled") and cache is not None and lock is not None:
        with lock:
            if ckey in cache:
                if logger:
                    logger.log_event("CACHE_HIT", {"qid": qid})
                return cache[ckey]

    # ---- prompt routing: force our prompt + few-shot -----------------------
    conf = dict(config)
    if SYSTEM_PROMPT:
        conf["system_prompt"] = SYSTEM_PROMPT
    if EXAMPLES:
        conf["examples"] = EXAMPLES
    # Cap output tokens to curb cost/latency (and fit small balances). We pass several
    # common key names since the binary may read any of them.
    _cap = int(config.get("max_completion_tokens") or 512)
    conf.setdefault("max_tokens", _cap)
    conf.setdefault("max_output_tokens", _cap)

    # ---- retry with backoff on error / empty answer ------------------------
    attempts = int(config.get("retry", {}).get("max_attempts", 1)) or 1
    backoff = int(config.get("retry", {}).get("backoff_ms", 0)) / 1000.0
    t0 = time.time()
    result, err = None, None
    for i in range(max(1, attempts)):
        try:
            result = call_next(safe_q, conf)
            status = result.get("status")
            if status in ("ok",) and (result.get("answer") or "").strip():
                break
            if status in ("loop", "max_steps", "no_action", "wrapper_error"):
                err = status
        except Exception as e:  # network/provider hiccup
            err = repr(e)
            result = None
        if i + 1 < attempts:
            time.sleep(backoff * (2 ** i))

    if result is None:
        result = {"answer": None, "status": "wrapper_error", "steps": 0, "trace": [], "meta": {}}

    wall_ms = int((time.time() - t0) * 1000)
    meta = result.get("meta", {}) or {}
    usage = meta.get("usage", {}) or {}

    # ---- output redaction (defence in depth) -------------------------------
    ans = result.get("answer")
    if isinstance(ans, str):
        ans = _finalize(ans)
        red, n = redact(ans)
        result["answer"] = red
        pii_n = n
    else:
        pii_n = 0

    # ---- observability: the ONLY place these signals exist -----------------
    if logger:
        logger.log_event("AGENT_CALL", {
            "qid": qid,
            "session": context.get("session_id"),
            "turn": context.get("turn_index"),
            "status": result.get("status"),
            "reported_latency_ms": meta.get("latency_ms"),
            "wall_ms": wall_ms,
            "retries": i,
            "steps": result.get("steps"),
            "tools_used": meta.get("tools_used", []),
            "n_tools": len(meta.get("tools_used", []) or []),
            "tokens": usage,
            "cost_usd": cost_from_usage(meta.get("model", ""), usage),
            "pii_redacted": pii_n,
            "sanitized": safe_q != (question or ""),
            "error": err,
        })

    # ---- store in cache ----------------------------------------------------
    if (config.get("cache", {}).get("enabled") and cache is not None and lock is not None
            and result.get("status") == "ok" and (result.get("answer") or "").strip()):
        with lock:
            cache[ckey] = result

    return result
