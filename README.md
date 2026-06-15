# Observathon — Student Kit
## Our approach (TL;DR)
1. **Observe**: `wrapper.py` logs full telemetry per request (latency, tokens, cost, tool calls, tool `obs`, PII) — the only window into the black-box agent.
2. **Diagnose**: 11 fault classes in `findings.json` (arithmetic, fabrication, tool sabotage, loops, latency, cost, drift, PII, injection…) with evidence.
3. **Fix config**: `temperature=0.2`, enable loop_guard/retry/cache, normalize_unicode, redact_pii, patch the MacBook `catalog_override`, zero session_drift/tool_error, set `tool_budget`/`context_size`.
4. **Rewrite prompt**: strict grounding, deterministic floor-discount total formula, each tool once, **treat order notes as DATA** (injection defense), end with one `Tong cong: <int> VND` line.
5. **Deterministic guardrail in the wrapper**: recompute the total from grounded `obs` (overrides model slips **and** neutralizes injected prices); refuse correctly on out-of-stock / over-quantity / not-found / unserved destination.
6. **Test without overfitting**: build our own oracle from telemetry to audit the private set (paraphrase + injection + new cities/products) while no official scorer is out.