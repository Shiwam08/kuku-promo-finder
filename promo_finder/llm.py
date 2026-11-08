"""One place that talks to a model. Swap models with a string, e.g.

    gemini/gemini-2.5-flash          (free tier, GEMINI_API_KEY)
    groq/llama-3.3-70b-versatile     (free tier, GROQ_API_KEY)
    openai/gpt-4.1 | anthropic/claude-sonnet-4-5 | ...   (any LiteLLM model string)
    heuristic                        (no model: keyword rules, used as a baseline/offline test)

Features: on-disk cache (re-runs are free and reproducible), simple rate limiting,
retry with backoff, robust JSON extraction, and a usage log for cost estimates.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path


class LLM:
    def __init__(self, model: str, cache_dir: str | Path = ".cache", rpm: int = 10,
                 temperature: float = 0.2, max_retries: int = 5, reasoning: str | None = None,
                 max_tokens: int | None = None):
        self.max_tokens = max_tokens
        self.model = model
        self.reasoning = reasoning      # e.g. "low": less thinking per call -> faster/cheaper
        self._lock = threading.Lock()
        self._mem: dict[str, str] = {}  # results already fetched in this run (parallel prefetch)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_gap = 60.0 / rpm if rpm > 0 else 0.0
        self.temperature = temperature
        self.max_retries = max_retries
        self._last = 0.0
        self.calls: list[dict] = []   # usage log

    @property
    def is_heuristic(self) -> bool:
        return self.model == "heuristic"

    def _key(self, system: str, user: str) -> str:
        r = f"\nreasoning={self.reasoning}" if self.reasoning else ""
        h = hashlib.sha256(f"{self.model}\n{self.temperature}{r}\n{system}\n{user}".encode()).hexdigest()
        return h[:24]

    def complete(self, system: str, user: str, step: str = "") -> str:
        if self.is_heuristic:
            raise RuntimeError("heuristic mode has no model; pipeline steps use rules instead")
        key = self._key(system, user)
        if key in self._mem:          # fetched earlier in this run: don't double-count usage
            return self._mem[key]
        cpath = self.cache_dir / f"{key}.json"
        if cpath.exists():
            rec = json.loads(cpath.read_text())
            self.calls.append({**rec["usage"], "step": step, "cached": True})
            self._mem[key] = rec["text"]
            return rec["text"]

        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # skip slow network fetch of the price list
        import litellm  # imported lazily so heuristic mode needs no dependency
        litellm.suppress_debug_info = True
        litellm.drop_params = True    # silently drop params a provider doesn't support
        err = None
        for attempt in range(self.max_retries):
            with self._lock:          # thread-safe rate limit: reserve the next free slot
                slot = max(time.time(), self._last + self.min_gap)
                self._last = slot
            if slot > time.time():
                time.sleep(slot - time.time())
            try:
                t0 = time.time()
                kwargs = dict(model=self.model,
                              messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": user}])
                if "gemini-3" not in self.model:   # Gemini 3+ deprecates sampling params
                    kwargs["temperature"] = self.temperature
                if self.reasoning:
                    kwargs["reasoning_effort"] = self.reasoning
                if self.max_tokens:
                    kwargs["max_tokens"] = self.max_tokens
                if os.environ.get("PROMO_JSON_MODE", "1") == "1":
                    kwargs["response_format"] = {"type": "json_object"}
                resp = litellm.completion(**kwargs)
                text = resp.choices[0].message.content or ""
                u = getattr(resp, "usage", None)
                usage = {"model": self.model,
                         "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                         "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                         "seconds": round(time.time() - t0, 2)}
                try:
                    usage["cost_usd"] = float(litellm.completion_cost(completion_response=resp) or 0)
                except Exception:
                    usage["cost_usd"] = None
                cpath.write_text(json.dumps({"text": text, "usage": usage, "step": step}))
                self.calls.append({**usage, "step": step, "cached": False})
                self._mem[key] = text
                return text
            except Exception as e:  # rate limits, transient 5xx, unsupported json mode...
                err = e
                msg = str(e).lower()
                if "interpreter shutdown" in msg:   # user pressed Ctrl+C: don't keep retrying in worker threads
                    raise
                if "response_format" in msg or "json_object" in msg:
                    os.environ["PROMO_JSON_MODE"] = "0"
                    continue
                if type(e).__name__ in ("NotFoundError", "AuthenticationError", "PermissionDeniedError") \
                        or "api key not valid" in msg or "request too large" in msg:
                    raise RuntimeError(f"{type(e).__name__}: check the model name / API key / request size (try --compact). {str(e)[:300]}") from e
                if "perday" in msg.replace(" ", "") or "per day" in msg or "daily" in msg:
                    raise RuntimeError("Daily quota used up for this key/model. Finished calls are cached: "
                                       "rerun the same command after the reset or with another key. "
                                       f"{str(e)[:200]}") from e
                backoff = min(60, 5 * 2 ** attempt)
                print(f"  [llm] {type(e).__name__}: {str(e)[:120]} -> retry in {backoff}s")
                time.sleep(backoff)
        raise RuntimeError(f"LLM call failed after {self.max_retries} tries: {err}")

    def json(self, system: str, user: str, step: str = "") -> dict:
        """Call the model and parse a JSON object; one re-ask if the JSON is broken."""
        text = self.complete(system, user, step)
        try:
            return extract_json(text)
        except ValueError:
            fix = self.complete(system, user + "\n\nYour previous reply was not valid JSON. "
                                "Reply with ONLY the JSON object, no prose, no code fences.", step + ":retry")
            return extract_json(fix)

    def usage_summary(self) -> dict:
        live = [c for c in self.calls if not c.get("cached")]
        costs = [c["cost_usd"] for c in live if c.get("cost_usd") is not None]
        return {
            "model": self.model,
            "calls_total": len(self.calls),
            "calls_live": len(live),
            "calls_cached": len(self.calls) - len(live),
            "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in self.calls),
            "completion_tokens": sum(c.get("completion_tokens", 0) for c in self.calls),
            "model_seconds": round(sum(c.get("seconds", 0) for c in live), 1),
            "cost_usd_reported": round(sum(costs), 4) if costs else None,
            "by_step": _group(self.calls),
        }


def _group(calls):
    out = {}
    for c in calls:
        s = (c.get("step") or "").split(":")[0]
        g = out.setdefault(s, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        g["calls"] += 1
        g["prompt_tokens"] += c.get("prompt_tokens", 0)
        g["completion_tokens"] += c.get("completion_tokens", 0)
    return out


def extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except Exception:
        pass
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(t[s:e + 1])
        except Exception:
            pass
    raise ValueError(f"no JSON object in model output: {text[:200]!r}")
