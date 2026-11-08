#!/usr/bin/env python3
"""Find promos in a microdrama screenplay pack.

Examples
  python run.py data/MyShow                                  # default: Gemini Flash (free tier)
  python run.py data/MyShow --model groq/llama-3.3-70b-versatile
  python run.py data/MyShow --model heuristic                # no API key: keyword baseline
  python run.py data/MyShow --episodes 1-5                   # quick test on a few episodes
  python run.py show.zip                                     # a .zip of episode .txt files also works
"""
import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

from promo_finder.pipeline import run


def parse_range(s):
    if not s:
        return None
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("show", help="folder (or .zip) with one screenplay .txt per episode")
    ap.add_argument("--model", default=os.environ.get("PROMO_MODEL", "gemini/gemini-3.5-flash"),
                    help="any LiteLLM model string, or 'heuristic'")
    ap.add_argument("--judge-model", default=os.environ.get("PROMO_JUDGE_MODEL"),
                    help="separate model for the cold-viewer judge (default: same as --model)")
    ap.add_argument("--episodes", help="e.g. 1-10 or 1,4,7")
    ap.add_argument("--out", help="output folder (default outputs/<show>/<model>)")
    ap.add_argument("--rpm", type=int, default=int(os.environ.get("PROMO_RPM", 8)), help="max requests/minute")
    ap.add_argument("--no-montage", action="store_true", help="skip cross-episode promos")
    ap.add_argument("--workers", type=int, default=4, help="parallel model calls (default 4)")
    ap.add_argument("--reasoning", choices=["low", "medium", "high"], help="thinking effort for reasoning models")
    ap.add_argument("--compact", action="store_true",
                    help="shorter prompts for small free-tier request limits (auto-on for groq/)")
    ap.add_argument("--max-tokens", type=int, help="cap on each model reply")
    ap.add_argument("--judge-batch", type=int, help="promos per judge call (default 15, 3 in compact mode)")
    a = ap.parse_args()
    if a.model.startswith("groq/"):
        a.compact = True
        a.max_tokens = a.max_tokens or 2000

    show = Path(a.show)
    if show.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        zipfile.ZipFile(show).extractall(tmp)
        name = show.stem
        show = tmp
    else:
        name = show.name
    if not show.exists():
        sys.exit(f"not found: {show}")
    out = a.out or f"outputs/{name}/{a.model.replace('/', '_')}"
    s = run(str(show), out, a.model, a.judge_model, parse_range(a.episodes), a.rpm, not a.no_montage,
            workers=a.workers, reasoning=a.reasoning, compact=a.compact, max_tokens=a.max_tokens,
            judge_batch=a.judge_batch or (3 if a.compact else 15))
    print(f"\n{s['after_dedup']} promos -> {out}/report.html")


if __name__ == "__main__":
    main()
