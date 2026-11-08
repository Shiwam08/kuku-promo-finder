#!/usr/bin/env python3
"""Human quality check, done blind.

1) Make a blind sheet that mixes the top-K promos of two runs (e.g. LLM vs keyword baseline),
   shuffled, with the source hidden:
       python qa.py blind outputs/<show>/gemini_gemini-2.5-flash outputs/<show>/heuristic -k 10
   -> qa/blind_review.csv  (fill the human_* columns)  +  qa/blind_key.json  (don't peek)

2) Score it:
       python qa.py score qa/blind_review.csv
   -> precision ("would post" >= 4), hook/cliffhanger/standalone pass rates per source,
      and rank correlation between model score and human rating.
"""
import argparse
import csv
import json
import random
from pathlib import Path

from promo_finder.parse import fmt

HUMAN = ["human_hook_ok(y/n)", "human_cliff_ok(y/n)", "human_standalone_ok(y/n)", "human_would_post(1-5)", "human_notes"]


def blind(a: str, b: str, k: int, out: str, seed: int):
    rows, key = [], {}
    for run in (a, b):
        ranked = json.loads((Path(run) / "promos_ranked.json").read_text())[:k]
        for p in ranked:
            rows.append((run, p))
    random.Random(seed).shuffle(rows)
    Path(out).mkdir(exist_ok=True)
    with open(Path(out) / "blind_review.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "segments", "duration_s", "transcript"] + HUMAN)
        for n, (run, p) in enumerate(rows, 1):
            item = f"R{n:02d}"
            key[item] = {"run": run, "id": p["id"], "rank": p.get("rank"), "score": p.get("score")}
            segs = " + ".join(f"Ep{s['ep']} {fmt(s['start'])}-{fmt(s['end'])}" for s in p["segments"])
            w.writerow([item, segs, p["duration"], "\n".join(p["transcript"])] + [""] * len(HUMAN))
    (Path(out) / "blind_key.json").write_text(json.dumps(key, indent=2))
    print(f"{len(rows)} items -> {out}/blind_review.csv (fill the human_* columns, then run: python qa.py score)")


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 2) if den else None


def score(sheet: str):
    sheet = Path(sheet)
    key = json.loads((sheet.parent / "blind_key.json").read_text())
    groups = {}
    for r in csv.DictReader(open(sheet)):
        if not r["human_would_post(1-5)"].strip():
            continue
        k = key[r["item"]]
        groups.setdefault(Path(k["run"]).name, []).append((k, r))
    yes = lambda v: v.strip().lower().startswith("y")
    report = {}
    for src, items in groups.items():
        post = [float(r["human_would_post(1-5)"]) for _, r in items]
        report[src] = {
            "reviewed": len(items),
            "would_post_rate(>=4)": round(sum(p >= 4 for p in post) / len(post), 2),
            "avg_would_post": round(sum(post) / len(post), 2),
            "hook_ok": round(sum(yes(r["human_hook_ok(y/n)"]) for _, r in items) / len(items), 2),
            "cliff_ok": round(sum(yes(r["human_cliff_ok(y/n)"]) for _, r in items) / len(items), 2),
            "standalone_ok": round(sum(yes(r["human_standalone_ok(y/n)"]) for _, r in items) / len(items), 2),
            "spearman(model_score, human)": spearman([k["score"] or 0 for k, _ in items], post),
        }
    print(json.dumps(report, indent=2))
    (sheet.parent / "qa_scores.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("blind"); b.add_argument("run_a"); b.add_argument("run_b")
    b.add_argument("-k", type=int, default=10); b.add_argument("--out", default="qa"); b.add_argument("--seed", type=int, default=7)
    s = sp.add_parser("score"); s.add_argument("sheet")
    a = ap.parse_args()
    blind(a.run_a, a.run_b, a.k, a.out, a.seed) if a.cmd == "blind" else score(a.sheet)
