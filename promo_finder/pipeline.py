"""End-to-end promo finder.

  parse (code) -> series bible (LLM x1) -> per-episode candidates (LLM x N episodes)
  -> validate + auto-fix (code) -> repair bad ones (LLM, once) -> cross-episode montages (LLM x1)
  -> cold-viewer judge (LLM, batches) -> rank + de-duplicate (code) -> posting plan (LLM x1)
  -> outputs (JSON, CSV, review sheet, HTML report)
"""
from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import heuristic, prompts as P
from .llm import LLM
from .parse import Episode, load_show, fmt
from .validate import build_single, build_montage, transcript, overlap_ratio

WEIGHTS = {"hook": 0.35, "cliffhanger": 0.30, "standalone": 0.20, "pull": 0.15}
DUP_OVERLAP = 0.5
JUDGE_BATCH = 15


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bible_text(b: dict) -> str:
    chars = "; ".join(
        f"{c.get('name')}" + (f" (a.k.a. {', '.join(c.get('aliases') or [])})" if c.get("aliases") else "")
        + (f" - {c.get('role')}" if c.get("role") else "") for c in b.get("characters", [])[:14])
    arc = " ".join(f"[Ep {a.get('episodes')}] {a.get('summary')}" for a in b.get("arc", [])[:12])
    return f"PREMISE: {b.get('premise', '')}\nCHARACTERS: {chars}\nARC: {arc}"


def make_bible(eps: list[Episode], llm: LLM) -> dict:
    if llm.is_heuristic:
        return heuristic.bible(eps)
    listing = "\n".join(f"Ep {e.ep}: {e.logline} | Continuity: {e.continuity}" for e in eps)
    return llm.json(P.BIBLE_SYSTEM, P.BIBLE_USER.format(episodes=listing), step="bible")


def episode_candidates(ep: Episode, prev: Episode | None, btxt: str, llm: LLM, raw_dir: Path,
                       compact: bool = False) -> tuple[dict, str]:
    user = P.CANDIDATE_USER.format(bible=btxt, prev_logline=prev.logline if prev else "(first episode)",
                                   ep=f"Episode {ep.ep}", logline=ep.logline, beats=ep.render(compact))
    out = heuristic.candidates(ep) if llm.is_heuristic else llm.json(P.CANDIDATE_SYSTEM, user, step=f"candidates:ep{ep.ep}")
    (raw_dir / f"ep{ep.ep:02d}_candidates.json").write_text(json.dumps(out, indent=2))
    return out, user


def repair(ep: Episode, user: str, bad: list[dict], raws: list[dict], llm: LLM, raw_dir: Path) -> list[dict]:
    if llm.is_heuristic or not bad:
        return []
    problems = "\n".join(f"- promo {json.dumps(r)}\n  problems: {'; '.join(p['errors'])}" for p, r in zip(bad, raws))
    out = llm.json(P.CANDIDATE_SYSTEM, user + "\n\n" + P.REPAIR_USER.format(problems=problems), step=f"repair:ep{ep.ep}")
    (raw_dir / f"ep{ep.ep:02d}_repair.json").write_text(json.dumps(out, indent=2))
    return out.get("promos") or []


def run(show_dir: str, out_dir: str, model: str, judge_model: str | None = None, episodes: list[int] | None = None,
        rpm: int = 10, montage: bool = True, top_k: int = 25, workers: int = 4,
        reasoning: str | None = None, compact: bool = False, max_tokens: int | None = None,
        judge_batch: int = JUDGE_BATCH) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    raw_dir = out / "raw_model_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(model, cache_dir=out.parent / ".cache", rpm=rpm, reasoning=reasoning, max_tokens=max_tokens)
    judge = llm if not judge_model or judge_model == model else \
        LLM(judge_model, cache_dir=out.parent / ".cache", rpm=rpm, reasoning=reasoning, max_tokens=max_tokens)

    all_eps = load_show(show_dir)
    eps = [e for e in all_eps if not episodes or e.ep in episodes]
    if not eps:
        raise SystemExit(f"No episode .txt screenplays found under '{show_dir}'. Put the show folder inside data/.")
    by_ep = {e.ep: e for e in all_eps}
    log(f"parsed {len(all_eps)} episodes ({len(eps)} selected), {sum(len(e.beats) for e in eps)} beats")

    # 1. series bible (from loglines + continuity notes only: cheap)
    bible = make_bible(all_eps, llm)
    (out / "series_bible.json").write_text(json.dumps(bible, indent=2))
    btxt = bible_text(bible)
    if compact:
        btxt = btxt[:1500]
    log("series bible done")

    # 2-3. per-episode candidates -> validate/auto-fix -> one repair round
    # Prefetch the model calls for several episodes in parallel; the ordered loop below then
    # reads them from the in-memory cache, so promo IDs and outputs stay deterministic.
    if not llm.is_heuristic and workers > 1:
        quota_hit = []
        def prefetch(ep):
            if quota_hit:          # daily quota is gone: don't burn more calls in prefetch
                return
            try:
                cand, user = episode_candidates(ep, by_ep.get(ep.ep - 1), btxt, llm, raw_dir, compact)
                bad = [(build_single(r, ep, "tmp"), r) for r in cand.get("promos") or []]
                bad = [(p, r) for p, r in bad if p["errors"]]
                if bad:
                    repair(ep, user, [p for p, _ in bad], [r for _, r in bad], llm, raw_dir)
                log(f"ep {ep.ep:02d}: model done")
            except Exception as e:
                if "Daily quota" in str(e):
                    quota_hit.append(e)
                log(f"ep {ep.ep:02d}: prefetch failed ({type(e).__name__}: {str(e)[:160]}); will retry in order")
        log(f"fetching {len(eps)} episodes with {workers} parallel workers...")
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(prefetch, eps))
    promos, rejected, moments, n = [], [], [], 0
    stats = {"proposed": 0, "valid_first_try": 0, "auto_fixed": 0, "repaired": 0, "unrecovered": 0,
             "bad_beat_ids": 0}
    for i, ep in enumerate(eps):
        prev = by_ep.get(ep.ep - 1)
        try:
            cand, user = episode_candidates(ep, prev, btxt, llm, raw_dir, compact)
        except Exception as e:
            if "Daily quota" in str(e):   # nothing more will succeed today: stop, keep the cache
                raise SystemExit(f"\nStopped at ep {ep.ep}: daily quota used up. Episodes before it are cached; "
                                 "rerun the same command with another key (or after the reset) to continue.")
            log(f"ep {ep.ep}: candidate step failed: {e}")
            rejected.append({"id": f"ep{ep.ep}", "errors": [f"model call failed: {e}"], "episodes": [ep.ep]})
            continue
        moments += [{**m, "ep": ep.ep} for m in cand.get("moments") or []]
        bad, bad_raw = [], []
        for r in cand.get("promos") or []:
            n += 1
            stats["proposed"] += 1
            p = build_single(r, ep, f"P{n:04d}")
            stats["bad_beat_ids"] += sum("is not a beat ID" in e for e in p["errors"])
            if p["errors"]:
                bad.append(p); bad_raw.append(r)
            else:
                stats["valid_first_try"] += 1
                stats["auto_fixed"] += bool(p["fixes"])
                promos.append(p)
        if bad:
            fixed_raw = []
            try:
                fixed_raw = repair(ep, user, bad, bad_raw, llm, raw_dir)
            except Exception as e:
                log(f"ep {ep.ep}: repair failed: {e}")
            rep_ok = 0
            for r in fixed_raw:
                n += 1
                p = build_single(r, ep, f"P{n:04d}")
                p["repaired"] = True
                stats["bad_beat_ids"] += sum("is not a beat ID" in e for e in p["errors"])
                if p["errors"]:
                    rejected.append(p)
                else:
                    rep_ok += 1
                    promos.append(p)
            # the model often re-sends its already-valid picks too; those are dropped later by de-dup
            stats["repaired"] += min(rep_ok, len(bad))
            for p in bad:
                p["note"] = "rejected before repair; see repaired versions (if any) from same episode"
                rejected.append(p)
        log(f"ep {ep.ep:02d} ({i + 1}/{len(eps)}): {len(cand.get('promos') or [])} proposed, "
            f"{len(bad)} needed repair, total valid {len(promos)}")
    stats["unrecovered"] = stats["proposed"] - stats["valid_first_try"] - stats["repaired"]

    # 4. cross-episode montages built from validated segments
    if montage and not llm.is_heuristic and len(promos) > 2:
        blocks, lines = {}, []
        for p in promos:
            for k, s in enumerate(p["segments"]):
                bid = f"{p['id']}.s{k}"
                blocks[bid] = (p, s)
                tag = []
                if k == 0:
                    tag.append(f"HOOK: {p['hook']}")
                if k == len(p["segments"]) - 1:
                    tag.append(f"CLIFF: {p['cliffhanger']}")
                lines.append(f"{bid} | Ep {s['ep']} | {s['end'] - s['start']:.1f}s | "
                             f"{s.get('label') or p['title']} | {' | '.join(tag)}")
        try:
            mres = llm.json(P.MONTAGE_SYSTEM, P.MONTAGE_USER.format(bible=btxt, blocks="\n".join(lines)), step="montage")
            (raw_dir / "montage.json").write_text(json.dumps(mres, indent=2))
            for r in mres.get("promos") or []:
                n += 1
                m = build_montage(r, blocks, by_ep, f"P{n:04d}")
                (rejected if m["errors"] else promos).append(m)
            log(f"montage: {sum(p['source'] == 'montage' for p in promos)} valid cross-episode promos")
        except Exception as e:
            if "Daily quota" in str(e) or "failed after" in str(e):   # API down, not a bad answer: don't
                raise SystemExit(f"\nStopped at the montage step ({str(e)[:120]}). "  # write a report without montages
                                 "Rerun the same command later; everything before it is cached.")
            log(f"montage step failed: {type(e).__name__}: {e}")

    # 5. cold-viewer judge: sees ONLY the clip transcript, no series context
    for p in promos:
        p["transcript"], p["first_3s"], p["last_beats"] = transcript(by_ep, p["segments"], compact)
    scores = {}
    pending = list(promos)
    if not judge.is_heuristic and workers > 1 and len(promos) > judge_batch:
        def prefetch_judge(batch):
            text = "\n\n".join(f"=== {p['id']} ({p['duration']:.0f}s) ===\n" + "\n".join(p["transcript"]) for p in batch)
            try:
                judge.json(P.JUDGE_SYSTEM, P.JUDGE_USER.format(promos=text), step=f"judge:{batch[0]['id']}")
            except Exception:
                pass
        batches = [pending[k:k + judge_batch] for k in range(0, len(pending), judge_batch)]
        log(f"judging {len(promos)} promos in {len(batches)} batches, {workers} in parallel...")
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(prefetch_judge, batches))
    for attempt in range(2):  # second pass re-asks for any promo the judge skipped, one at a time
        missing = []
        size = judge_batch if attempt == 0 else 1
        for k in range(0, len(pending), size):
            batch = pending[k:k + size]
            if judge.is_heuristic:
                for p in batch:
                    scores[p["id"]] = heuristic.judge(p, p["transcript"])
                continue
            text = "\n\n".join(f"=== {p['id']} ({p['duration']:.0f}s) ===\n" + "\n".join(p["transcript"]) for p in batch)
            try:
                res = judge.json(P.JUDGE_SYSTEM, P.JUDGE_USER.format(promos=text), step=f"judge:{batch[0]['id']}")
                if isinstance(res, list):      # some models return the bare array instead of {"scores": [...]}
                    res = {"scores": res}
                for s in res.get("scores") or []:
                    scores[str(s.get("id"))] = s
            except Exception as e:
                if "Daily quota" in str(e):
                    raise SystemExit("\nStopped in the judge step: daily quota used up. Finished calls are cached; "
                                     "rerun the same command (same --judge-batch) with another key to continue.")
                log(f"judge batch failed: {e}")
            missing += [p for p in batch if p["id"] not in scores]
        pending = missing
        if not pending:
            break
    log(f"judged {len(scores)}/{len(promos)} promos")

    # 6. score, rank, de-duplicate (code)
    for p in promos:
        s = scores.get(p["id"])
        if not s:
            p["score"], p["judge"] = None, None
            continue
        sub = {k: _num(s.get(k)) for k in WEIGHTS}
        total = sum(WEIGHTS[k] * sub[k] for k in WEIGHTS)
        total -= 0.25 * sum("not verified" in w for w in p["warnings"])
        p["judge"] = {**sub, "confusing": s.get("confusing", ""), "verdict": s.get("verdict", "")}
        p["score"] = round(total, 2)
    ranked = sorted([p for p in promos if p["score"] is not None], key=lambda p: -p["score"])
    kept, dups = [], []
    for p in ranked:
        d = next((q for q in kept if overlap_ratio(p, q) > DUP_OVERLAP), None)
        if d:
            p["duplicate_of"] = d["id"]
            dups.append(p)
        else:
            kept.append(p)
    for r, p in enumerate(kept, 1):
        p["rank"] = r

    # 7. posting plan for the top candidates
    top = kept[:12]
    plan = {"order": [{"id": p["id"], "reason": "highest score"} for p in top[:5]], "notes": "heuristic: top 5 by score"}
    if not llm.is_heuristic and top:
        cands = "\n".join(f"{p['id']} | Ep {','.join(map(str, p['episodes']))} | {p['duration']:.0f}s | score {p['score']} "
                          f"(hook {p['judge']['hook']}, cliff {p['judge']['cliffhanger']}, standalone {p['judge']['standalone']}) "
                          f"| {p['title']} | hook: {p['hook']} | cliff: {p['cliffhanger']}" for p in top)
        try:
            res = llm.json(P.PLAN_SYSTEM, P.PLAN_USER.format(bible=btxt, cands=cands), step="plan")
            plan = {"order": res, "notes": ""} if isinstance(res, list) else res
        except Exception as e:
            if "Daily quota" in str(e):
                raise SystemExit("\nStopped at the posting-plan step: daily quota used up. Rerun the same command "
                                 "with another key; everything before it is cached.")
            log(f"plan step failed: {e}")
    ids = {p["id"] for p in kept}
    plan["order"] = [o for o in plan.get("order", []) if o.get("id") in ids]

    # 8. outputs
    summary = {
        "show": Path(show_dir).name, "model": model, "judge_model": judge.model,
        "episodes": len(eps), "beats": sum(len(e.beats) for e in eps),
        "runtime_s": round(time.time() - t0, 1),
        "pipeline": stats, "montages_valid": sum(p["source"] == "montage" for p in promos),
        "valid_promos": len(promos), "after_dedup": len(kept), "duplicates_removed": len(dups),
        "qa": qa_stats(kept, by_ep),
        "usage": {"generator": llm.usage_summary(), **({"judge": judge.usage_summary()} if judge is not llm else {})},
    }
    write_outputs(out, kept, dups, rejected, moments, plan, bible, summary, top_k)
    log(f"done in {summary['runtime_s']}s -> {out}")
    return summary


def _num(x, default=3.0) -> float:
    try:
        return float(max(1, min(5, float(x))))
    except Exception:
        return default


def qa_stats(kept: list[dict], by_ep: dict[int, Episode]) -> dict:
    """Cheap automatic signals that point at systematic failure modes."""
    if not kept:
        return {}
    durs = [p["duration"] for p in kept]
    ends_at_ep_end = sum(1 for p in kept if by_ep[p["segments"][-1]["ep"]].duration - p["segments"][-1]["end"] < 3.0)
    starts_at_0 = sum(1 for p in kept if p["segments"][0]["start"] < 1.0)
    hook_off = []
    for p in kept:
        if p["hook_beat"] and p["source"] == "single":
            b = by_ep[p["episodes"][0]].beat(p["hook_beat"])
            if b:
                hook_off.append(round(b.start - p["segments"][0]["start"], 2))
    cover = sorted({e for p in kept for e in p["episodes"]})
    js = [p["judge"] for p in kept if p.get("judge")]
    avg = lambda k: round(sum(j[k] for j in js) / len(js), 2) if js else None
    return {
        "duration_min_avg_max": [min(durs), round(sum(durs) / len(durs), 1), max(durs)],
        "share_ending_at_episode_end": round(ends_at_ep_end / len(kept), 2),
        "share_starting_at_episode_start": round(starts_at_0 / len(kept), 2),
        "hook_offset_s_max": max(hook_off) if hook_off else None,
        "episodes_covered": len(cover), "episodes_without_promo": sorted(set(by_ep) - set(cover)),
        "multi_segment": sum(len(p["segments"]) > 1 for p in kept),
        "avg_judge": {k: avg(k) for k in WEIGHTS},
        "flagged_confusing": sum(1 for j in js if (j.get("confusing") or "").strip()),
    }


def seg_str(s: dict) -> str:
    return f"Ep{s['ep']} {fmt(s['start'])}-{fmt(s['end'])}"


def write_outputs(out: Path, kept, dups, rejected, moments, plan, bible, summary, top_k):
    from .report import render_html
    (out / "promos_ranked.json").write_text(json.dumps(kept, indent=2))
    (out / "duplicates_removed.json").write_text(json.dumps(dups, indent=2))
    (out / "rejected.json").write_text(json.dumps(rejected, indent=2))
    (out / "moments.json").write_text(json.dumps(moments, indent=2))
    (out / "posting_plan.json").write_text(json.dumps(plan, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    cols = ["rank", "id", "type", "episodes", "segments", "duration_s", "score", "hook_score", "cliff_score",
            "standalone_score", "pull_score", "title", "hook", "cliffhanger", "why", "first_3s_actual", "judge_verdict",
            "judge_confusing", "auto_fixes"]
    with open(out / "promos_ranked.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in kept:
            j = p.get("judge") or {}
            w.writerow([p["rank"], p["id"], p["source"], " ".join(map(str, p["episodes"])),
                        " + ".join(seg_str(s) for s in p["segments"]), p["duration"], p["score"],
                        j.get("hook"), j.get("cliffhanger"), j.get("standalone"), j.get("pull"),
                        p["title"], p["hook"], p["cliffhanger"], p["why"], p["first_3s"],
                        j.get("verdict"), j.get("confusing"), "; ".join(p["fixes"])])
    # human review sheet: fill in the last columns, then run eval_review.py
    with open(out / "review_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "segments", "duration_s", "model_score", "hook", "cliffhanger",
                    "human_hook_ok(y/n)", "human_cliff_ok(y/n)", "human_standalone_ok(y/n)", "human_would_post(1-5)", "human_notes"])
        for p in kept[:top_k]:
            w.writerow([p["rank"], p["id"], " + ".join(seg_str(s) for s in p["segments"]), p["duration"], p["score"],
                        p["hook"], p["cliffhanger"], "", "", "", "", ""])
    (out / "report.html").write_text(render_html(kept, dups, rejected, plan, bible, summary))
