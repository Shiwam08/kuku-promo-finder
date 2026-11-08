"""Self-contained HTML report an editor can skim."""
from __future__ import annotations

from html import escape as e

from .parse import fmt

CSS = """
:root{--bg:#fafaf8;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e4e4e0;--acc:#c2410c;--ok:#15803d;--warn:#b45309;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1d1d20;--ink:#ececee;--mute:#9a9aa2;--line:#2e2e33;--acc:#fb923c;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 80px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
.mute{color:var(--mute)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.stat b{display:block;font-size:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:12px 0}
.head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}.rank{font-weight:700;color:var(--acc);font-size:18px}
.score{margin-left:auto;font-weight:700}.pill{display:inline-block;border:1px solid var(--line);border-radius:99px;padding:1px 8px;font-size:12px;margin-right:4px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
.k{color:var(--mute);width:110px}.fix{color:var(--warn);font-size:13px}.err{color:var(--bad);font-size:13px}
details{margin-top:6px}summary{cursor:pointer;color:var(--mute);font-size:13px}pre{white-space:pre-wrap;font-size:12.5px;background:var(--bg);padding:8px;border-radius:8px;overflow-x:auto}
.bars span{display:inline-block;margin-right:12px;font-size:13px}
"""


def _segs(p):
    return "".join(f"<tr><td>{i + 1}</td><td>Ep {s['ep']}</td><td>{fmt(s['start'])}</td><td>{fmt(s['end'])}</td>"
                   f"<td>{s['end'] - s['start']:.1f}s</td><td>{e(s.get('label') or '')}</td></tr>"
                   for i, s in enumerate(p["segments"]))


def _card(p, plan_reason=None):
    j = p.get("judge") or {}
    bars = "".join(f"<span>{k} <b>{j.get(k, '-')}</b></span>" for k in ("hook", "cliffhanger", "standalone", "pull"))
    fixes = "".join(f"<div class=fix>auto-fix: {e(x)}</div>" for x in p.get("fixes", []))
    warns = "".join(f"<div class=fix>warning: {e(x)}</div>" for x in p.get("warnings", []))
    errs = "".join(f"<div class=err>error: {e(x)}</div>" for x in p.get("errors", []))
    rank = f"<span class=rank>#{p['rank']}</span>" if p.get("rank") else ""
    tags = f"<span class=pill>{p['source']}</span><span class=pill>{p['duration']:.0f}s</span>" + \
           ("<span class=pill>repaired</span>" if p.get("repaired") else "")
    plan = f"<tr><td class=k>post because</td><td><b>{e(plan_reason)}</b></td></tr>" if plan_reason else ""
    conf = j.get("confusing") or ""
    return f"""<div class=card id="{p['id']}"><div class=head>{rank}<b>{e(p.get('title') or '')}</b>
<span class=mute>{p['id']}</span>{tags}<span class=score>{p.get('score') if p.get('score') is not None else ''}</span></div>
<table><tr><th>#</th><th>Episode</th><th>Start</th><th>End</th><th>Len</th><th>What happens</th></tr>{_segs(p)}</table>
<table>{plan}<tr><td class=k>hook</td><td>{e(p.get('hook') or '')}</td></tr>
<tr><td class=k>first 3 s (actual)</td><td class=mute>{e(p.get('first_3s') or '')}</td></tr>
<tr><td class=k>cliffhanger</td><td>{e(p.get('cliffhanger') or '')}</td></tr>
<tr><td class=k>why it works</td><td>{e(p.get('why') or '')}</td></tr>
{f"<tr><td class=k>cold viewer</td><td>{e(j.get('verdict') or '')}{(' <span class=fix>Confusing: ' + e(conf) + '</span>') if conf else ''}</td></tr>" if j else ''}
</table><div class=bars>{bars}</div>{fixes}{warns}{errs}
<details><summary>clip transcript</summary><pre>{e(chr(10).join(p.get('transcript') or []))}</pre></details></div>"""


def render_html(kept, dups, rejected, plan, bible, s):
    pl = s["pipeline"]
    qa = s.get("qa") or {}
    u = s["usage"]["generator"]
    stats = [("valid promos (deduped)", s["after_dedup"]), ("proposed by model", pl["proposed"]),
             ("valid first try", pl["valid_first_try"]), ("auto-fixed by code", pl["auto_fixed"]),
             ("fixed by repair call", pl["repaired"]), ("hallucinated beat IDs", pl["bad_beat_ids"]),
             ("cross-episode", s["montages_valid"]), ("duplicates removed", s["duplicates_removed"]),
             ("episodes covered", f"{qa.get('episodes_covered', 0)}/{s['episodes']}"),
             ("end at episode end", qa.get("share_ending_at_episode_end")),
             ("model calls", u["calls_total"]), ("tokens in/out", f"{u['prompt_tokens']:,}/{u['completion_tokens']:,}")]
    grid = "".join(f"<div class=stat><b>{e(str(v))}</b><span class=mute>{e(k)}</span></div>" for k, v in stats)
    by_id = {p["id"]: p for p in kept}
    plan_html = "".join(_card(by_id[o["id"]], o.get("reason")) for o in plan.get("order", []) if o.get("id") in by_id)
    top = "".join(_card(p) for p in kept)
    rej = "".join(_card(p) for p in rejected if p.get("segments") is not None)
    return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Promo Finder Report</title><style>{CSS}</style></head><body><main>
<h1>Promo Finder: {e(s['show'])}</h1>
<div class=mute>generator <b>{e(s['model'])}</b> &middot; judge <b>{e(s['judge_model'])}</b> &middot; {s['episodes']} episodes &middot; {s['beats']} beats &middot; ran in {s['runtime_s']}s</div>
<p class=mute>{e(bible.get('premise', ''))}</p>
<div class=grid>{grid}</div>
<h2>Post these first</h2><p class=mute>{e(plan.get('notes', ''))}</p>{plan_html}
<h2>All promos, ranked ({len(kept)})</h2>
<p class=mute>Score = 0.35&middot;hook + 0.30&middot;cliffhanger + 0.20&middot;standalone + 0.15&middot;pull (1-5 each, from a judge that sees only the clip, never the series).</p>
{top}
<h2>Rejected by the validator ({len(rejected)})</h2><p class=mute>Model picks that broke a rule and could not be fixed. Kept for honesty and debugging.</p>{rej}
</main></body></html>"""
