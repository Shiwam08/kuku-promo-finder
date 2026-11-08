"""Rules-only baseline (no model). Two uses:
1. offline test of the whole pipeline without an API key;
2. a baseline to compare the LLM against in the quality check ("does the model beat keywords?").
"""
from __future__ import annotations

import re
from collections import Counter

from .parse import Episode

HOOK_WORDS = {
    r"\bslap": 5, r"\bstrik|\bstruck|\bpunch|\bshove|\bgrab": 4, r"\bkneel|\bknees": 3,
    r"\bgasp|\bshock|\bstun|\bhorrif": 3, r"\bscream|\bshout|\bsob|\btears?\b": 2,
    r"\bblood|\bdagger|\bsword|\bblade|\bpistol": 4, r"\bkiss": 3, r"\breveal|\bseal|\bcrest|\bsignet": 3,
    r"\bhow dare|\bdisgrace|\bworthless|\bbeggar|\bwhore|\bharlot|\bpathetic": 4,
    r"\bduke\b|\bgrace\b": 2, r"\bregret": 3, r"\bmarry|\bwedding|\baltar": 2,
}
CLIFF_WORDS = {r"cliffhanger": 5, r"\bsting|\bstinger|\bominous|\bswells": 3, r"cut to black": 3,
               r"\?\"$": 2, r"\.\.\.\"$": 2, r"\bwho\b|\bwhat\b": 1}


def score(text: str, table: dict) -> int:
    t = text.lower()
    return sum(w for pat, w in table.items() if re.search(pat, t, re.I))


def bible(eps: list[Episode]) -> dict:
    speakers = Counter(b.speaker for e in eps for b in e.beats if b.kind == "DLG")
    return {
        "title": "", "premise": eps[0].logline if eps else "",
        "characters": [{"name": s.title(), "aliases": [], "role": ""} for s, _ in speakers.most_common(12)],
        "arc": [{"episodes": str(e.ep), "summary": e.logline} for e in eps[::5]],
        "big_reveals": [],
    }


def candidates(ep: Episode) -> dict:
    beats = ep.beats
    hook_s = [(score(b.text, HOOK_WORDS) + (1 if b.kind == "DLG" else 0), b) for b in beats]
    cliff_s = [(score(b.text, CLIFF_WORDS), b) for b in beats]
    moments = [{"beat_id": b.id, "type": "hook", "strength": min(5, s), "what": b.text[:60]}
               for s, b in sorted(hook_s, key=lambda x: -x[0])[:4] if s > 0]
    moments += [{"beat_id": b.id, "type": "cliffhanger", "strength": min(5, s), "what": b.text[:60]}
                for s, b in sorted(cliff_s, key=lambda x: -x[0])[:2] if s > 0]
    promos = []
    # Promo A: end on the episode's cliffhanger, start on the strongest hook 22-55 s before it
    dlg = [b for b in beats if b.kind == "DLG"]
    cliff = dlg[-1] if dlg else beats[-1]
    win = [(s, b) for s, b in hook_s if cliff.end - 55 <= b.start <= cliff.end - 22 and b.kind != "CAM"]
    if win:
        s, h = max(win, key=lambda x: x[0])
        promos.append({"title": "Episode-ending cliffhanger", "segments": [{"start_beat": h.id, "end_beat": cliff.id, "label": ""}],
                       "hook_beat": h.id, "hook": h.text[:80], "cliffhanger_beat": cliff.id,
                       "cliffhanger": cliff.text[:80], "why": f"keyword hook score {s}, ends on episode's last line"})
    # Promo B: strongest hook anywhere, run ~35 s to the next question/exclamation line
    s, h = max(((s, b) for s, b in hook_s if b.kind != "CAM"), key=lambda x: x[0], default=(0, None))
    if h is not None:
        ends = [b for b in dlg if h.end + 20 <= b.end <= h.start + 55 and re.search(r"[?!]\"$|\.\.\.\"$", b.text)]
        if ends:
            e = ends[min(len(ends) - 1, 2)]
            promos.append({"title": "Strongest moment", "segments": [{"start_beat": h.id, "end_beat": e.id, "label": ""}],
                           "hook_beat": h.id, "hook": h.text[:80], "cliffhanger_beat": e.id,
                           "cliffhanger": e.text[:80], "why": f"highest keyword hook score ({s})"})
    return {"moments": moments, "promos": promos}


def judge(promo: dict, lines: list[str]) -> dict:
    body = [l for l in lines if not l.startswith("---")]
    first = " ".join(l for l in body if l[1:6] < "00:03")
    last = " ".join(body[-3:])
    names = {m for l in body for m in re.findall(r"\] ([A-Z][A-Z' ]+?)(?: \(|:)", l)}
    clamp = lambda x: max(1, min(5, x))
    return {"id": promo["id"], "hook": clamp(1 + score(first, HOOK_WORDS) // 2),
            "cliffhanger": clamp(1 + score(last, CLIFF_WORDS)),
            "standalone": clamp(5 - max(0, len(names) - 3)),
            "pull": clamp(1 + score(" ".join(body), HOOK_WORDS) // 6),
            "confusing": "", "verdict": "heuristic score"}
