"""Turn model picks (beat IDs) into exact, checked timestamps.

The model only names beats. Code does everything that must be exact:
- beat ID -> timestamp lookup (a hallucinated ID is caught here)
- never cut in the middle of a line of dialogue (snap to line boundaries)
- hook must start within 3 s of the clip start (auto-trim if the model started too early)
- clip must end on the cliffhanger (auto-trim trailing resolution)
- 20-60 s total, segments in one episode each, no overlaps
Anything that can't be auto-fixed goes back to the model once, then to rejected.json.
"""
from __future__ import annotations

from .parse import Episode, Beat, fmt

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0       # hook must start within this many seconds of the clip start
CLIFF_TAIL = 4.0        # at most this much footage after the cliffhanger beat ends


def snap_start(ep: Episode, t: float) -> float:
    for b in ep.beats:
        if b.kind == "DLG" and b.start < t < b.end:
            t = b.start
    return t


def snap_end(ep: Episode, t: float) -> float:
    for b in ep.beats:
        if b.kind == "DLG" and b.start < t < b.end:
            t = b.end
    return min(t, ep.duration)


def beats_in(ep: Episode, start: float, end: float) -> list[Beat]:
    return [b for b in ep.beats if start - 0.01 <= b.start < end - 0.01]


def transcript(eps: dict[int, Episode], segments: list[dict], compact: bool = False) -> tuple[list[str], str, str]:
    """What the viewer sees/hears, re-timed from 0:00. Returns (lines, first3s, last_lines)."""
    lines, first3, offset = [], [], 0.0
    for n, s in enumerate(segments):
        ep = eps[s["ep"]]
        if len(segments) > 1:
            lines.append(f"--- cut to segment {n + 1}: Ep {s['ep']} {fmt(s['start'])}-{fmt(s['end'])} ---")
        for b in beats_in(ep, s["start"], s["end"]):
            rel = offset + (b.start - s["start"])
            rel_end = offset + (min(b.end, s["end"]) - s["start"])
            body = b.body(compact)
            lines.append(f"[{fmt(rel)}-{fmt(rel_end)}] {body}")
            if rel < HOOK_WINDOW and b.kind != "CAM":
                first3.append(body)
            if rel < HOOK_WINDOW and b.kind == "CAM" and not first3:
                first3.append(body)
        offset += s["end"] - s["start"]
    tail = [l for l in lines if not l.startswith("---")][-3:]
    return lines, " | ".join(first3), " | ".join(tail)


def _resolve(ep: Episode, beat_id: str, field: str, errors: list[str]) -> Beat | None:
    b = ep.beat(str(beat_id).strip()) if beat_id else None
    if b is None:
        errors.append(f"{field} '{beat_id}' is not a beat ID in episode {ep.ep}")
    return b


def build_single(raw: dict, ep: Episode, pid: str) -> dict:
    """Validate + auto-fix one single-episode promo proposed by the model."""
    errors, warnings, fixes = [], [], []
    segs = []
    for s in raw.get("segments") or []:
        a = _resolve(ep, s.get("start_beat"), "start_beat", errors)
        z = _resolve(ep, s.get("end_beat"), "end_beat", errors)
        if not (a and z):
            continue
        start, end = a.start, max(z.end, a.end)
        if end <= start:
            errors.append(f"segment {s.get('start_beat')}..{s.get('end_beat')} ends before it starts")
            continue
        segs.append({"ep": ep.ep, "start": start, "end": end, "start_beat": a.id,
                     "end_beat": z.id, "label": s.get("label", "")})
    if not segs and not errors:
        errors.append("no segments")

    hook = _resolve(ep, raw.get("hook_beat"), "hook_beat", errors) if raw.get("hook_beat") else None
    cliff = _resolve(ep, raw.get("cliffhanger_beat"), "cliffhanger_beat", errors) if raw.get("cliffhanger_beat") else None
    if not raw.get("hook_beat"):
        errors.append("missing hook_beat")
    if not raw.get("cliffhanger_beat"):
        errors.append("missing cliffhanger_beat")

    if segs and hook:
        s0 = segs[0]
        if not (s0["start"] - HOOK_WINDOW <= hook.start <= s0["end"]):
            errors.append(f"hook_beat {hook.id} ({fmt(hook.start)}) is not inside the first segment "
                          f"({fmt(s0['start'])}-{fmt(s0['end'])})")
        else:
            off = hook.start - s0["start"]
            if off > HOOK_WINDOW or off < 0:
                # start on the camera shot that shows the hook, if it begins shortly before it
                cams = [b for b in ep.beats if b.kind == "CAM" and b.start <= hook.start < b.end
                        and hook.start - b.start <= 1.5]
                new = cams[0].start if cams else hook.start
                fixes.append(f"moved start {fmt(s0['start'])} -> {fmt(new)} so the hook lands in the first 3 s "
                             f"(was {off:+.1f}s)")
                s0["start"] = new
    if segs and cliff:
        sl = segs[-1]
        if not (sl["start"] <= cliff.start < sl["end"] + 0.01 or sl["start"] <= cliff.end <= sl["end"] + 0.01):
            errors.append(f"cliffhanger_beat {cliff.id} ({fmt(cliff.start)}) is not inside the last segment "
                          f"({fmt(sl['start'])}-{fmt(sl['end'])})")
        else:
            if cliff.end > sl["end"]:
                fixes.append(f"extended end {fmt(sl['end'])} -> {fmt(cliff.end)} to include the cliffhanger")
                sl["end"] = cliff.end
            tail = sl["end"] - cliff.end
            if tail > CLIFF_TAIL:
                new = cliff.end + 1.0  # one beat of reaction, then cut
                fixes.append(f"trimmed end {fmt(sl['end'])} -> {fmt(new)}: cut on the cliffhanger "
                             f"instead of {tail:.1f}s later")
                sl["end"] = new

    for k, s in enumerate(segs):
        orig = s["start"]
        s["start"] = round(snap_start(ep, orig), 2)
        if k == 0 and hook and hook.start - s["start"] > HOOK_WINDOW:
            # snapping back to the line's start would push the hook past 3 s: skip that line instead
            nxt = [b.end for b in ep.beats if b.kind == "DLG" and b.start < orig < b.end and b.end <= hook.start]
            if nxt:
                s["start"] = round(max(nxt), 2)
                fixes.append(f"started after the line in progress ({fmt(s['start'])}) to keep the hook in the first 3 s")
        s["end"] = round(snap_end(ep, s["end"]), 2)

    # a line that continues the previous one ("...but you never") makes a jarring start/cut
    for s in segs:
        first = next((b for b in beats_in(ep, s["start"], s["end"]) if b.kind == "DLG"), None)
        if first and first.text.lstrip('"“ ').startswith(("...", "…")) and first.start - s["start"] < HOOK_WINDOW:
            warnings.append(f"segment at {fmt(s['start'])} opens on a half-sentence ({first.id}: {first.text[:40]!r})")

    # order / overlap
    for x, y in zip(segs, segs[1:]):
        if y["start"] < x["end"] and x["start"] < y["end"]:
            errors.append(f"segments overlap ({fmt(x['start'])}-{fmt(x['end'])} and {fmt(y['start'])}-{fmt(y['end'])})")
        elif y["start"] < x["start"]:
            warnings.append("non-chronological segments (cold open)")
    if len(segs) > 1:
        gaps = sum(max(0, y["start"] - x["end"]) for x, y in zip(segs, segs[1:]))
        if gaps < 1.0:
            # adjacent segments: merge into one continuous stretch
            segs = [{**segs[0], "end": segs[-1]["end"], "end_beat": segs[-1]["end_beat"],
                     "label": " / ".join(s["label"] for s in segs if s["label"])}]
            fixes.append("merged touching segments into one continuous clip")

    dur = round(sum(s["end"] - s["start"] for s in segs), 2)
    if segs and not (MIN_LEN <= dur <= MAX_LEN):
        errors.append(f"total length {dur:.1f}s is outside 20-60s")

    return {
        "id": pid, "source": "single", "episodes": [ep.ep], "title": raw.get("title", ""),
        "segments": segs, "duration": dur,
        "hook_beat": hook.id if hook else raw.get("hook_beat"), "hook": raw.get("hook", ""),
        "cliffhanger_beat": cliff.id if cliff else raw.get("cliffhanger_beat"),
        "cliffhanger": raw.get("cliffhanger", ""), "why": raw.get("why", ""),
        "errors": errors, "warnings": warnings, "fixes": fixes,
    }


def build_montage(raw: dict, blocks: dict[str, tuple[dict, dict]], eps: dict[int, Episode], pid: str) -> dict:
    """Validate a cross-episode promo made of already-validated segments ('P0012.s0' ...)."""
    errors, warnings, segs = [], [], []
    ids = raw.get("blocks") or []
    for bid in ids:
        if bid not in blocks:
            errors.append(f"block '{bid}' does not exist")
            continue
        promo, seg = blocks[bid]
        segs.append({**seg, "from": bid})
    episodes = sorted({s["ep"] for s in segs})
    if len(episodes) < 2:
        errors.append("montage must use at least 2 episodes")
    dur = round(sum(s["end"] - s["start"] for s in segs), 2)
    if segs and not (MIN_LEN <= dur <= MAX_LEN):
        errors.append(f"total length {dur:.1f}s is outside 20-60s")
    hook_beat = cliff_beat = None
    if segs:
        p0, s0 = blocks[ids[0]] if ids and ids[0] in blocks else (None, None)
        if p0 and p0["segments"] and p0["segments"][0] is s0:
            hook_beat = p0["hook_beat"]
        else:
            warnings.append("first block is not the opening of its source promo: hook not verified by code")
        pl = blocks[ids[-1]][0] if ids and ids[-1] in blocks else None
        if pl and pl["segments"] and pl["segments"][-1] is blocks[ids[-1]][1]:
            cliff_beat = pl["cliffhanger_beat"]
        else:
            warnings.append("last block is not the ending of its source promo: cliffhanger not verified by code")
    for x, y in zip(segs, segs[1:]):
        if (y["ep"], y["start"]) < (x["ep"], x["start"]):
            warnings.append("non-chronological (cold open)")
    return {
        "id": pid, "source": "montage", "episodes": episodes, "title": raw.get("title", ""),
        "segments": [{k: v for k, v in s.items()} for s in segs], "duration": dur,
        "hook_beat": hook_beat, "hook": raw.get("hook", ""),
        "cliffhanger_beat": cliff_beat, "cliffhanger": raw.get("cliffhanger", ""),
        "why": raw.get("why", ""), "errors": errors, "warnings": warnings, "fixes": [],
    }


def overlap_ratio(p: dict, q: dict) -> float:
    """Shared footage / shorter promo's length (0..1). Used to drop near-duplicates."""
    shared = 0.0
    for a in p["segments"]:
        for b in q["segments"]:
            if a["ep"] == b["ep"]:
                shared += max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    m = min(p["duration"], q["duration"]) or 1.0
    return shared / m
