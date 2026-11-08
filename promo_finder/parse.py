"""Deterministic screenplay parser.

Turns each episode .txt into a list of timestamped "beats" (one CAMERA / ACTION /
SOUND FX / DIALOGUE block each). Every beat gets a stable ID like "E07-B031" so the
LLM can point at moments by ID instead of writing timestamps (which models invent).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

TS_RE = re.compile(r"Start:\s*\((\d+):(\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((\d+):(\d+(?:\.\d+)?)\)")
LABEL_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
HEADER_RE = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES):\s*(.*)$", re.I)
SCENE_HEADING_RE = re.compile(r"^scene heading:\s*(.*)$", re.I)
NON_DIALOGUE = {"CAMERA": "CAM", "ACTION": "ACT", "SOUND FX": "SFX"}


def to_sec(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def fmt(t: float) -> str:
    """Seconds -> 'mm:ss.xx' (same format as the screenplays)."""
    t = max(0.0, t)
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:05.2f}"


def shorten(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "..."


@dataclass
class Beat:
    id: str            # e.g. E07-B031
    ep: int
    idx: int
    kind: str          # CAM | ACT | SFX | DLG
    start: float
    end: float
    text: str
    speaker: str = ""  # dialogue only
    direction: str = ""  # e.g. "Cold, Commanding"
    scene: int = 0
    location: str = ""

    def body(self, compact: bool = False) -> str:
        """What is seen/heard. compact=True shortens camera/action text (dialogue stays whole)
        so a long episode fits small free-tier request limits."""
        if self.kind == "DLG":
            d = f" ({self.direction})" if self.direction and not compact else ""
            return f"{self.speaker}{d}: {self.text}"
        text = shorten(self.text, {"CAM": 50, "ACT": 100, "SFX": 40}[self.kind]) if compact else self.text
        return f"{self.kind}: {text}"

    def line(self, offset: float = 0.0, compact: bool = False) -> str:
        """One-line rendering used in prompts."""
        if compact:  # plain seconds: far fewer tokens than mm:ss.xx
            ts = f"[{self.start - offset:.1f}-{self.end - offset:.1f}]"
        else:
            ts = f"[{fmt(self.start - offset)}-{fmt(self.end - offset)}]"
        return f"{self.id} {ts} {self.body(compact)}"


@dataclass
class Episode:
    ep: int
    file: str
    logline: str
    continuity: str
    beats: list[Beat] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((b.end for b in self.beats), default=0.0)

    def beat(self, beat_id: str) -> Beat | None:
        return self._index().get(beat_id)

    def _index(self):
        if not hasattr(self, "_idx"):
            self._idx = {b.id: b for b in self.beats}
        return self._idx

    def render(self, compact: bool = False) -> str:
        return "\n".join(b.line(compact=compact) for b in self.beats)

    def to_dict(self):
        d = asdict(self)
        d["duration"] = self.duration
        return d


def parse_episode(path: Path) -> Episode:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    ep_num, logline, continuity = 0, "", ""
    scene, location = 0, ""
    beats: list[Beat] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        h = HEADER_RE.match(line)
        if h:
            key, val = h.group(1).upper(), h.group(2).strip()
            if key == "EPISODE NUMBER":
                ep_num = int(re.sub(r"\D", "", val) or 0)
            elif key == "EPISODE LOGLINE":
                logline = val
            else:
                continuity = val
        elif SCENE_RE.match(line):
            scene = int(SCENE_RE.match(line).group(1))
        elif SCENE_HEADING_RE.match(line):
            location = SCENE_HEADING_RE.match(line).group(1).strip()
        elif LABEL_RE.match(line):
            label = LABEL_RE.match(line).group(1).strip()
            # next non-empty line should be the timestamp
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            ts = TS_RE.search(lines[j]) if j < len(lines) else None
            if ts:
                start = to_sec(ts.group(1), ts.group(2))
                end = to_sec(ts.group(3), ts.group(4))
                body = []
                k = j + 1
                while k < len(lines) and lines[k].strip() and not LABEL_RE.match(lines[k].strip()):
                    body.append(lines[k].strip())
                    k += 1
                text = " ".join(body).strip()
                lab = label.upper()
                if lab in NON_DIALOGUE:
                    kind, speaker, direction = NON_DIALOGUE[lab], "", ""
                else:
                    kind = "DLG"
                    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", label)
                    speaker, direction = (m.group(1), m.group(2)) if m else (label, "")
                    text = text.strip('"“”')
                    text = f'"{text}"'
                beats.append(Beat(id="", ep=ep_num, idx=0, kind=kind, start=start, end=end,
                                  text=text, speaker=speaker.strip().upper(), direction=direction.strip(),
                                  scene=scene, location=location))
                i = k - 1
        i += 1
    if not ep_num:
        m = re.search(r"episode-(\d+)", path.name)
        ep_num = int(m.group(1)) if m else 0
    beats.sort(key=lambda b: (b.start, b.end))
    for n, b in enumerate(beats):
        b.ep, b.idx, b.id = ep_num, n, f"E{ep_num:02d}-B{n:03d}"
    return Episode(ep=ep_num, file=path.name, logline=logline, continuity=continuity, beats=beats)


def load_show(folder: str | Path) -> list[Episode]:
    """Load every episode .txt in a folder (recursively), skipping macOS junk."""
    folder = Path(folder)
    files = [p for p in folder.rglob("*.txt") if "__MACOSX" not in p.parts and not p.name.startswith("._")]
    eps = [parse_episode(p) for p in files]
    eps = [e for e in eps if e.beats]
    eps.sort(key=lambda e: e.ep)
    return eps


if __name__ == "__main__":
    import sys
    eps = load_show(sys.argv[1])
    for e in eps:
        kinds = {}
        for b in e.beats:
            kinds[b.kind] = kinds.get(b.kind, 0) + 1
        print(f"Ep {e.ep:02d}  {fmt(e.duration)}  beats={len(e.beats):3d}  {kinds}")
    print(f"{len(eps)} episodes, {sum(len(e.beats) for e in eps)} beats, "
          f"total {fmt(sum(e.duration for e in eps))}")
