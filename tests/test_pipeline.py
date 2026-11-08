"""Offline test of the LLM path with a scripted fake model (no API key needed).

The fake model deliberately makes the mistakes real models make - invented beat IDs,
clips that are too long, hooks that start late, trailing footage after the cliffhanger -
so we can check the validator / repair / montage / judge / ranking code paths.

    python -m pytest -q tests      (or: python tests/test_pipeline.py)
"""
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from promo_finder import llm as llm_mod, pipeline  # noqa: E402
from promo_finder.parse import load_show  # noqa: E402

SHOW = next((ROOT / "data").glob("*/"), None)


def beats_from_prompt(user):
    return [(m.group(1), float(m.group(2)) * 60 + float(m.group(3))) for m in
            re.finditer(r"^(E\d+-B\d+) \[(\d+):(\d+\.\d+)-", user, re.M)]


def fake_complete(self, system, user, step=""):
    self.calls.append({"step": step, "prompt_tokens": len(user) // 4, "completion_tokens": 200, "seconds": 0})
    if step == "bible":
        return json.dumps({"title": "T", "premise": "A heroine is betrayed and fights back.",
                           "characters": [{"name": "Heroine", "aliases": ["HEROINE"], "role": "heroine"}],
                           "arc": [{"episodes": "1-5", "summary": "the betrayal"}], "big_reveals": []})
    if step.startswith("candidates"):
        b = beats_from_prompt(user)
        def at(t):  # first beat starting at/after t
            return next((i for i, x in b if x >= t), b[-1][0])
        promos = [
            # valid
            {"title": "ok", "segments": [{"start_beat": b[2][0], "end_beat": at(b[2][1] + 30), "label": "x"}],
             "hook_beat": b[2][0], "cliffhanger_beat": at(b[2][1] + 30), "hook": "h", "cliffhanger": "c", "why": "w"},
            # hook 8 s after start -> code should auto-trim start
            {"title": "late hook", "segments": [{"start_beat": b[0][0], "end_beat": at(40)}],
             "hook_beat": at(8), "cliffhanger_beat": at(40), "hook": "h", "cliffhanger": "c", "why": "w"},
            # invented ID -> must go to repair
            {"title": "bad id", "segments": [{"start_beat": "E99-B999", "end_beat": at(30)}],
             "hook_beat": "E99-B999", "cliffhanger_beat": at(30), "hook": "h", "cliffhanger": "c", "why": "w"},
            # too long -> repair
            {"title": "too long", "segments": [{"start_beat": b[0][0], "end_beat": b[-1][0]}],
             "hook_beat": b[0][0], "cliffhanger_beat": b[-1][0], "hook": "h", "cliffhanger": "c", "why": "w"},
        ]
        return json.dumps({"moments": [{"beat_id": b[3][0], "type": "hook", "strength": 4, "what": "x"}], "promos": promos})
    if step.startswith("repair"):
        b = beats_from_prompt(user)
        i, j = b[5][0], next(x for x, t in b if t >= b[5][1] + 25)
        return json.dumps({"promos": [{"title": "repaired", "segments": [{"start_beat": i, "end_beat": j}],
                                       "hook_beat": i, "cliffhanger_beat": j, "hook": "h", "cliffhanger": "c", "why": "w"}]})
    if step == "montage":
        blocks = re.findall(r"^(P\d+\.s\d+) \| Ep (\d+) \| ([\d.]+)s", user, re.M)
        a = min((x for x in blocks if x[1] == "1"), key=lambda x: float(x[2]))
        z = min((x for x in blocks if x[1] == "2"), key=lambda x: float(x[2]))
        assert float(a[2]) + float(z[2]) <= 60
        return json.dumps({"promos": [{"title": "m", "blocks": [a[0], z[0]], "hook": "h", "cliffhanger": "c", "why": "w"},
                                      {"title": "bad", "blocks": [a[0], "P9999.s0"]}]})
    if step.startswith("judge"):
        ids = re.findall(r"^=== (P\d+)", user, re.M)
        if step.endswith(":retry"):
            ids = ids
        # drop the last id once to exercise the "missing score" retry path
        return json.dumps({"scores": [{"id": i, "hook": 3 + n % 3, "cliffhanger": 4, "standalone": 3, "pull": 3,
                                       "confusing": "", "verdict": "v"} for n, i in enumerate(ids[:-1] if len(ids) > 1 else ids)]})
    if step == "plan":
        ids = re.findall(r"^(P\d+) \|", user, re.M)
        return json.dumps({"order": [{"id": i, "reason": "r"} for i in ids[:5]] + [{"id": "P0000", "reason": "bogus"}],
                           "notes": "n"})
    raise AssertionError(step)


def test_llm_path():
    assert SHOW, "put the screenplay folder under data/"
    llm_mod.LLM.complete = fake_complete
    with tempfile.TemporaryDirectory() as d:
        s = pipeline.run(str(SHOW), d + "/out", "fake/model", episodes=[1, 2, 3], rpm=0)
        out = Path(d) / "out"
        ranked = json.loads((out / "promos_ranked.json").read_text())
        rejected = json.loads((out / "rejected.json").read_text())
        plan = json.loads((out / "posting_plan.json").read_text())
        eps = {e.ep: e for e in load_show(SHOW)}

        assert s["pipeline"]["bad_beat_ids"] >= 3, "invented IDs must be caught"
        assert s["pipeline"]["repaired"] == 3, "one repaired promo per episode"
        assert s["pipeline"]["auto_fixed"] >= 3, "late hooks must be auto-trimmed"
        assert s["montages_valid"] == 1 and any("P9999.s0" in str(r["errors"]) for r in rejected)
        assert all(o["id"] != "P0000" for o in plan["order"]), "plan may not reference unknown promos"
        for p in ranked:
            assert 20 <= p["duration"] <= 60, p
            assert p["score"] is not None, "every promo judged (missing ones retried)"
            for sg in p["segments"]:
                ep = eps[sg["ep"]]
                # never cut inside a line of dialogue
                for b in ep.beats:
                    if b.kind == "DLG":
                        assert not (b.start < sg["start"] < b.end) and not (b.start < sg["end"] < b.end)
            if p["source"] == "single":
                hb = eps[p["episodes"][0]].beat(p["hook_beat"])
                assert hb.start - p["segments"][0]["start"] <= 3.0
        assert (out / "report.html").stat().st_size > 1000
        print("OK", json.dumps(s["pipeline"]), "valid:", len(ranked), "rejected:", len(rejected))


if __name__ == "__main__":
    test_llm_path()
