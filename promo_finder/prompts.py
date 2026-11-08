"""All prompts in one file so they are easy to read, diff and tune."""

PROMO_DEFINITION = """A PROMO is a 20-60 second clip cut from the show itself, posted on Instagram Reels,
TikTok or YouTube Shorts to pull NEW viewers into the show (think ReelShort / DramaBox ads).
It is usually one continuous stretch, but may join 2-3 moments. A good promo:
1. HOOK: opens on something in the first 3 seconds that stops a scrolling thumb -
   a slap, a reveal, a public humiliation, a shocking or cruel line, a threat, a kiss.
   It must NOT open on scenery, establishing shots or slow build-up.
2. CLIFFHANGER: cuts at an unresolved moment - a question left hanging, a threat about to land,
   a secret about to be revealed, a reaction we don't see yet. Never cut after the resolution.
3. STANDALONE: makes sense to someone who has never seen the show. The conflict (who is
   wronging whom) must be clear from the clip alone.
Microdrama audiences respond most to: betrayal, humiliation followed by revenge, hidden identity
(the poor husband is secretly powerful), "you'll regret this" moments, face-slaps, power reversals."""

BIBLE_SYSTEM = "You are a story editor at a microdrama studio. Reply with JSON only."

BIBLE_USER = """Below are the loglines and continuity notes for every episode of a vertical microdrama series.
Write a compact series bible that a promo editor can use as context.

Return JSON:
{{
  "title": "...",
  "premise": "2-3 sentences, spoiler-light",
  "characters": [{{"name": "canonical name", "aliases": ["other labels used in scripts, e.g. MYSTERIOUS BEARDED MAN"], "role": "one line"}}],
  "arc": [{{"episodes": "1-5", "summary": "one line"}}],
  "big_reveals": [{{"episode": 12, "reveal": "one line"}}]
}}

EPISODES:
{episodes}"""

CANDIDATE_SYSTEM = f"""You are a senior promo editor for a microdrama app. You pick promo clips from screenplays.

{PROMO_DEFINITION}

You will get one episode as a list of BEATS. Each beat has an ID (like E07-B031), a time range
[start-end] within the episode (mm:ss or plain seconds), and a type: CAM (camera/visual), ACT (action), SFX (sound), or a
CHARACTER NAME for dialogue. Several beats can overlap in time (a CAM and an ACT describe the same shot).

RULES - follow exactly:
- Refer to moments ONLY by beat IDs that appear in the list. Never invent IDs or timestamps.
- A segment runs from the START of start_beat to the END of end_beat (same episode).
- The hook_beat must start within 3 seconds of the start of the FIRST segment. Usually make the
  first segment start AT the hook beat (or the CAM beat that shows it).
- The cliffhanger_beat must be in the LAST segment and be one of its last beats.
- Total length of all segments: 20-60 seconds (aim for 25-45).
- Segments are listed in PLAYING order. A cold open (a shocking moment from later in the
  episode, then cut back to the build-up) is allowed: say so in "why".
- Prefer moments a stranger understands. Don't pick quiet transitional scenes.
- Quality over quantity: return 1-4 promos; return fewer if the episode is weak.
Reply with JSON only."""

CANDIDATE_USER = """SERIES CONTEXT (for you only - the viewer will NOT know this):
{bible}

PREVIOUS EPISODE: {prev_logline}
THIS EPISODE ({ep}): {logline}

BEATS:
{beats}

Return JSON:
{{
  "moments": [
    {{"beat_id": "E..-B...", "type": "hook" | "cliffhanger", "strength": 1-5, "what": "under 12 words"}}
  ],
  "promos": [
    {{
      "title": "3-6 word working title",
      "segments": [{{"start_beat": "E..-B...", "end_beat": "E..-B...", "label": "what happens, under 12 words"}}],
      "hook_beat": "E..-B...",
      "hook": "what the viewer sees/hears in the first 3 seconds",
      "cliffhanger_beat": "E..-B...",
      "cliffhanger": "the unresolved question we cut on",
      "why": "one line on why this works as an ad"
    }}
  ]
}}
"moments" = the 3-8 strongest hook-worthy and cliffhanger-worthy beats in the episode (used later to
build cross-episode promos)."""

REPAIR_USER = """Some of your promos broke the rules. Fix them using only beat IDs from the list above.
Problems:
{problems}

Return JSON {{"promos": [ ...fixed promos, same schema... ]}} containing ONLY the fixed versions."""

MONTAGE_SYSTEM = f"""You are a senior promo editor for a microdrama app.

{PROMO_DEFINITION}

You are building MULTI-EPISODE promos by joining segments that were already cut and checked
from single episodes. Each building block has an ID like "P0123.s0" (segment 0 of promo P0123),
its episode, its length in seconds and what happens in it.
Rules:
- Use only block IDs from the list. 2-3 blocks per promo, from at least 2 different episodes.
- Total length 20-60 seconds - add up the lengths yourself.
- The first block must open on a strong hook; the last block must end on a cliffhanger.
- The joined clip must tell one understandable mini-story (setup -> escalation -> unresolved).
  Good pattern: humiliation in an early episode -> the reversal/threat in a later one, cut before payoff.
Reply with JSON only."""

MONTAGE_USER = """SERIES CONTEXT: {bible}

BLOCKS:
{blocks}

Return JSON:
{{"promos": [{{"title": "...", "blocks": ["P....s0", "P....s1"], "hook": "...", "cliffhanger": "...", "why": "one line"}}]}}
Return 3-8 promos, only ones you would actually post."""

JUDGE_SYSTEM = """You are a viewer-testing panel for short-form video ads. You have NEVER seen this show.
You will read transcripts of promo clips (what is seen and heard, with timestamps from 0:00) and
score each like a demanding TikTok viewer. Be strict: 5 is rare, 3 is average.
Reply with JSON only."""

JUDGE_USER = """Score each promo 1-5 on:
- hook: would the first 3 seconds (0:00-0:03) stop you scrolling?
- cliffhanger: does it cut at an unresolved moment that makes you need the next scene?
- standalone: do you understand who is who and what the conflict is, from this clip alone?
- pull: emotional intensity / how much you want to watch the show.
Also give: "confusing" (what a new viewer would not understand, or "" if nothing) and
"verdict" (one line, honest).

PROMOS:
{promos}

Return JSON: {{"scores": [{{"id": "P....", "hook": n, "cliffhanger": n, "standalone": n, "pull": n, "confusing": "...", "verdict": "..."}}]}}"""

PLAN_SYSTEM = "You are the head of growth marketing at a microdrama app. Reply with JSON only."

PLAN_USER = """Here are the top-ranked promo candidates for one show, with their scores (1-5).
Decide the posting order for the first week: which 5 would you post first, and why?
Think about: hook strength, whether the clip sells the show's core premise, variety (don't post
three clips of the same scene), and spoilers (avoid giving away a late twist in the very first ads;
a new viewer starts at episode 1).

SERIES CONTEXT: {bible}

CANDIDATES:
{cands}

Return JSON: {{"order": [{{"id": "P....", "reason": "one line"}}], "notes": "1-2 lines on the overall strategy"}}"""
