# Promo Finder: Kuku AI Generalist Assignment (Problem 1)

This tool reads microdrama screenplays and finds **20–60 s promo clips** that open on a hook in the first 3 seconds, cut on a cliffhanger, and make sense to a stranger. For every promo it returns the episode, exact start and end timestamps for each segment, the hook, the cliffhanger and why it works. It also ranks the promos and lists which ones to post first.

## How it works

```
screenplays ─► parse (code) ─► series bible (LLM ×1) ─► candidates per episode (LLM ×N)
           ─► validate + auto-fix (code) ─► repair bad picks (LLM, once) ─► cross-episode promos (LLM ×1)
           ─► cold-viewer judge (LLM, sees only the clip) ─► score + rank + de-dup (code) ─► posting plan (LLM ×1)
```

| Decision | Why |
|---|---|
| The model picks **beat IDs** (`E07-B031`), never timestamps | Models invent times. Code looks up the real ones, and a made-up ID is caught and counted |
| Code enforces every hard rule | 20–60 s length, hook within 3 s (auto-trims the start), ends on the cliffhanger (auto-trims extra footage after it), never cuts mid-line |
| Separate **judge** that has never seen the show | Tests "makes sense on its own" honestly. The generator knows the plot; a new viewer doesn't |
| Score = 0.35·hook + 0.30·cliff + 0.20·standalone + 0.15·pull | Weights are set by hand to follow the brief: the first 3 s matter most |
| Keyword `heuristic` baseline | Runs without a key, and gives a baseline the LLM has to beat in a blind review |
| One `--model` flag (LiteLLM) | You can swap in Gemini, Groq, OpenAI, Claude or any other model without code changes |
| Disk cache of every model call | Re-runs are free and reproducible, and the raw responses are saved |

## Run it

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...            # free key from https://aistudio.google.com/apikey
python run.py "data/<show folder>"                     # default model: gemini/gemini-3.5-flash; or pass show.zip
python run.py "data/<show folder>" --episodes 1-3      # quick test first
python run.py "data/<show folder>" --model groq/openai/gpt-oss-120b   # GROQ_API_KEY (auto compact mode)
python run.py "data/<show folder>" --model heuristic   # no key needed
python run.py "data/<show folder>" --workers 4 --reasoning low   # faster: parallel calls, less thinking
streamlit run app.py                                   # UI: upload a zip, pick a model
```

If a model name is rejected, pick one from the list in AI Studio and pass it as `--model gemini/<name>`. If you hit rate-limit errors, lower `--rpm`.

**Free-tier notes.** The Gemini free tier allows only about 20 requests per key per day, and a 50-episode show needs about 67. Every call is saved to disk, so when the daily quota runs out the run stops cleanly; rerun the same command (with another key, or after the reset) and it carries on from there. `503 overloaded` errors are retried automatically, so don't press Ctrl+C. Keep `--judge-batch` the same between reruns so saved judge calls are reused.

## Results on the provided 50-episode pack (Gemini 3.5 Flash)

110 promos proposed, 105 valid first try, 14 auto-fixed, 5 repaired, **0 invented beat IDs**, 94 kept after de-dup, all 50 episodes covered, 67 calls (432k input / 219k output tokens, about $2.60 at list price). In a blind top-10 comparison, 80% of the LLM's picks were rated "would post" against 20% for the keyword baseline. Known weak spots: the judge's ranking is noisy within the top 10, joins between segments can confuse, and there is no brand-safety filter yet. The screenplays and outputs are confidential, so they are not in this repo.

## Outputs (`outputs/<show>/<model>/`)

| File | What it is |
|---|---|
| `report.html` | Read this first: posting plan, ranked promos, segment table, the actual first 3 s, judge verdict, auto-fixes, rejected picks |
| `promos_ranked.csv` / `.json` | Every promo with segments, timestamps, hook, cliffhanger, why, scores |
| `posting_plan.json` | The 5 to post first, with reasons |
| `rejected.json`, `duplicates_removed.json` | Picks that broke a rule, and near-duplicates (kept for transparency) |
| `raw_model_responses/` | Unedited model output for every step |
| `summary.json` | Counts (proposed, auto-fixed, repaired, hallucinated IDs), QA stats, tokens |
| `review_sheet.csv` | Top 25 with blank columns for a human reviewer |

## Quality check

```bash
python qa.py blind outputs/<show>/gemini_gemini-3.5-flash outputs/<show>/heuristic -k 10
# fill qa/blind_review.csv (hook ok? cliff ok? standalone ok? would post 1-5?) without looking at the key
python qa.py score qa/blind_review.csv
python tests/test_pipeline.py      # offline test: a fake model that makes typical mistakes
```

## Confidentiality

The screenplays (`data/`) and outputs are git-ignored. Don't publish them. The Streamlit app takes the screenplays as an upload at run time.
