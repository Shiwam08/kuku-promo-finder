"""Streamlit UI: upload a zip of episode screenplays, pick a model, get ranked promos.

    streamlit run app.py

The screenplays are uploaded at run time and never stored in the repo.
"""
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from promo_finder.pipeline import run

st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Promo Finder")
st.caption("Reads microdrama screenplays and finds 20-60 s promo clips: hook in the first 3 s, cut on a cliffhanger.")

with st.sidebar:
    model = st.text_input("Model (any LiteLLM string)", os.environ.get("PROMO_MODEL", "gemini/gemini-3.5-flash"),
                          help="e.g. gemini/gemini-3.5-flash, groq/openai/gpt-oss-120b, openai/gpt-4.1, or 'heuristic'")
    judge_model = st.text_input("Judge model (optional)", "")
    key_name = st.selectbox("API key variable", ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
    key = st.text_input("API key (kept in this session only)", type="password")
    eps = st.text_input("Episodes (blank = all, e.g. 1-5)", "")
    rpm = st.number_input("Max requests/minute", 1, 1000, 8)
    montage = st.checkbox("Cross-episode promos", True)

up = st.file_uploader("Zip of episode .txt screenplays", type="zip")
if up and st.button("Find promos", type="primary"):
    if key:
        os.environ[key_name] = key
    tmp = Path(tempfile.mkdtemp())
    show = tmp / Path(up.name).stem
    zipfile.ZipFile(io.BytesIO(up.getvalue())).extractall(show)
    out = tmp / "out"
    episodes = None
    if eps.strip():
        episodes = []
        for part in eps.split(","):
            a, _, b = part.partition("-")
            episodes += list(range(int(a), int(b or a) + 1))
    with st.spinner("Running... (free-tier models take a few minutes for 50 episodes)"):
        summary = run(str(show),str(out), model.strip(), judge_model.strip() or None, episodes, int(rpm), montage)
    st.success(f"{summary['after_dedup']} promos found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in out.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(out))
    st.download_button("Download all outputs (zip)", buf.getvalue(), "promo_outputs.zip")
    components.html((out / "report.html").read_text(), height=1400, scrolling=True)
    with st.expander("Run summary"):
        st.json(summary)
