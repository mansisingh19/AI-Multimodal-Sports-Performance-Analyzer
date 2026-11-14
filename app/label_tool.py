"""Labelling tool: a coach / rater watches each clip and gives scores.
    streamlit run app/label_tool.py
Scores are saved automatically in data/labels/<sport>_scores.csv after every clip.
You can stop any time and continue later."""
import base64
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import config
from src.labeling import (RUBRIC, clip_gif, ensure_label_file, find_video, is_labelled,
                          review_order, save_labels)

st.set_page_config(page_title="Clip labelling tool", layout="centered")
st.title("Technique rating tool")

sport = st.sidebar.selectbox("Sport", list(config.SPORTS), format_func=lambda s: config.SPORTS[s]["title"])
params = config.SPORTS[sport]["parameters"]
rater_note = st.sidebar.text_input("Rater name (for your report)", "")

st.sidebar.markdown("### How to rate (1 to 10)")
st.sidebar.write("1 = very poor, 5 = average, 10 = excellent. Use the whole range: "
                 "if every clip gets 7, the model cannot learn anything.")
for p in params:
    st.sidebar.markdown(f"**{config.pretty(p)}**: {RUBRIC.get(p, '')}")

df = ensure_label_file(sport)
done = int(is_labelled(df, sport).sum())
st.progress(done / len(df), text=f"{done} of {len(df)} clips labelled (aim for 150 or more)")

skipped = st.session_state.setdefault("skipped", set())
queue = [v for v in review_order(df, sport) if v not in skipped]
if not queue:
    st.success("All clips are labelled (or skipped). You can now train the scoring model.")
    st.stop()

vid = queue[0]
st.subheader(f"Clip: {vid}")


@st.cache_data(show_spinner=False)
def load_gif(path):
    return clip_gif(path)


try:
    gif = load_gif(str(find_video(sport, vid)))
    b64 = base64.b64encode(gif).decode()
    # shown as an inline picture so it works in every browser and Streamlit version
    st.markdown(f'<img src="data:image/gif;base64,{b64}" style="max-width:100%;border-radius:8px">',
                unsafe_allow_html=True)
except Exception as e:
    st.error(f"Could not show this clip: {e}")

with st.form("rating_form"):
    values = {p: st.slider(config.pretty(p), 1, 10, 5, key=f"{vid}_{p}") for p in params}
    c1, c2 = st.columns(2)
    save = c1.form_submit_button("Save and next", type="primary")
    skip = c2.form_submit_button("Skip this clip (unclear / bad video)")

if save:
    idx = df.index[df["video_id"] == vid][0]
    for p, v in values.items():
        df.loc[idx, p] = v * 10          # stored as 10..100
    save_labels(sport, df)
    st.rerun()
if skip:
    skipped.add(vid)
    st.rerun()
