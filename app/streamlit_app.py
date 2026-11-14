"""Web dashboard.   streamlit run app/streamlit_app.py"""
import base64, io, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src import config, db
from src.feedback import band

st.set_page_config(page_title="AI Sports Performance Assessment", layout="wide")
st.title("AI-Based Multimodal Sports Performance Assessment")

def jpeg_b64(rgb):
    """Picture -> text, so it can be shown inline (works in every browser/Streamlit version)."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


tab_analyze, tab_history, tab_models = st.tabs(["Analyze", "History", "Model results"])


@st.cache_resource(show_spinner="Loading models...")
def get_analyzer(sport):
    from src.pipeline import SportsAnalyzer
    return SportsAnalyzer(sport)


with tab_analyze:
    c1, c2 = st.columns([1, 2])
    sport = c1.selectbox("Select sport", list(config.SPORTS), format_func=lambda s: config.SPORTS[s]["title"])
    use_tracker_default = True
    video = c2.file_uploader("Upload video", type=[t.strip(".") for t in config.VIDEO_TYPES])
    use_tracker = st.checkbox("Use YOLO player tracking (slower, finds the player better; needs ultralytics)", value=True)

    if st.button("Analyze Performance", type="primary", disabled=video is None):
        try:
            analyzer = get_analyzer(sport)
        except FileNotFoundError as e:
            st.error(str(e)); st.stop()
        with tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix, delete=False) as tmp:
            tmp.write(video.read()); path = tmp.name
        with st.spinner("Analyzing video..."):
            try:
                r = analyzer.analyze(path, use_tracker=use_tracker)
            except Exception as e:
                st.error(f"Could not analyze this video: {e}"); st.stop()
        db.save_result(sport, video.name, r)

        if r["pose_detection"] < config.MIN_DETECTION_SCORING:
            st.warning(f"The player was found in only {r['pose_detection']:.0%} of the frames, so these "
                       "results may be unreliable. Try a clearer video where the player is bigger in the picture.")

        m = st.columns(4)
        m[0].metric("Sport", config.SPORTS[sport]["title"])
        m[1].metric("Detected action", config.pretty(r["action"]))
        m[2].metric("Confidence", f"{r['confidence']:.0%}")
        m[3].metric("Overall score", f"{r['overall']:.0f}/100", band(r["overall"]), delta_color="off")

        left, right = st.columns(2)
        names = [config.pretty(p) for p in r["scores"]]
        vals = list(r["scores"].values())
        bar = go.Figure(go.Bar(x=vals, y=names, orientation="h", text=[f"{v:.0f}" for v in vals]))
        bar.update_layout(title="Parameter scores", xaxis_range=[0, 100], height=350, yaxis_autorange="reversed")
        left.plotly_chart(bar, use_container_width=True)
        radar = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=names + names[:1], fill="toself"))
        radar.update_layout(title="Parameter comparison", polar=dict(radialaxis=dict(range=[0, 100])), height=350)
        right.plotly_chart(radar, use_container_width=True)

        probs = r["action_probs"]
        conf = go.Figure(go.Bar(x=[config.pretty(k) for k in probs], y=[v * 100 for v in probs.values()]))
        conf.update_layout(title="Action confidence (%)", yaxis_range=[0, 100], height=300)
        st.plotly_chart(conf, use_container_width=True)

        st.subheader("Pose visualization")
        if r["pose_frames"]:
            cols = st.columns(3)
            for i, img in enumerate(r["pose_frames"]):
                cols[i % 3].markdown(
                    f'<img src="data:image/jpeg;base64,{jpeg_b64(img)}" style="width:100%;border-radius:6px">',
                    unsafe_allow_html=True)
        else:
            st.info("Pose frames could not be drawn for this video.")

        fb = r["feedback"]
        st.subheader("Feedback")
        st.info(fb["summary"])
        a, b = st.columns(2)
        a.markdown("**Strengths**")
        if fb["strengths"]:
            for t in fb["strengths"]:
                a.success(t)
        else:
            a.write("No parameter scored high enough to count as a clear strength.")
        b.markdown("**Areas to improve**")
        for t in fb["improvements"]:
            b.warning(t)
        st.markdown("**Recommendations**")
        for t in fb["recommendations"]:
            st.markdown(f"- {t}")

with tab_history:
    rows = db.load_history()
    if rows:
        st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if k != "scores"} for r in rows]),
                     use_container_width=True)
    else:
        st.write("No analyses yet.")

with tab_models:
    import json
    for s in config.SPORTS:
        st.subheader(config.SPORTS[s]["title"])
        d = config.MODEL_DIR / s
        am, sm = d / "action_metrics.json", d / "scoring_metrics.json"
        if am.exists():
            m = json.loads(am.read_text())
            st.write({k: round(v, 3) for k, v in m.items() if k.startswith("test_")})
            if (d / "confusion_matrix.png").exists():
                st.image(str(d / "confusion_matrix.png"), width=380)
        if sm.exists():
            st.dataframe(pd.DataFrame(json.loads(sm.read_text())["comparison"])[["model", "MAE", "RMSE", "R2"]])
        if not am.exists() and not sm.exists():
            st.write("Not trained yet.")
