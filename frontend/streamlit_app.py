"""PlantGuard AI Streamlit dashboard."""

import os
from datetime import datetime

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL_NAME = "EfficientNetB1"
CLASS_COUNT = 15
INPUT_SIZE = "240 × 240"

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱", layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Nunito:wght@600;700;800&display=swap');

:root { --ink:#102b20; --green:#14763b; --soft:#f5faf5; --line:#dceadf; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink) !important; }
.stApp { background:linear-gradient(135deg,#f2faf4 0%,#ffffff 48%,#eef8f0 100%); }
#MainMenu, footer, header { visibility:hidden; }
.block-container { max-width:1260px; padding:1.5rem 2rem 2.5rem; }

/* Streamlit renders prediction rows inside generated Markdown containers.
   Force their text black instead of inheriting a dark-theme text colour. */
div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] *,
div.st-emotion-cache-4cktc5,
div.st-emotion-cache-4cktc5 * {
    color: rgb(0, 0, 0) !important;
}
div[data-testid="stMarkdownContainer"] {
    font-family: "Source Sans Pro", "Source Sans", sans-serif !important;
    font-size: 1rem;
    max-width: 100%;
    width: 100%;
    overflow-wrap: break-word;
}
[data-testid="stSidebar"] { background:linear-gradient(160deg,#073d20,#07552a 52%,#063c20); }
[data-testid="stSidebar"] * { color:#ffffff !important; }
[data-testid="stSidebar"] > div:first-child { padding:1.4rem 1.1rem; }
[data-testid="stSidebar"] .stButton button { border-color:#78b88b; background:transparent; }

.brand { text-align:center; padding:1.3rem 0 1.1rem; }
.brand h1 { font:800 2.35rem 'Nunito',sans-serif; margin:0; color:#0b3d24; letter-spacing:-.04em; }
.brand p { margin:.2rem 0 0; color:#51655a; font-size:1rem; }
.side-brand { text-align:center; padding:1.1rem .4rem 1.8rem; border-bottom:1px solid rgba(255,255,255,.16); margin-bottom:1.5rem; }
.side-brand .leaf { font-size:2.2rem; }.side-brand h2 { font:800 1.55rem 'Nunito',sans-serif; line-height:1.15; margin:.3rem 0 0; }
.side-title { font:800 1rem 'Nunito',sans-serif; margin:1.35rem 0 .55rem; }
.side-copy { color:#dceee1 !important; font-size:.9rem; line-height:1.75; }
.model-line { display:flex; gap:.65rem; align-items:flex-start; margin:.85rem 0; }
.model-line b { display:block; font-size:.86rem; }.model-line span { color:#dceee1 !important; font-size:.86rem; }
.important { margin-top:1.5rem; border:1px solid #8a7e24; border-radius:13px; padding:.85rem; background:rgba(101,89,8,.25); }
.important b { color:#ffe269 !important; }.important p { color:#eff8ed !important; font-size:.82rem; line-height:1.6; margin:.45rem 0 0; }

.stat-card { min-height:96px; background:rgba(255,255,255,.94); border:1px solid #e2ebe4; border-radius:15px; padding:1rem; box-shadow:0 4px 16px rgba(29,82,45,.07); display:flex; gap:.8rem; align-items:center; }
.stat-icon { width:52px; height:52px; border-radius:50%; display:grid; place-items:center; font-size:1.55rem; flex:0 0 52px; }
.stat-card small { color:#62736a; display:block; }.stat-card strong { font:800 1.38rem 'Nunito',sans-serif; color:#102b20; display:block; line-height:1.1; margin:.15rem 0; }
.icon-green{background:#e1f3e5}.icon-blue{background:#e7f2ff}.icon-purple{background:#f0e8ff}.icon-orange{background:#fff0e5}
.panel { background:rgba(255,255,255,.9); border:1px solid #e0ebe3; border-radius:16px; padding:1.15rem; box-shadow:0 5px 18px rgba(25,75,40,.06); }
.panel-title { font:800 1.08rem 'Nunito',sans-serif; margin:0 0 .9rem; color:#153d27; }
[data-testid="stFileUploader"] { border:1.5px dashed #8acb9d; border-radius:12px; background:#fbfefb; padding:.3rem .7rem .8rem; }
[data-testid="stFileUploader"] label { color:#153d27 !important; font-weight:700; }
[data-testid="stFileUploader"] section { border:0 !important; background:transparent !important; }
[data-testid="stFileUploader"] button { background:#24954a !important; color:white !important; border:0 !important; border-radius:8px !important; }
.uploaded-label { font:800 1rem 'Nunito',sans-serif; margin:1.1rem 0 .55rem; }
.result-hero { border:1px solid #d6e9da; border-radius:13px; padding:1.15rem; background:linear-gradient(135deg,#f8fcf8,#eff8f1); display:flex; gap:1rem; align-items:center; }
.leaf-badge { width:66px; height:66px; border-radius:50%; background:#dff0c9; display:grid; place-items:center; font-size:2rem; border:1px solid #aed28b; }
.result-kicker { color:#53655b; font-size:.82rem; }.result-name { font:800 1.4rem 'Nunito',sans-serif; color:#11632f; margin:.15rem 0 .5rem; }
.conf-row { display:flex; justify-content:space-between; align-items:end; gap:1rem; }.conf-row b { font:800 1.35rem 'Nunito',sans-serif; color:#178438; }
.rank-row { display:grid; grid-template-columns:28px minmax(135px,1.2fr) 2fr 56px; gap:.7rem; align-items:center; margin:.85rem 0; font-size:.9rem; }
.rank-num { background:#d8f0dc; color:#125d2c; border-radius:5px; text-align:center; padding:.28rem 0; font-weight:700; }.rank-num.first { background:#2c963f; color:white; }
.rank-track { height:8px; background:#e8efea; border-radius:8px; overflow:hidden; }.rank-fill { height:100%; background:linear-gradient(90deg,#208d41,#075728); border-radius:8px; }
.footer-note { margin-top:1rem; background:#eff8ef; border:1px solid #dcecdf; border-radius:12px; padding:.8rem 1rem; color:#385143 !important; font-size:.84rem; }
button[kind="primary"] { background:#1c8b40 !important; border:0 !important; border-radius:8px !important; font-weight:700 !important; min-height:2.6rem; }
.stProgress > div > div { background:#269645; }
@media(max-width:800px) { .block-container{padding:1rem}.brand h1{font-size:1.75rem}.rank-row{grid-template-columns:25px 1fr 52px}.rank-track{grid-column:2/4;grid-row:2}.stat-card{margin-bottom:.4rem} }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=10)
def backend_health() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=4)
        return response.json() if response.ok else {"status": "offline", "detail": response.text}
    except requests.RequestException as exc:
        return {"status": "offline", "detail": str(exc)}


def label_text(label: str) -> str:
    return label.replace("___", " — ").replace("__", " — ").replace("_", " ")


def stat(icon: str, icon_class: str, title: str, value: str, detail: str) -> None:
    st.markdown(
        f"<div class='stat-card'><div class='stat-icon {icon_class}'>{icon}</div>"
        f"<div><small>{title}</small><strong>{value}</strong><small>{detail}</small></div></div>",
        unsafe_allow_html=True,
    )


if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None

health = backend_health()
model_ready = health.get("model_ready", False)

with st.sidebar:
    st.markdown("<div class='side-brand'><div class='leaf'>🌱</div><h2>Plant Disease<br>Classifier</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='side-title'>ⓘ About</div><div class='side-copy'>Upload a plant leaf image and our AI model will identify the disease or condition.</div>", unsafe_allow_html=True)
    st.markdown("<div class='side-title'>⌁ Model Information</div>", unsafe_allow_html=True)
    for icon, name, value in [("⚙", "Model", MODEL_NAME), ("✤", "Classes", str(CLASS_COUNT)), ("▣", "Input Size", INPUT_SIZE), ("◉", "Framework", "TensorFlow / Keras")]:
        st.markdown(f"<div class='model-line'><span>{icon}</span><div><b>{name}</b><span>{value}</span></div></div>", unsafe_allow_html=True)
    if model_ready:
        st.success("Model online")
    else:
        st.warning("Model setup required")
    st.markdown("<div class='important'><b>⚠ Important</b><p>This model supports 15 plant conditions. Predictions are AI guidance and do not replace expert advice.</p></div>", unsafe_allow_html=True)

st.markdown("<div class='brand'><h1>🌱 Plant Disease Classifier</h1><p>AI-powered plant leaf disease classification using EfficientNetB1</p></div>", unsafe_allow_html=True)
stats = st.columns(4)
with stats[0]: stat("◎", "icon-green", "Test Accuracy", "98.06%", "Final evaluation")
with stats[1]: stat("↗", "icon-blue", "Validation Accuracy", "97.97%", "Final evaluation")
with stats[2]: stat("♟", "icon-purple", "Classes", "15", "Plant conditions")
with stats[3]: stat("▣", "icon-orange", "Model", "EfficientNetB1", "Transfer learning")

left, right = st.columns([.94, 1.5], gap="medium")
with left:
    st.markdown("<div class='panel-title'>↥ Upload Leaf Image</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drag and drop an image here, or click to browse", type=["jpg", "jpeg", "png", "webp"], label_visibility="visible")
    if uploaded and uploaded.size > 10 * 1024 * 1024:
        st.error("Image exceeds the 10 MB limit.")
        uploaded = None
    if uploaded:
        st.markdown("<div class='uploaded-label'>▣ Uploaded Image</div>", unsafe_allow_html=True)
        st.image(uploaded, width="stretch")
        st.caption(f"{uploaded.name} · {(uploaded.size / 1024):.1f} KB · Image loaded successfully")
    analyze = st.button("🔍 Analyze Leaf", type="primary", use_container_width=True, disabled=uploaded is None or not model_ready)
    if uploaded and not model_ready:
        st.info("Start the backend or confirm that the model files are in the model folder.")

    if analyze and uploaded:
        try:
            with st.spinner("Analysing leaf pattern…"):
                response = requests.post(f"{API_BASE_URL}/predict", files={"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}, timeout=60)
            if not response.ok:
                st.error(response.json().get("detail", "Diagnosis failed. Please try again."))
            else:
                st.session_state.result = response.json()
                best = st.session_state.result["prediction"]
                st.session_state.history.insert(0, {"time": datetime.now().strftime("%H:%M"), "label": label_text(best["label"]), "confidence": best["confidence"]})
        except requests.RequestException:
            st.error("Could not reach the API. Check that the backend is running.")

with right:
    st.markdown("<div class='panel-title'>⌕ Prediction Result</div>", unsafe_allow_html=True)
    result = st.session_state.result
    if result:
        best = result["prediction"]
        confidence = best["confidence"]
        status = "High confidence" if confidence >= .80 else "Moderate confidence" if confidence >= .50 else "Low confidence"
        st.markdown(f"<div class='result-hero'><div class='leaf-badge'>🌿</div><div style='flex:1'><div class='result-kicker'>Predicted Disease / Condition</div><div class='result-name'>{label_text(best['label'])}</div><div class='conf-row'><span>Confidence score</span><b>{confidence:.2%}</b></div></div></div>", unsafe_allow_html=True)
        st.progress(confidence)
        st.caption(f"● {status}")
        st.markdown("<div class='panel-title' style='margin-top:1.2rem'>▥ Top Predictions</div>", unsafe_allow_html=True)
        for number, item in enumerate(result["top_predictions"], start=1):
            pct = item["confidence"] * 100
            first = " first" if number == 1 else ""
            st.markdown(f"<div class='rank-row'><span class='rank-num{first}'>{number}</span><b>{label_text(item['label'])}</b><div class='rank-track'><div class='rank-fill' style='width:{max(pct, .4):.2f}%'></div></div><span>{pct:.2f}%</span></div>", unsafe_allow_html=True)
        if result["low_confidence"]:
            st.warning("The confidence is low. Try a brighter, closer image focused on the affected part of the leaf.")
    else:
        st.markdown("<div class='panel' style='min-height:280px;display:grid;place-items:center;text-align:center'><div><div style='font-size:3rem'>🍃</div><div class='panel-title'>Ready for a diagnosis</div><p>Upload a clear leaf image to see the predicted condition, confidence score, and top matches.</p></div></div>", unsafe_allow_html=True)

st.markdown("<div class='footer-note'><b>ⓘ About this model</b><br>EfficientNetB1 · 240 × 240 input · 15 plant conditions · TensorFlow / Keras. Use clear, well-lit photos for the most useful predictions.</div>", unsafe_allow_html=True)
