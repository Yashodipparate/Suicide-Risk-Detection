import re
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# NLTK stopwords (optional but matches your notebook)
import nltk
from nltk.corpus import stopwords

# ----------------------------
# Page config + basic styling
# ----------------------------
st.set_page_config(
    page_title="Suicide Risk Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ✅ Professional background + UI styling (NO model logic changes)
CUSTOM_CSS = """
<style>
/* ====== App Background ====== */
.stApp {
    background: linear-gradient(135deg, rgba(13, 18, 32, 0.96), rgba(22, 30, 55, 0.92));
    background-attachment: fixed;
}

/* Subtle dot pattern overlay */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image: radial-gradient(rgba(255,255,255,0.06) 1px, transparent 1px);
    background-size: 26px 26px;
    opacity: 0.22;
    z-index: 0;
}

/* Keep content above overlay */
.block-container {
    position: relative;
    z-index: 1;
    padding-top: 1.4rem;
}

/* Text */
.small-note {
    font-size: 0.90rem;
    opacity: 0.95;
    color: rgba(255,255,255,0.85);
}

/* Cards */
.card {
    padding: 1rem 1.2rem;
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 16px;
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(8px);
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.10);
}

/* Inputs styling */
textarea, input, .stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}

/* Metrics styling */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    padding: 0.75rem;
    border-radius: 14px;
}

/* Tables */
div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 0.25rem;
    border: 1px solid rgba(255,255,255,0.10);
}

/* ✅ Smaller separators */
hr {
    margin: 0.35rem 0;
    opacity: 0.18;
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.18);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------
# Helpers: safe loading
# ----------------------------
DEFAULT_MODEL_FILE = "suicide_logreg_model.joblib"
DEFAULT_VECT_FILE = "tfidf_vectorizer.joblib"

def find_file(filename: str) -> Path | None:
    """Try to find file in current directory and common subdirs."""
    candidates = [
        Path(filename),
        Path.cwd() / filename,
        Path.cwd() / "models" / filename,
        Path.cwd() / "artifacts" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

@st.cache_resource
def load_artifacts(model_name: str = DEFAULT_MODEL_FILE, vect_name: str = DEFAULT_VECT_FILE):
    model_path = find_file(model_name)
    vect_path = find_file(vect_name)

    if model_path is None:
        raise FileNotFoundError(
            f"Model file not found: '{model_name}'. Put it in the same folder as app.py "
            f"or inside ./models or ./artifacts."
        )
    if vect_path is None:
        raise FileNotFoundError(
            f"Vectorizer file not found: '{vect_name}'. Put it in the same folder as app.py "
            f"or inside ./models or ./artifacts."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)
    return model, vectorizer, str(model_path), str(vect_path)

@st.cache_resource
def get_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words("english"))

STOP_WORDS = get_stopwords()

# ----------------------------
# Preprocessing (fast + reliable)
# ----------------------------
URL_RE = re.compile(r"http\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
NUM_RE = re.compile(r"\d+")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
WS_RE = re.compile(r"\s+")

def preprocess(text: str) -> str:
    """Matches your notebook logic but optimized."""
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""

    # remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)

    # remove stopwords
    text = " ".join(w for w in text.split() if w.lower() not in STOP_WORDS)

    # remove emojis/special unicode
    text = NON_ASCII_RE.sub(" ", text)

    # remove urls, tags, numbers; lowercase
    text = text.lower()
    text = URL_RE.sub("", text)
    text = HTML_RE.sub("", text)
    text = NUM_RE.sub("", text)

    # normalize whitespace
    text = WS_RE.sub(" ", text).strip()
    return text

# ----------------------------
# Prediction helpers
# ----------------------------
def get_suicide_class_index(model) -> int | None:
    """Find index for suicide class in predict_proba output."""
    if not hasattr(model, "classes_"):
        return None
    classes = list(model.classes_)

    # string labels
    for i, c in enumerate(classes):
        if isinstance(c, str) and c.strip().lower() in ("suicide", "suicidal"):
            return i

    # numeric labels where 1 means suicide
    if 1 in classes:
        return classes.index(1)

    # fallback: if binary numeric like [0,1] not found or weird ordering
    if len(classes) == 2:
        return 1

    return None

def predict(model, vectorizer, raw_text: str):
    cleaned = preprocess(raw_text)
    if not cleaned:
        return None

    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    prob = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(vec)[0]
        idx = get_suicide_class_index(model)
        if idx is not None:
            prob = float(p[idx])
        else:
            prob = float(np.max(p))

    # Format label for UI
    label_str = str(pred).strip().lower()
    if isinstance(pred, (int, np.integer)):
        label = "Suicide" if int(pred) == 1 else "Non-suicide"
    else:
        label = "Suicide" if label_str in ("suicide", "suicidal", "1", "true", "yes") else "Non-suicide"

    return {
        "raw": raw_text,
        "cleaned": cleaned,
        "label": label,
        "prob": prob,
    }

def risk_band(prob: float | None):
    if prob is None:
        return ("Unknown", "badge")
    if prob < 0.35:
        return ("Low", "badge")
    if prob < 0.70:
        return ("Medium", "badge")
    return ("High", "badge")

# ----------------------------
# Sidebar: controls + artifact info
# ----------------------------
st.sidebar.title("⚙️ Controls")

with st.sidebar.expander("Model files", expanded=True):
    model_file = st.text_input("Model filename", value=DEFAULT_MODEL_FILE)
    vect_file = st.text_input("Vectorizer filename", value=DEFAULT_VECT_FILE)
    st.caption("Tip: keep both .joblib files in the same folder as app.py")

with st.sidebar.expander("Decision settings", expanded=True):
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    show_cleaned = st.checkbox("Show cleaned text", value=True)
    save_history = st.checkbox("Save prediction history", value=True)

with st.sidebar.expander("About", expanded=False):
    st.markdown(
        "- This is a **project/educational** demo.\n"
        "- It predicts from **TF-IDF + Logistic Regression**.\n"
        "- Threshold controls how strict the app is."
    )

# Load artifacts with user-provided names
try:
    model, vectorizer, model_path, vect_path = load_artifacts(model_file, vect_file)
    st.sidebar.success("Artifacts loaded ✅")
    st.sidebar.caption(f"Model: {model_path}")
    st.sidebar.caption(f"Vectorizer: {vect_path}")
except Exception as e:
    st.sidebar.error("Failed to load artifacts ❌")
    st.sidebar.exception(e)
    st.stop()

# ----------------------------
# Main UI
# ----------------------------
st.title("🧠 Suicide Risk Detection")
st.markdown("<div class='small-note'>Enter a message.</div>", unsafe_allow_html=True)

with st.expander("⚠️ Disclaimer (read)", expanded=True):
    st.warning(
        "This application is for **academic/project** purposes only and is **not** a medical diagnosis tool. "
        "If you or someone is at immediate risk, contact local emergency services or a trusted professional."
    )

tab1, tab2 = st.tabs(["🔎 Single Prediction", "🧪 Batch Test"])

# ---------- TAB 1: Single
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    colA, colB = st.columns([3, 2], vertical_alignment="top")

    with colA:
        user_text = st.text_area(
            "Input text",
            height=160,
            placeholder="Type or paste a message here...",
        )

    with colB:
        st.markdown("**How decision works**")
        st.write(f"- Probability ≥ **{threshold:.2f}** → **Suicide**")
        st.write(f"- Probability < **{threshold:.2f}** → **Non-suicide**")
        st.caption("You can tune threshold to reduce false alarms or catch more risk.")

    st.markdown("</div>", unsafe_allow_html=True)

    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze:
        if not user_text.strip():
            st.error("Please enter some text first.")
        else:
            result = predict(model, vectorizer, user_text)
            if result is None:
                st.error("After cleaning, the text became empty. Please enter meaningful text.")
            else:
                prob = result["prob"]
                band, _ = risk_band(prob)

                # Decide final output
                if prob is None:
                    decision = result["label"]
                else:
                    decision = "Suicide" if prob >= threshold else "Non-suicide"

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Result")

                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction", decision)
                c2.metric("Risk band", band)
                if prob is not None:
                    c3.metric("Suicide probability", f"{prob*100:.2f}%")
                else:
                    c3.metric("Suicide probability", "N/A")

                if prob is not None:
                    st.progress(min(max(prob, 0.0), 1.0))

                if decision == "Suicide":
                    st.error("⚠️ The model indicates higher suicide-risk language based on the learned patterns.")
                else:
                    st.success("✅ The model indicates non-suicidal language based on the learned patterns.")

                if show_cleaned:
                    with st.expander("See cleaned text used for prediction"):
                        st.code(result["cleaned"])

                st.markdown("</div>", unsafe_allow_html=True)

                # Save history
                if save_history:
                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    st.session_state["history"].append({
                        "text": result["raw"][:120] + ("..." if len(result["raw"]) > 120 else ""),
                        "prediction": decision,
                        "risk_band": band,
                        "probability": None if prob is None else round(prob, 4),
                    })

    # History display
    if save_history and "history" in st.session_state and len(st.session_state["history"]) > 0:
        st.markdown("### 🧾 Prediction History (this session)")
        hist_df = pd.DataFrame(st.session_state["history"]).tail(10)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ---------- TAB 2: Batch
with tab2:
    st.markdown("Paste multiple lines (one text per line) to test quickly.")
    batch_text = st.text_area(
        "Batch input (one message per line)",
        height=180,
        placeholder="Line 1...\nLine 2...\nLine 3..."
    )

    run_batch = st.button("🧪 Run batch prediction", use_container_width=True)
    if run_batch:
        lines = [ln.strip() for ln in batch_text.splitlines() if ln.strip()]
        if not lines:
            st.error("Please enter at least one line.")
        else:
            rows = []
            for ln in lines:
                res = predict(model, vectorizer, ln)
                if res is None:
                    rows.append({"text": ln, "cleaned": "", "prediction": "N/A", "probability": None})
                    continue
                prob = res["prob"]
                if prob is None:
                    decision = res["label"]
                else:
                    decision = "Suicide" if prob >= threshold else "Non-suicide"
                rows.append({
                    "text": ln,
                    "cleaned": res["cleaned"],
                    "prediction": decision,
                    "probability": None if prob is None else round(prob, 4),
                })

            out_df = pd.DataFrame(rows)
            st.dataframe(out_df, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download results as CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="suicide_detection_batch_results.csv",
                mime="text/csv",
                use_container_width=True
            )
