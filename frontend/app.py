
import time
from typing import Any, Dict
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CardioML | AI Health Analytics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APP DATA — PROVIDED TRAINING ANALYTICS
# ============================================================

DATASET_ROWS = 918
DATASET_COLUMNS = 12
TEST_SUPPORT = 184

ACCURACY = 0.8859
PRECISION = 0.8857
RECALL = 0.9118
F1_SCORE = 0.8986

CONFUSION_MATRIX = {
    "actual_0_pred_0": 70,
    "actual_0_pred_1": 12,
    "actual_1_pred_0": 9,
    "actual_1_pred_1": 93,
}

CLASS_METRICS = pd.DataFrame(
    {
        "Class": ["0 — No Heart Disease", "1 — Heart Disease"],
        "Precision": [0.89, 0.89],
        "Recall": [0.85, 0.91],
        "F1 Score": [0.87, 0.90],
        "Support": [82, 102],
    }
)

FEATURES = [
    ("Age", "Numeric", "0–100"),
    ("Sex", "Categorical", "M / F"),
    ("ChestPainType", "Categorical", "ATA / NAP / ASY / TA"),
    ("RestingBP", "Numeric", "30–200"),
    ("Cholesterol", "Numeric", "100–600"),
    ("FastingBS", "Numeric", "0 / 1"),
    ("RestingECG", "Categorical", "Normal / ST / LVH"),
    ("MaxHR", "Numeric", "40–220"),
    ("ExerciseAngina", "Categorical", "N / Y"),
    ("Oldpeak", "Numeric", "Float"),
    ("ST_Slope", "Categorical", "Up / Flat / Down"),
]

NUMERIC_FEATURES = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]


# ============================================================
# URL + REQUEST HELPERS
# ============================================================

def get_default_api_url() -> str:
    try:
        if "API_URL" in st.secrets:
            return str(st.secrets["API_URL"]).rstrip("/")
    except Exception:
        pass
    return "https://heart-disease-prediction-mnat.onrender.com/"


def validate_api_url(raw_url: str) -> tuple[bool, str, str]:
    url = raw_url.strip()

    if not url:
        return False, "", "Please enter your FastAPI backend URL."

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "", "The backend URL format is invalid."

    if parsed.scheme not in {"http", "https"}:
        return False, "", "Use a URL starting with http:// or https://."

    if not parsed.netloc:
        return False, "", "Please enter a complete backend URL."

    if parsed.username or parsed.password:
        return False, "", "Embedded usernames/passwords are not supported."

    return True, url.rstrip("/"), ""


def api_error_message(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.InvalidURL):
        return "The backend URL is invalid. Check the address in the sidebar."

    if isinstance(exc, requests.exceptions.MissingSchema):
        return "The backend URL must start with http:// or https://."

    if isinstance(exc, requests.exceptions.Timeout):
        return (
            "The backend took too long to respond. "
            "Your Render service may be waking up. Please try again."
        )

    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            "The backend could not be reached. "
            "Check the URL and make sure the Render service is running."
        )

    if isinstance(exc, requests.exceptions.RequestException):
        return "The request could not be completed. Please check the backend."

    return "An unexpected error occurred while contacting the backend."


def extract_backend_error(response: requests.Response) -> str:
    try:
        data = response.json()

        if isinstance(data, dict):
            detail = data.get("detail")

            if isinstance(detail, str) and detail.strip():
                return detail.strip()

            if isinstance(detail, list):
                messages = []
                for item in detail:
                    if isinstance(item, dict) and item.get("msg"):
                        messages.append(str(item["msg"]))
                if messages:
                    return " • ".join(messages)

        return "The backend returned an error."
    except (ValueError, TypeError):
        return "The backend returned an error."


def check_backend(base_url: str) -> tuple[bool, str, float | None]:
    started = time.perf_counter()

    try:
        response = requests.get(
            f"{base_url}/",
            timeout=(5, 10),
            headers={"Accept": "application/json"},
        )

        latency = (time.perf_counter() - started) * 1000

        if response.ok:
            return True, "Backend is online", latency

        if response.status_code == 404:
            return (
                False,
                "Server responded, but the FastAPI root endpoint was not found.",
                latency,
            )

        return False, f"Backend responded with HTTP {response.status_code}.", latency

    except requests.exceptions.RequestException as exc:
        return False, api_error_message(exc), None


# ============================================================
# SESSION STATE
# ============================================================

state_defaults = {
    "last_result": None,
    "last_payload": None,
    "last_latency_ms": None,
    "api_status": None,
}

for key, value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# STYLE
# ============================================================

st.markdown(
    r"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(37,99,235,.16), transparent 27%),
                radial-gradient(circle at 94% 7%, rgba(6,182,212,.11), transparent 25%),
                linear-gradient(180deg, #07101c 0%, #0a1422 55%, #08111f 100%);
            font-family: "DM Sans", sans-serif;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1.75rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.03em;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(7,15,27,.98), rgba(10,22,38,.98));
            border-right: 1px solid rgba(148,163,184,.13);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 11px;
            margin-bottom: 1.45rem;
        }

        .brand-mark {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            background: linear-gradient(135deg, rgba(96,165,250,.24), rgba(34,211,238,.16));
            border: 1px solid rgba(96,165,250,.22);
            box-shadow: 0 12px 38px rgba(37,99,235,.17);
        }

        .brand-title {
            font-family: "Space Grotesk", sans-serif;
            color: #f8fafc;
            font-size: 1.08rem;
            font-weight: 700;
            line-height: 1;
        }

        .brand-subtitle {
            color: #718098;
            font-size: .74rem;
            margin-top: 4px;
        }

        .hero {
            padding: 1.8rem 1.9rem 1.6rem;
            border-radius: 24px;
            border: 1px solid rgba(148,163,184,.13);
            background: linear-gradient(135deg, rgba(16,36,62,.94), rgba(10,22,38,.83));
            box-shadow: 0 30px 75px rgba(0,0,0,.20);
            margin-bottom: 1.15rem;
        }

        .hero-kicker {
            color: #8cbcff;
            font-size: .72rem;
            text-transform: uppercase;
            letter-spacing: .13em;
            font-weight: 700;
        }

        .hero h1 {
            margin: 7px 0 8px;
            font-size: clamp(2.1rem, 4vw, 3.55rem);
            line-height: 1;
        }

        .hero-gradient {
            background: linear-gradient(90deg, #f8fafc, #bfdbfe 48%, #67e8f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-copy {
            max-width: 850px;
            color: #9fb0c5;
            line-height: 1.65;
            margin-bottom: .9rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .chip {
            border: 1px solid rgba(255,255,255,.06);
            background: rgba(255,255,255,.035);
            border-radius: 999px;
            padding: 6px 10px;
            color: #cad5e2;
            font-size: .74rem;
        }

        .section-kicker {
            color: #7e8da4;
            text-transform: uppercase;
            letter-spacing: .12em;
            font-size: .69rem;
            font-weight: 700;
            margin-bottom: 3px;
        }

        .section-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 2px;
        }

        .section-subtitle {
            color: #75849a;
            font-size: .85rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            border: 1px solid rgba(148,163,184,.12);
            background: linear-gradient(180deg, rgba(15,29,48,.84), rgba(11,22,37,.77));
            border-radius: 18px;
            padding: 1.05rem 1.1rem;
            min-height: 112px;
            box-shadow: 0 14px 35px rgba(0,0,0,.12);
        }

        .metric-label {
            color: #7f8ea3;
            font-size: .68rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-weight: 700;
        }

        .metric-value {
            font-family: "Space Grotesk", sans-serif;
            color: #f8fafc;
            font-size: 1.55rem;
            line-height: 1.1;
            font-weight: 700;
            margin-top: 7px;
        }

        .metric-caption {
            color: #718098;
            font-size: .72rem;
            margin-top: 5px;
        }

        .panel {
            border: 1px solid rgba(148,163,184,.12);
            background: linear-gradient(180deg, rgba(15,29,48,.82), rgba(11,22,37,.76));
            border-radius: 19px;
            padding: 1.2rem;
            box-shadow: 0 16px 42px rgba(0,0,0,.13);
        }

        .mini-stat {
            padding: 11px 13px;
            border-radius: 13px;
            background: rgba(255,255,255,.025);
            border: 1px solid rgba(255,255,255,.05);
        }

        .mini-label {
            color: #708097;
            font-size: .67rem;
            text-transform: uppercase;
            letter-spacing: .07em;
            font-weight: 700;
        }

        .mini-value {
            color: #eaf1f8;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1rem;
            font-weight: 700;
            margin-top: 3px;
        }

        .matrix {
            width: 100%;
            border-collapse: separate;
            border-spacing: 8px;
        }

        .matrix th,
        .matrix td {
            border-radius: 12px;
            text-align: center;
            padding: 14px 10px;
            border: 1px solid rgba(255,255,255,.05);
        }

        .matrix th {
            color: #8090a7;
            font-size: .69rem;
            text-transform: uppercase;
            letter-spacing: .06em;
            background: rgba(255,255,255,.02);
        }

        .matrix td {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.28rem;
            font-weight: 700;
            color: #f8fafc;
            background: rgba(255,255,255,.025);
        }

        .matrix .diag {
            background: rgba(16,185,129,.09);
            border-color: rgba(52,211,153,.12);
        }

        .matrix .offdiag {
            background: rgba(251,113,133,.07);
            border-color: rgba(251,113,133,.10);
        }

        .result-card {
            border-radius: 20px;
            padding: 1.35rem;
            border: 1px solid rgba(255,255,255,.06);
        }

        .result-positive {
            background: linear-gradient(135deg, rgba(127,29,29,.28), rgba(69,10,10,.10));
            border-color: rgba(251,113,133,.2);
        }

        .result-negative {
            background: linear-gradient(135deg, rgba(6,78,59,.25), rgba(6,78,59,.09));
            border-color: rgba(52,211,153,.18);
        }

        .result-badge {
            display: inline-block;
            padding: 5px 9px;
            border-radius: 999px;
            background: rgba(255,255,255,.045);
            color: #cbd5e1;
            font-size: .68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .09em;
        }

        .result-title {
            color: #f8fafc;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.65rem;
            font-weight: 700;
            margin: .65rem 0 .3rem;
        }

        .result-copy {
            color: #9fb0c5;
            font-size: .88rem;
            line-height: 1.6;
        }

        .footnote {
            color: #617088;
            text-align: center;
            font-size: .72rem;
            margin-top: 1.6rem;
        }

        .api-online {
            color: #6ee7b7;
            font-weight: 600;
        }

        .api-offline {
            color: #fda4af;
            font-weight: 600;
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stDownloadButton"] button {
            border-radius: 12px !important;
            min-height: 44px;
            font-weight: 700 !important;
        }

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div {
            background: rgba(255,255,255,.022) !important;
            border-color: rgba(148,163,184,.15) !important;
            border-radius: 11px !important;
        }

        @media (max-width: 850px) {
            .hero {
                padding: 1.35rem;
                border-radius: 19px;
            }

            .hero h1 {
                font-size: 2.35rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">🫀</div>
            <div>
                <div class="brand-title">CardioML</div>
                <div class="brand-subtitle">AI / ML model dashboard</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Backend")
    raw_api_url = st.text_input(
        "FastAPI URL",
        value=get_default_api_url(),
        placeholder="https://your-api.onrender.com",
        help="Use the backend URL only. The app adds / and /predict automatically.",
    )

    url_is_valid, api_url, url_message = validate_api_url(raw_api_url)

    if not url_is_valid:
        st.warning(url_message, icon="⚠️")

    c1, c2 = st.columns(2)

    with c1:
        check_api = st.button(
            "Check API",
            use_container_width=True,
            disabled=not url_is_valid,
        )

    with c2:
        if st.button("Reset", use_container_width=True):
            st.session_state.last_result = None
            st.session_state.last_payload = None
            st.session_state.last_latency_ms = None
            st.session_state.api_status = None
            st.rerun()

    if check_api:
        online, message, latency = check_backend(api_url)

        if online:
            st.session_state.api_status = {
                "online": True,
                "message": f"Online • {latency:.0f} ms",
            }
            st.success("FastAPI backend is reachable")
        else:
            st.session_state.api_status = {
                "online": False,
                "message": "Connection unavailable",
            }
            st.error(message)

    if st.session_state.api_status:
        status = st.session_state.api_status
        css = "api-online" if status["online"] else "api-offline"
        icon = "●" if status["online"] else "○"

        st.markdown(
            f'<div class="small-note"><span class="{css}">{icon}</span> {status["message"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Model artifact")
    st.markdown(
        """
        <div class="mini-stat">
            <div class="mini-label">Saved model</div>
            <div class="mini-value">model.pkl</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="small-note">
            This dashboard uses the training metrics supplied for your
            heart-disease model. The prediction endpoint remains the source
            of truth for live inference.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="small-note">
            <strong>Notice:</strong> This is an AI/ML demonstration.
            A model prediction is not a medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">AI / ML analytics • deployed inference</div>
        <h1>Cardio<span class="hero-gradient">ML</span></h1>
        <p class="hero-copy">
            A professional model-monitoring and inference dashboard for your
            heart-disease prediction pipeline. Explore the training dataset,
            evaluate model performance, inspect classification behavior, and
            run live predictions through FastAPI.
        </p>
        <div class="chip-row">
            <span class="chip">918 training records</span>
            <span class="chip">11 input features</span>
            <span class="chip">88.59% accuracy</span>
            <span class="chip">89.86% F1 score</span>
            <span class="chip">FastAPI inference</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP NAV
# ============================================================

overview_tab, predict_tab, schema_tab = st.tabs(
    ["📊 Model Overview", "🧠 Live Prediction", "🧬 Feature Schema"]
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

with overview_tab:
    st.markdown(
        """
        <div class="section-kicker">Model performance</div>
        <div class="section-title">Training & evaluation overview</div>
        <div class="section-subtitle">
            Metrics below are from the training analytics you provided.
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    metrics = [
        ("Accuracy", f"{ACCURACY * 100:.2f}%", "Overall correct predictions"),
        ("Precision", f"{PRECISION * 100:.2f}%", "Positive prediction quality"),
        ("Recall", f"{RECALL * 100:.2f}%", "Positive cases detected"),
        ("F1 Score", f"{F1_SCORE * 100:.2f}%", "Precision / recall balance"),
    ]

    for col, (label, value, caption) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-caption">{caption}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.05, .95], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-kicker">Dataset profile</div>
            <div class="section-title">Training dataset snapshot</div>
            """,
            unsafe_allow_html=True,
        )

        a, b, c = st.columns(3)

        with a:
            st.markdown(
                f"""
                <div class="mini-stat">
                    <div class="mini-label">Rows</div>
                    <div class="mini-value">{DATASET_ROWS:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with b:
            st.markdown(
                f"""
                <div class="mini-stat">
                    <div class="mini-label">Columns</div>
                    <div class="mini-value">{DATASET_COLUMNS}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c:
            st.markdown(
                f"""
                <div class="mini-stat">
                    <div class="mini-label">Eval support</div>
                    <div class="mini-value">{TEST_SUPPORT}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**Feature composition**")

        feature_profile = pd.DataFrame(
            {
                "Type": ["Numeric", "Categorical"],
                "Count": [len(NUMERIC_FEATURES), len(CATEGORICAL_FEATURES)],
            }
        )

        st.dataframe(
            feature_profile,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="small-note">
                The dataset contains 6 numeric and 5 categorical predictor features,
                plus the binary <code>HeartDisease</code> target.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-kicker">Classification behavior</div>
            <div class="section-title">Confusion matrix</div>
            <div class="section-subtitle">
                Evaluation support: 184 samples
            </div>
            """,
            unsafe_allow_html=True,
        )

        matrix_html = f"""
        <table class="matrix">
            <tr>
                <th></th>
                <th>Predicted 0</th>
                <th>Predicted 1</th>
            </tr>
            <tr>
                <th>Actual 0</th>
                <td class="diag">{CONFUSION_MATRIX["actual_0_pred_0"]}</td>
                <td class="offdiag">{CONFUSION_MATRIX["actual_0_pred_1"]}</td>
            </tr>
            <tr>
                <th>Actual 1</th>
                <td class="offdiag">{CONFUSION_MATRIX["actual_1_pred_0"]}</td>
                <td class="diag">{CONFUSION_MATRIX["actual_1_pred_1"]}</td>
            </tr>
        </table>
        """

        st.markdown(matrix_html, unsafe_allow_html=True)

        x, y = st.columns(2)

        with x:
            st.markdown(
                f"""
                <div class="mini-stat">
                    <div class="mini-label">True negatives</div>
                    <div class="mini-value">70</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with y:
            st.markdown(
                f"""
                <div class="mini-stat">
                    <div class="mini-label">True positives</div>
                    <div class="mini-value">93</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        x2, y2 = st.columns(2)

        with x2:
            st.markdown(
                """
                <div class="mini-stat">
                    <div class="mini-label">False positives</div>
                    <div class="mini-value">12</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with y2:
            st.markdown(
                """
                <div class="mini-stat">
                    <div class="mini-label">False negatives</div>
                    <div class="mini-value">9</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    l2, r2 = st.columns([1.2, .8], gap="large")

    with l2:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-kicker">Per-class metrics</div>
            <div class="section-title">Classification report</div>
            <div class="section-subtitle">
                Values rounded to two decimal places, matching the supplied report.
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_metrics = CLASS_METRICS.copy()
        for col in ["Precision", "Recall", "F1 Score"]:
            display_metrics[col] = (display_metrics[col] * 100).map(
                lambda x: f"{x:.0f}%"
            )

        st.dataframe(
            display_metrics,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-kicker">Model summary</div>
            <div class="section-title">Evaluation snapshot</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="mini-stat">
                <div class="mini-label">Dataset shape</div>
                <div class="mini-value">({DATASET_ROWS}, {DATASET_COLUMNS})</div>
            </div>
            <br>
            <div class="mini-stat">
                <div class="mini-label">Numeric features</div>
                <div class="mini-value">{len(NUMERIC_FEATURES)}</div>
            </div>
            <br>
            <div class="mini-stat">
                <div class="mini-label">Categorical features</div>
                <div class="mini-value">{len(CATEGORICAL_FEATURES)}</div>
            </div>
            <br>
            <div class="mini-stat">
                <div class="mini-label">Binary target</div>
                <div class="mini-value">HeartDisease</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# LIVE PREDICTION
# ============================================================

with predict_tab:
    st.markdown(
        """
        <div class="section-kicker">Inference</div>
        <div class="section-title">Run a live prediction</div>
        <div class="section-subtitle">
            Inputs are validated by the FastAPI Pydantic schema before reaching the model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form", clear_on_submit=False):
        left_panel, right_panel = st.columns([1, 1], gap="large")

        with left_panel:
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            st.markdown("#### Patient profile")

            c1, c2 = st.columns(2)

            with c1:
                age = st.number_input(
                    "Age",
                    min_value=0,
                    max_value=100,
                    value=55,
                    step=1,
                )

            with c2:
                sex = st.selectbox(
                    "Sex",
                    ["M", "F"],
                )

            chest_pain = st.selectbox(
                "Chest Pain Type",
                ["ATA", "NAP", "ASY", "TA"],
                index=2,
                format_func=lambda x: {
                    "ATA": "ATA — Atypical Angina",
                    "NAP": "NAP — Non-Anginal Pain",
                    "ASY": "ASY — Asymptomatic",
                    "TA": "TA — Typical Angina",
                }[x],
            )

            st.markdown("#### Resting measurements")

            c3, c4 = st.columns(2)

            with c3:
                resting_bp = st.number_input(
                    "Resting Blood Pressure",
                    min_value=30,
                    max_value=200,
                    value=140,
                    step=1,
                )

            with c4:
                cholesterol = st.number_input(
                    "Cholesterol",
                    min_value=100,
                    max_value=600,
                    value=240,
                    step=1,
                )

            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                [0, 1],
                format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
            )

            st.markdown("</div>", unsafe_allow_html=True)

        with right_panel:
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            st.markdown("#### Cardiac / ECG indicators")

            resting_ecg = st.selectbox(
                "Resting ECG",
                ["Normal", "ST", "LVH"],
                format_func=lambda x: {
                    "Normal": "Normal",
                    "ST": "ST-T Wave Abnormality",
                    "LVH": "Left Ventricular Hypertrophy",
                }[x],
            )

            c5, c6 = st.columns(2)

            with c5:
                max_hr = st.number_input(
                    "Maximum Heart Rate",
                    min_value=40,
                    max_value=220,
                    value=150,
                    step=1,
                )

            with c6:
                exercise_angina = st.selectbox(
                    "Exercise-Induced Angina",
                    ["N", "Y"],
                    format_func=lambda x: "No (N)" if x == "N" else "Yes (Y)",
                )

            c7, c8 = st.columns(2)

            with c7:
                oldpeak = st.number_input(
                    "Oldpeak",
                    min_value=-10.0,
                    max_value=20.0,
                    value=1.2,
                    step=0.1,
                    format="%.1f",
                )

            with c8:
                st_slope = st.selectbox(
                    "ST Slope",
                    ["Up", "Flat", "Down"],
                    index=1,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        p1, p2, p3 = st.columns([1.5, .8, .8])

        with p1:
            predict_clicked = st.form_submit_button(
                "Run AI Prediction  →",
                use_container_width=True,
                type="primary",
            )

        with p2:
            st.form_submit_button(
                "Load Sample",
                use_container_width=True,
            )

        with p3:
            st.form_submit_button(
                "Clear",
                use_container_width=True,
            )

    if predict_clicked:
        payload: Dict[str, Any] = {
            "Age": int(age),
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": int(resting_bp),
            "Cholesterol": int(cholesterol),
            "FastingBS": int(fasting_bs),
            "RestingECG": resting_ecg,
            "MaxHR": int(max_hr),
            "ExerciseAngina": exercise_angina,
            "Oldpeak": float(oldpeak),
            "ST_Slope": st_slope,
        }

        if not url_is_valid:
            st.error(
                f"Backend URL error: {url_message}",
                icon="⚠️",
            )
        else:
            try:
                started = time.perf_counter()

                response = requests.post(
                    f"{api_url}/predict",
                    json=payload,
                    timeout=(5, 30),
                    headers={"Accept": "application/json"},
                )

                latency_ms = (time.perf_counter() - started) * 1000

                if response.ok:
                    try:
                        result = response.json()
                    except ValueError:
                        st.error(
                            "The backend responded, but the response was not valid JSON.",
                            icon="⚠️",
                        )
                    else:
                        if "prediction" not in result:
                            st.error(
                                "The backend response is missing the 'prediction' field.",
                                icon="⚠️",
                            )
                        else:
                            st.session_state.last_result = result
                            st.session_state.last_payload = payload
                            st.session_state.last_latency_ms = latency_ms
                            st.session_state.api_status = {
                                "online": True,
                                "message": f"Online • {latency_ms:.0f} ms",
                            }
                            st.success("Prediction completed successfully.")

                elif response.status_code == 422:
                    st.error(
                        f"Input validation failed: {extract_backend_error(response)}",
                        icon="⚠️",
                    )

                elif response.status_code == 404:
                    st.error(
                        "The /predict endpoint was not found. Check your FastAPI URL.",
                        icon="⚠️",
                    )

                elif response.status_code == 500:
                    st.error(
                        "The FastAPI server returned an internal error. Check your Render logs.",
                        icon="⚠️",
                    )

                else:
                    st.error(
                        f"Prediction failed (HTTP {response.status_code}): "
                        f"{extract_backend_error(response)}",
                        icon="⚠️",
                    )

            except requests.exceptions.RequestException as exc:
                st.error(api_error_message(exc), icon="🔌")

            except Exception:
                st.error(
                    "Something went wrong while contacting the prediction service.",
                    icon="⚠️",
                )

    if st.session_state.last_result is not None:
        result = st.session_state.last_result
        prediction_text = str(result.get("prediction", "Unknown"))
        prediction_value = result.get("data")
        latency = st.session_state.last_latency_ms or 0

        is_positive = (
            prediction_value == 1
            or prediction_text.lower() == "heart disease"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-kicker">Inference result</div>
            <div class="section-title">Latest model output</div>
            """,
            unsafe_allow_html=True,
        )

        result_class = "result-positive" if is_positive else "result-negative"

        st.markdown(
            f"""
            <div class="result-card {result_class}">
                <div class="result-badge">Model prediction</div>
                <div class="result-title">{prediction_text}</div>
                <div class="result-copy">
                    Response received from the FastAPI endpoint in {latency:.0f} ms.
                    This value represents the deployed model's output for the submitted features.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Prediction code</div>
                    <div class="metric-value">{prediction_value}</div>
                    <div class="metric-caption">0 = No Heart Disease · 1 = Heart Disease</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">API latency</div>
                    <div class="metric-value">{latency:.0f} ms</div>
                    <div class="metric-caption">Round-trip inference request</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Model inputs</div>
                    <div class="metric-value">11 / 11</div>
                    <div class="metric-caption">All required features submitted</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View submitted payload"):
            st.json(st.session_state.last_payload)

        with st.expander("View raw API response"):
            st.json(result)


# ============================================================
# FEATURE SCHEMA
# ============================================================

with schema_tab:
    st.markdown(
        """
        <div class="section-kicker">Data contract</div>
        <div class="section-title">Feature schema</div>
        <div class="section-subtitle">
            These are the 11 predictor columns used by the FastAPI request model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    schema_df = pd.DataFrame(
        FEATURES,
        columns=["Feature", "Type", "Accepted / Example"],
    )

    left, right = st.columns([1.2, .8], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.dataframe(
            schema_df,
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-kicker">Pipeline input</div>
            <div class="section-title">Feature groups</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="mini-stat">
                <div class="mini-label">Numeric features</div>
                <div class="mini-value">{len(NUMERIC_FEATURES)}</div>
            </div>
            <br>
            <div class="small-note">{", ".join(NUMERIC_FEATURES)}</div>
            <br><br>
            <div class="mini-stat">
                <div class="mini-label">Categorical features</div>
                <div class="mini-value">{len(CATEGORICAL_FEATURES)}</div>
            </div>
            <br>
            <div class="small-note">{", ".join(CATEGORICAL_FEATURES)}</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "Your FastAPI backend receives these fields with the exact same names. "
            "Keeping the frontend schema aligned with the backend prevents common "
            "422 validation errors.",
            icon="ℹ️",
        )

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footnote">
        CardioML • AI / ML model analytics + live FastAPI inference
        <br>
        For demonstration and portfolio use only; model output is not a medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)
