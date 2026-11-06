"""Student Performance Prediction — portfolio-friendly web interface.

The original Tkinter application remains untouched as interface.py.
Predictions are generated with the repository's trained SVR model.
Preprocessing is reconstructed from the training notebook and spreadsheet
instead of incorrectly fitting a StandardScaler on a single prediction row.
"""
from __future__ import annotations

import io
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_svr_model.joblib"
DATA_PATH = ROOT / "StudentPerformanceFactors.xlsx"

st.set_page_config(
    page_title="Student Performance Prediction | AI Predictions",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)
theme = ROOT / "student_theme.css"
if theme.exists():
    st.markdown(f"<style>{theme.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

NUMERIC = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]
OPTIONS = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["Yes", "No"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Internet_Access": ["Yes", "No"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "School_Type": ["Public", "Private"],
    "Peer_Influence": ["Positive", "Neutral", "Negative"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
}
FEATURES = NUMERIC + [f"{name}_encoded" for name in OPTIONS]
DEFAULTS = {
    "Hours_Studied": 22,
    "Attendance": 85,
    "Sleep_Hours": 7,
    "Previous_Scores": 75,
    "Tutoring_Sessions": 2,
    "Physical_Activity": 3,
    "Parental_Involvement": "Medium",
    "Access_to_Resources": "Medium",
    "Extracurricular_Activities": "Yes",
    "Motivation_Level": "Medium",
    "Internet_Access": "Yes",
    "Teacher_Quality": "Medium",
    "School_Type": "Public",
    "Peer_Influence": "Neutral",
    "Parental_Education_Level": "College",
    "Distance_from_Home": "Moderate",
}
RANGES = {
    "Hours_Studied": (0, 44),
    "Attendance": (0, 100),
    "Sleep_Hours": (0, 12),
    "Previous_Scores": (0, 100),
    "Tutoring_Sessions": (0, 10),
    "Physical_Activity": (0, 10),
}


@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load the saved model and training-consistent preprocessing statistics."""
    model = joblib.load(MODEL_PATH)
    frame = pd.read_excel(DATA_PATH, engine="openpyxl")
    for name in NUMERIC:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frame[name] = frame[name].fillna(frame[name].mean())
    for name in OPTIONS:
        frame[name] = frame[name].fillna(frame[name].mode(dropna=True).iloc[0]).astype(str)

    # Replicate the preprocessing cells in predicting-student-score.ipynb:
    # StandardScaler on 3 features; MinMaxScaler on 2 features;
    # Tutoring_Sessions left as-is; LabelEncoder per categorical column.
    standard_cols = ["Hours_Studied", "Sleep_Hours", "Physical_Activity"]
    minmax_cols = ["Attendance", "Previous_Scores"]
    standard = StandardScaler().fit(frame[standard_cols])
    minmax = MinMaxScaler().fit(frame[minmax_cols])
    encoders = {name: LabelEncoder().fit(frame[name]) for name in OPTIONS}
    expected = list(getattr(model, "feature_names_in_", FEATURES))
    if len(expected) != len(FEATURES) or set(expected) != set(FEATURES):
        raise ValueError(
            "The saved model expects a different feature schema. "
            "Check the original training notebook and model artifact."
        )
    return model, standard, minmax, encoders, expected


def predict_many(rows: pd.DataFrame, predictor) -> np.ndarray:
    model, standard, minmax, encoders, expected = predictor
    raw = rows.copy()
    missing = [column for column in NUMERIC + list(OPTIONS) if column not in raw.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(missing))
    processed = pd.DataFrame(index=raw.index)
    for name in NUMERIC:
        vals = pd.to_numeric(raw[name], errors="coerce")
        if vals.isna().any():
            raise ValueError(f"{name} contains missing or non-numeric values.")
        low, high = RANGES[name]
        if not vals.between(low, high).all():
            raise ValueError(f"{name} must be between {low} and {high}.")
        processed[name] = vals.astype(float)
    processed[["Hours_Studied", "Sleep_Hours", "Physical_Activity"]] = standard.transform(
        processed[["Hours_Studied", "Sleep_Hours", "Physical_Activity"]]
    )
    processed[["Attendance", "Previous_Scores"]] = minmax.transform(
        processed[["Attendance", "Previous_Scores"]]
    )
    for name, encoder in encoders.items():
        labels = raw[name].astype(str).str.strip()
        invalid = sorted(set(labels).difference(encoder.classes_))
        if invalid:
            raise ValueError(f"Unexpected value in {name}: {', '.join(invalid)}. "
                             f"Allowed: {', '.join(encoder.classes_)}")
        processed[f"{name}_encoded"] = encoder.transform(labels)
    return np.asarray(model.predict(processed[expected]), dtype=float)


def display_score(score: float) -> None:
    score_percent = min(max(float(score), 0), 100)
    st.markdown(
        '<div class="score-kicker">MODEL OUTPUT</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.05, 1.3], gap="large")
    with left:
        st.markdown(
            f'<div class="score-big">{score:.1f}<small> / 100</small></div>'
            '<div class="score-under">Predicted exam score</div>',
            unsafe_allow_html=True,
        )
        st.progress(score_percent / 100)
        st.caption("Exploratory estimate from the trained SVR model, not an observed result.")
    with right:
        st.markdown("**How to read this prediction**")
        st.write("This value is a model estimate based on the selected student profile. "
                 "It is not a grade guarantee, a diagnosis, or a measure of a student's potential.")
        st.caption("The dataset and saved model were created for an educational machine-learning project.")


st.markdown(
    """
    <section class="hero">
      <div class="hero-text">
        <span class="eyebrow"><span class="eyebrow-dot"></span> MACHINE LEARNING · EDUCATION ANALYTICS</span>
        <h1>Student Performance <span>Studio</span></h1>
        <p>Explore the factors behind predicted exam outcomes in an interactive,
           model-powered learning analytics workspace.</p>
        <div class="hero-chips"><span>SVR regression</span><span>16 inputs</span><span>Interactive what-if analysis</span></div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    predictor = load_predictor()
except Exception as exc:
    st.error("The model or its preprocessing reference could not be loaded.")
    st.exception(exc)
    st.stop()

if "profile" not in st.session_state:
    st.session_state.profile = DEFAULTS.copy()
if "prediction" not in st.session_state:
    try:
        st.session_state.prediction = float(
            predict_many(pd.DataFrame([st.session_state.profile]), predictor)[0]
        )
    except Exception:
        st.session_state.prediction = None

tabs = st.tabs(["◈  Prediction Dashboard", "▦  Scenario Lab", "▤  Batch Predictions", "ⓘ  Model & Method"])

with tabs[0]:
    st.markdown('<div class="section-row"><div><span class="section-kicker">YOUR WORKSPACE</span>'
                '<h3>Student profile & predicted outcome</h3></div>'
                '<span class="section-aside">01 / PERSONALIZED FORECAST</span></div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Model", "SVR", help="Support Vector Regression")
    metric_cols[1].metric("Student factors", "16")
    metric_cols[2].metric("Numerical inputs", "6")
    metric_cols[3].metric("Contextual inputs", "10")

    with st.container(border=True):
        st.markdown("#### Configure a student profile")
        with st.form("prediction_form", border=False):
            a, b, c = st.columns(3, gap="large")
            p = st.session_state.profile
            with a:
                st.markdown("**Study & participation**")
                hours = st.slider("Study hours per week", 0, 44, int(p["Hours_Studied"]))
                attendance = st.slider("Attendance (%)", 0, 100, int(p["Attendance"]))
                sleep = st.slider("Sleep hours per night", 0, 12, int(p["Sleep_Hours"]))
                extracurricular = st.selectbox("Extracurricular activities",
                                               OPTIONS["Extracurricular_Activities"],
                                               index=OPTIONS["Extracurricular_Activities"].index(p["Extracurricular_Activities"]))
                physical = st.slider("Physical activity (hours/week)", 0, 10, int(p["Physical_Activity"]))
            with b:
                st.markdown("**Academic background**")
                previous = st.slider("Previous exam score", 0, 100, int(p["Previous_Scores"]))
                tutoring = st.slider("Tutoring sessions per month", 0, 10, int(p["Tutoring_Sessions"]))
                motivation = st.selectbox("Motivation", OPTIONS["Motivation_Level"],
                                          index=OPTIONS["Motivation_Level"].index(p["Motivation_Level"]))
                teacher = st.selectbox("Teacher quality", OPTIONS["Teacher_Quality"],
                                       index=OPTIONS["Teacher_Quality"].index(p["Teacher_Quality"]))
                peers = st.selectbox("Peer influence", OPTIONS["Peer_Influence"],
                                     index=OPTIONS["Peer_Influence"].index(p["Peer_Influence"]))
            with c:
                st.markdown("**Learning environment**")
                resources = st.selectbox("Access to resources", OPTIONS["Access_to_Resources"],
                                         index=OPTIONS["Access_to_Resources"].index(p["Access_to_Resources"]))
                parental = st.selectbox("Parental involvement", OPTIONS["Parental_Involvement"],
                                        index=OPTIONS["Parental_Involvement"].index(p["Parental_Involvement"]))
                internet = st.selectbox("Internet access", OPTIONS["Internet_Access"],
                                        index=OPTIONS["Internet_Access"].index(p["Internet_Access"]))
                school = st.selectbox("School type", OPTIONS["School_Type"],
                                      index=OPTIONS["School_Type"].index(p["School_Type"]))
                education = st.selectbox("Parental education", OPTIONS["Parental_Education_Level"],
                                         index=OPTIONS["Parental_Education_Level"].index(p["Parental_Education_Level"]))
                distance = st.selectbox("Distance from home", OPTIONS["Distance_from_Home"],
                                        index=OPTIONS["Distance_from_Home"].index(p["Distance_from_Home"]))
            submit = st.form_submit_button("✦  Generate prediction", type="primary", use_container_width=True)
            if submit:
                updated = {
                    "Hours_Studied": hours, "Attendance": attendance,
                    "Sleep_Hours": sleep, "Previous_Scores": previous,
                    "Tutoring_Sessions": tutoring, "Physical_Activity": physical,
                    "Parental_Involvement": parental, "Access_to_Resources": resources,
                    "Extracurricular_Activities": extracurricular,
                    "Motivation_Level": motivation, "Internet_Access": internet,
                    "Teacher_Quality": teacher, "School_Type": school,
                    "Peer_Influence": peers, "Parental_Education_Level": education,
                    "Distance_from_Home": distance,
                }
                try:
                    score = float(predict_many(pd.DataFrame([updated]), predictor)[0])
                    st.session_state.profile = updated
                    st.session_state.prediction = score
                    st.success("Prediction updated using the saved model.")
                except Exception as exc:
                    st.error(f"Could not generate prediction: {exc}")
    with st.container(border=True):
        if st.session_state.prediction is None:
            st.info("Generate a prediction to view the estimated exam score.")
        else:
            display_score(st.session_state.prediction)

with tabs[1]:
    st.markdown('<div class="section-row"><div><span class="section-kicker">EXPERIMENT</span>'
                '<h3>What-if scenario explorer</h3></div>'
                '<span class="section-aside">02 / MODEL SENSITIVITY</span></div>', unsafe_allow_html=True)
    st.caption("Change one input while keeping the other profile fields fixed. "
               "The plotted differences are model responses, not causal effects.")
    current = st.session_state.profile.copy()
    ca, cb = st.columns(2, gap="medium")
    with ca:
        target_hours = st.slider("Scenario · weekly study hours", 0, 44, int(current["Hours_Studied"]), key="whatif_hours")
    with cb:
        target_attendance = st.slider("Scenario · attendance (%)", 0, 100, int(current["Attendance"]), key="whatif_attendance")
    scenario = current.copy()
    scenario["Hours_Studied"] = target_hours
    scenario["Attendance"] = target_attendance
    try:
        baseline_score, scenario_score = predict_many(pd.DataFrame([current, scenario]), predictor)
        c1, c2, c3 = st.columns(3)
        c1.metric("Current profile", f"{baseline_score:.1f} / 100")
        c2.metric("Scenario estimate", f"{scenario_score:.1f} / 100")
        c3.metric("Model difference", f"{scenario_score - baseline_score:+.2f} pts")
        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True):
                st.markdown("**Response to study hours**")
                sampled_hours = sorted(set([0, 5, 10, 15, 20, 25, 30, 35, 40, 44, int(current["Hours_Studied"])]))
                samples = [{**current, "Hours_Studied": h} for h in sampled_hours]
                vals = predict_many(pd.DataFrame(samples), predictor)
                chart = pd.DataFrame({"Hours/week": sampled_hours, "Predicted score": vals})
                fig = alt.Chart(chart).mark_line(point=True, strokeWidth=3, color="#6560D9").encode(
                    x=alt.X("Hours/week:Q", scale=alt.Scale(domain=[0, 44])),
                    y=alt.Y("Predicted score:Q", scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("Hours/week:Q"), alt.Tooltip("Predicted score:Q", format=".2f")]
                ).properties(height=245)
                st.altair_chart(fig, use_container_width=True)
        with right:
            with st.container(border=True):
                st.markdown("**Response to attendance**")
                sampled_attendance = sorted(set([0, 20, 40, 60, 70, 80, 90, 100, int(current["Attendance"])]))
                samples = [{**current, "Attendance": x} for x in sampled_attendance]
                vals = predict_many(pd.DataFrame(samples), predictor)
                chart = pd.DataFrame({"Attendance (%)": sampled_attendance, "Predicted score": vals})
                fig = alt.Chart(chart).mark_area(
                    line={"color": "#20A8A0", "strokeWidth": 3},
                    color="#A7E8DE", opacity=0.8
                ).encode(
                    x=alt.X("Attendance (%):Q", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Predicted score:Q", scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("Attendance (%):Q"), alt.Tooltip("Predicted score:Q", format=".2f")]
                ).properties(height=245)
                st.altair_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Scenario could not be calculated: {exc}")

with tabs[2]:
    st.markdown('<div class="section-row"><div><span class="section-kicker">SCALE</span>'
                '<h3>Predict scores for multiple students</h3></div>'
                '<span class="section-aside">03 / CSV OR EXCEL</span></div>', unsafe_allow_html=True)
    st.write("Import a file containing the six numerical fields and ten categorical fields. "
             "The app calculates predictions in memory and offers a downloadable CSV.")
    example = pd.DataFrame([DEFAULTS, {
        **DEFAULTS, "Hours_Studied": 27, "Attendance": 94,
        "Motivation_Level": "High", "Previous_Scores": 83,
    }])
    st.download_button("↓ Download example CSV", example.to_csv(index=False).encode("utf-8"),
                       file_name="student_profiles_example.csv", mime="text/csv")
    upload = st.file_uploader("Upload student profiles (max 1,000 rows)",
                              type=["csv", "xlsx"], key="students_upload")
    if upload is not None:
        try:
            uploaded = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload, engine="openpyxl")
            if uploaded.empty:
                st.warning("The uploaded file contains no rows.")
            elif len(uploaded) > 1000:
                st.error("Upload a maximum of 1,000 rows per batch.")
            else:
                # Accept column names from the original desktop UI, as well as clean labels.
                for name in OPTIONS:
                    if name not in uploaded.columns and f"{name}_encoded" in uploaded.columns:
                        uploaded[name] = uploaded[f"{name}_encoded"]
                results = predict_many(uploaded, predictor)
                report = uploaded.copy()
                report["Predicted_Exam_Score"] = np.round(results, 2)
                a, b, c = st.columns(3)
                a.metric("Profiles processed", len(report))
                b.metric("Mean predicted score", f"{float(np.mean(results)):.1f}")
                c.metric("Model", "SVR")
                st.dataframe(report[["Predicted_Exam_Score"] + NUMERIC].head(50), use_container_width=True, hide_index=True)
                st.download_button(
                    "↓ Export predictions (CSV)",
                    report.to_csv(index=False).encode("utf-8"),
                    file_name="student_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as exc:
            st.error(f"Please check the uploaded dataset: {exc}")
    st.caption("Avoid uploading identifiable student records. This is a demonstration tool, not a school decision system.")

with tabs[3]:
    st.markdown('<div class="section-row"><div><span class="section-kicker">TRANSPARENCY</span>'
                '<h3>How the model works</h3></div>'
                '<span class="section-aside">04 / METHODOLOGY</span></div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        with st.container(border=True):
            st.markdown("**01 · Machine learning**")
            st.write("A saved Support Vector Regression (SVR) model predicts an exam score from a 16-feature profile.")
    with b:
        with st.container(border=True):
            st.markdown("**02 · Data preparation**")
            st.write("Original training-data statistics are used for standard/min-max scaling and categorical encoding.")
    with c:
        with st.container(border=True):
            st.markdown("**03 · Responsible use**")
            st.write("Predictions are experimental, not certified or calibrated for real-world educational decisions.")
    st.markdown("#### Inputs & assumptions")
    st.write("The predictor uses study habits, attendance, academic history, classroom support, "
             "and learning-environment factors. Scenarios illustrate model sensitivity, not the impact "
             "of an intervention. Evaluation metrics are intentionally not claimed without a reproducible test.")
    st.write("The original desktop Tkinter application and research notebooks remain available in the repository.")
    st.link_button("↗ View GitHub source", "https://github.com/ayoubennadifb-spec/Student_performance_prediction")

st.markdown('<div class="footer">STUDENT PERFORMANCE STUDIO <span>Educational ML portfolio demo · no personal student records stored</span></div>',
            unsafe_allow_html=True)
