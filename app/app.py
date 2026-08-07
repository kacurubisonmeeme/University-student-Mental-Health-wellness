"""
Student Mental Health Early-Warning Tool — Streamlit App

IMPORTANT FRAMING: this is a prototype of an institutional screening aid
for a university counseling center (per the project's original design),
NOT a self-diagnosis tool. It always displays crisis resources, and does
not present its output as a clinical or individual diagnosis.

Run locally:  streamlit run app.py
Deploy free:  push this file + model.pkl + encoders.pkl to a public
              GitHub repo, then deploy at streamlit.io/cloud pointing at app.py
"""

import pickle
import json
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Mental Health Early-Warning Tool",
                    page_icon="🎓", layout="centered")

# Resolve file paths relative to this script's own folder, not the process's
# working directory — Streamlit Cloud runs apps with cwd set to the repo
# root, so a bare "model.pkl" fails if app.py lives in a subfolder.
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(filename):
    return os.path.join(APP_DIR, filename)

@st.cache_resource
def load_artifacts():
    with open(_path("model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(_path("encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(_path("metrics.json"), "r") as f:
        metrics = json.load(f)
    return model, encoders, metrics

model, encoders, metrics = load_artifacts()

st.title("🎓 Student Mental Health Early-Warning Tool")

st.info(
    "**This is a prototype for institutional use, not a diagnostic tool.** "
    "It's designed to help a university counseling center prioritize proactive "
    "outreach, the same way the original project frames it — a decision-support "
    "aid for staff, not a self-assessment. It does not replace a conversation "
    "with a mental health professional.",
    icon="ℹ️"
)

with st.expander("🆘 If you or someone you know is struggling right now"):
    st.markdown(
        "- **Kenya:** Befrienders Kenya — 0722 178 177 / 0736 356 359\n"
        "- **US:** 988 Suicide & Crisis Lifeline — call or text 988\n"
        "- **International:** [findahelpline.com](https://findahelpline.com) "
        "lists crisis lines by country\n\n"
        "You don't need a 'high risk' result from a model to reach out — "
        "if something feels heavy, that's reason enough."
    )

st.divider()
st.caption(
    f"Model: Decision Tree (chosen for interpretability, per the model showdown "
    f"below), trained on {metrics['n_records_used']:,} student records."
)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", sorted(encoders["Gender"].classes_))
    age = st.number_input("Age", min_value=16, max_value=60, value=21)
    academic_pressure = st.slider("Academic Pressure (0 = none, 5 = extreme)", 0.0, 5.0, 2.5, 0.5)
    cgpa = st.number_input("CGPA (0-10 scale)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    study_satisfaction = st.slider("Study Satisfaction (0-5)", 0.0, 5.0, 2.5, 0.5)
    financial_stress = st.slider("Financial Stress (0-5)", 0.0, 5.0, 2.5, 0.5)

with col2:
    sleep = st.selectbox("Sleep Duration", sorted(encoders["Sleep Duration"].classes_))
    diet = st.selectbox("Dietary Habits", sorted(encoders["Dietary Habits"].classes_))
    suicidal_thoughts = st.selectbox(
        "Has the student ever reported suicidal thoughts?",
        sorted(encoders["Have you ever had suicidal thoughts ?"].classes_))
    family_history = st.selectbox("Family History of Mental Illness",
                                   sorted(encoders["Family History of Mental Illness"].classes_))
    work_study_hours = st.slider("Work/Study Hours per day", 0.0, 16.0, 6.0, 0.5)

if st.button("Run Screening", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "Gender": encoders["Gender"].transform([gender])[0],
        "Age": age,
        "Academic Pressure": academic_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Sleep Duration": encoders["Sleep Duration"].transform([sleep])[0],
        "Dietary Habits": encoders["Dietary Habits"].transform([diet])[0],
        "Have you ever had suicidal thoughts ?": encoders["Have you ever had suicidal thoughts ?"].transform([suicidal_thoughts])[0],
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": encoders["Family History of Mental Illness"].transform([family_history])[0],
    }])

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    flagged = pred == 1

    st.divider()

    if flagged:
        st.warning(
            f"**Flagged for proactive outreach** (model confidence: {proba[1]:.0%}). "
            "Per the project's recommendation, this would trigger a routine, "
            "no-pressure check-in from student services — not an alarm."
        )
    else:
        st.success(f"**Not flagged** (model confidence: {proba[0]:.0%}).")

    st.caption(
        "This reflects statistical patterns in historical survey data, not a "
        "clinical assessment of this individual. Always pair with human judgment."
    )

    if suicidal_thoughts == "Yes":
        st.error(
            "⚠️ This profile indicates a reported history of suicidal thoughts. "
            "This should always prompt direct, immediate human follow-up regardless "
            "of the model's overall flag — see crisis resources above.",
            icon="🆘"
        )

with st.expander("Model showdown & performance details"):
    st.write("Three models were compared, prioritizing interpretability for counselor trust:")
    showdown_df = pd.DataFrame(
        list(metrics["model_showdown_accuracy_pct"].items()),
        columns=["Model", "Accuracy (%)"]
    )
    st.dataframe(showdown_df, hide_index=True, use_container_width=True)
    st.write("**Deployed model:** Decision Tree — chosen over the slightly more "
             "accurate Random Forest because its decision path is fully transparent, "
             "which matters when a human is acting on the flag.")
    st.write("**Top predictors:**", ", ".join(metrics["top_predictors"]))
    st.image(_path("feature_importance.png"))
    st.image(_path("decision_tree.png"), caption="Decision path (top 3 levels)")

st.divider()
st.caption(
    "Data: Student Depression Dataset · "
    "[Project repo](https://github.com/kacurubisonmeeme/University-student-Mental-Health-wellness)"
)
