# 🎓 University Student Mental Health Wellness

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

**A machine learning initiative to proactively identify students at risk of depression using academic and lifestyle data.**

---

## 📖 Overview
The University Counseling Center is currently overwhelmed by a reactive support model—students are often only identified *after* a crisis occurs. This project utilizes a dataset of **28,000+ student records** to build a predictive "Early Warning System." By analyzing factors like GPA, sleep duration, and academic pressure, we can flag high-risk students early and intervene before their mental health deteriorates.

---

## 💼 Business Understanding

| **Feature** | **Description** |
| :--- | :--- |
| **Stakeholder** | **University Counseling Center** |
| **The Problem** | Reactive care leads to missed opportunities for early intervention. |
| **The Goal** | Deploy a classification model to identify at-risk students proactively. |
| **The Data** | Academic records (CGPA), lifestyle (Sleep, Diet), and self-reported Pressure. |

---

## 🤖 Modeling Approach & Results

We approached this as a tournament between three model types to balance **Accuracy** against **Interpretability**. All models were tuned using `GridSearchCV`.

### The Model Showdown

| Model Type | Algorithm | Accuracy | The Verdict |
| :--- | :--- | :--- | :--- |
| **Baseline** | Logistic Regression | **84.6%** | Strong start, proving linear relationships exist. |
| **The "Black Box"** | Random Forest | **84.2%** | High accuracy, but difficult to explain "why" a student is flagged. |
| **The "Glass Box"** 🏆 | **Decision Tree** | **82.4%** | **WINNER.** Slightly lower accuracy, but offers 100% transparency on decision logic. |

> **Why the Decision Tree?**
> In mental health, trust is paramount. While the Random Forest was slightly more robust, the **Decision Tree** allows a counselor to see the exact path (e.g., *"Suicidal Thoughts = Yes" → "Academic Pressure = High"*) that led to the risk flag.

---

## 📊 Evaluation & Visuals

### Key Risk Drivers
Our analysis identified that academic factors are the loudest signals for depression risk.

* **#1 Predictor:** `CGPA` (Grades)
* **#2 Predictor:** `Academic Pressure`

### Decision Logic
Below is a visualization of the Decision Tree model (The "Glass Box"), showing the transparent logic flow used to assess student risk.

![Decision Tree Visualization](images/output%203.png)
*(Note: Ensure your image file is named correctly in the 'images' folder)*

---

## 💡 Recommendations

Based on our findings, we propose three strategic actions for the university:

1.  **🎓 "Early Warning" Gradebook Integration**
    * **Insight:** CGPA is the strongest predictor.
    * **Action:** Integrate the model with the registrar. Automatically trigger a "We Care" check-in email from student services when a student's rolling GPA drops below the identified risk threshold.

2.  **🛑 Academic Relief Valves**
    * **Insight:** "High" Academic Pressure is a critical risk factor.
    * **Action:** Treat exam periods as mental health events. Deploy "Life Happens" tokens (no-questions-asked extensions) and stress-management workshops during midterms and finals.

3.  **📝 Data Enhancement**
    * **Insight:** Financial stress is a known risk but had data gaps in our set.
    * **Action:** Standardize the collection of financial aid and stress data during student intake to improve future model precision.

---

## 📂 Repository Structure

```text
├── data/                               # Dataset files
├── images/                             # Generated plots and decision tree visuals
├── student_depression_prediction.ipynb # Main analysis and modeling notebook
├── presentation.pdf                    # Non-technical slide deck
└── README.md                           # Project documentation
