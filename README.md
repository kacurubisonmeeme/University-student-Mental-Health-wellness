University Student Mental Health Wellness
Overview
This project aims to assist the University Counseling Center in proactively managing student mental health. By analyzing student data including academic performance, sleep habits, and self reported pressure levels we developed a predictive classification model to identify students at risk of depression. This "early warning system" allows counselors to intervene before a crisis occurs, shifting from a reactive to a proactive support model.

Business Understanding
Stakeholder: University Counseling Center.

Problem: Counseling services are currently reactive, relying on students to seek help on their own. This often means interventions happen only after a student’s mental health has significantly deteriorated.

Goal: Build a Machine Learning model that uses existing student data to flag those at high risk of depression, enabling targeted, early outreach.

The Dataset: We utilized a dataset of approximately 28,000 student records, containing features such as CGPA, sleep duration, dietary habits, and self reported academic/work pressure.

Modeling Approach
We adopted an iterative modeling process to balance predictive power with explainability:

Baseline Model (Logistic Regression): Established a strong baseline with ~84.6% accuracy, proving that linear relationships exist between academic stress and depression.

Complex Model (Random Forest): Deployed an ensemble method to capture non-linear patterns. While it offered high accuracy (~84.2%), it acts as a "Black Box," offering little insight into why a specific student was flagged.

The "Glass Box" (Decision Tree): Our final model of choice. It achieved comparable accuracy (~82.4%) while providing full transparency. Counselors can view the exact decision path (e.g., "Student has Suicidal Thoughts AND High Academic Pressure") that led to the risk flag.

Note: All models were tuned using GridSearchCV to ensure optimal performance without overfitting.

Evaluation
The models were evaluated using Accuracy and ROC-AUC scores to ensure they could distinguish between depressed and healthy students effectively.

Key Findings:

Academic Stress is the #1 Predictor: Feature Importance analysis revealed that CGPA and Academic Pressure are the strongest indicators of depression risk.

Transparency Wins: While the Logistic Regression and Random Forest models were slightly more accurate numerically, the Decision Tree was selected for deployment because its interpretability builds trust with clinical staff.

Conclusion & Recommendations
This project demonstrates that student data can reliably predict depression risk without requiring complex, opaque AI systems. By deploying the Decision Tree model, the University Counseling Center gains a transparent diagnostic tool that aligns with their clinical workflow.

Actionable Recommendations:

Gradebook Integration: Since CGPA is a top predictor, integrate the model with the registrar's system to trigger automated check-ins when a student's rolling GPA drops below a risk threshold.

Academic Relief Valves: Acknowledge academic pressure as a health risk. Implement specific stress-management interventions (e.g., "Life Happens" assignment extensions) during high-pressure periods like midterms and finals.

Data Enhancement: Improve the collection of financial stress data in intake forms to close gaps identified during modeling.

Repository Structure
├── data/                # Dataset used for analysis
├── images/              # Visualizations for README and presentation
├── student_depression_prediction.ipynb  # Main Jupyter Notebook
├── presentation.pdf     # Non-technical presentation slides
└── README.md            # Project overview (this file)
