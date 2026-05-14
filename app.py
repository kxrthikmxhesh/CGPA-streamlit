import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="CGPA Report System", layout="wide")

# ---------------- PREMIUM CSS ---------------- #
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Section Cards */
.section {
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #1e293b, #111827);
    box-shadow: 0px 10px 30px rgba(0,0,0,0.6);
}

/* Titles */
h1, h2, h3 {
    color: #38bdf8;
}

/* Prediction Card */
.prediction {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0ea5e9, #22c55e);
    color: black;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    color: black;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

/* Text blocks */
.text-block {
    font-size: 16px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #
st.title("🎓 CGPA Analytical Report System")
st.markdown("### 📊 Behavioral Analysis & Academic Risk Evaluation")

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("cgpa_model.joblib")
# ---------------- INPUT ---------------- #
st.markdown("## 📥 Input Details")

cc = st.selectbox("Computer Center Usage", [
    "Every week","Monthly","Only during exams",
    "When internet is not available at hostels","Never"
])

lib = st.selectbox("Library Usage", [
    "Every week","Monthly","Only during exams","Never"
])

res = st.selectbox("Primary Study Resource", [
    "Library resources (books, journals)",
    "Faculty office hours/support",
    "Tutoring/Academic support services/YouTube",
    "Seniors"
])

adv = st.slider("Academic Guidance Level",1,5)
study = st.slider("Study Hours (per week)",1,10)

att = st.selectbox("Attendance", [
    "<50%","50-75%","75-90%","90-99%",
    "I never miss classes unless there is an emergency"
])

back = st.selectbox("Backlogs", ["Yes","No"])

# ---------------- ENCODING ---------------- #
cc_list = ["Never","When internet is not available at hostels","Only during exams","Monthly","Every week"]
lib_list = ["Never","Only during exams","Monthly","Every week"]
res_list = ["Library resources (books, journals)","Faculty office hours/support","Tutoring/Academic support services/YouTube","Seniors"]
att_list = ["<50%","50-75%","75-90%","90-99%","I never miss classes unless there is an emergency"]

features = np.array([[
    cc_list.index(cc),
    lib_list.index(lib),
    res_list.index(res),
    adv,
    study,
    att_list.index(att),
    1 if back=="Yes" else 0
]])

# ---------------- ANALYSIS ---------------- #
if st.button("Generate Full Report"):

    pred = model.predict(features)[0]
    labels = ["<6.5","6.5-8.5",">8.5"]

    # ---------------- PREDICTION ---------------- #
    st.markdown(f"<div class='prediction'>Predicted CGPA Category: {labels[pred]}</div>", unsafe_allow_html=True)

    att_index = att_list.index(att)
    lib_index = lib_list.index(lib)

    # ---------------- DETAILED ANALYSIS ---------------- #
    st.markdown("## 📊 Detailed Behavioral Analysis")

    study_text = f"""
    Your reported study duration is {study} hours per week. This level of engagement plays a crucial role in determining conceptual clarity, retention, and exam readiness. 
    Students with higher academic performance typically demonstrate consistent and structured study patterns. 
    In your case, the current study duration indicates {'strong academic discipline and consistency' if study >=7 else 'moderate effort but inconsistent depth' if study>=4 else 'insufficient academic engagement which significantly impacts performance'}.
    """

    attendance_text = f"""
    Your attendance level falls under the category: {att}. Attendance directly reflects your engagement with academic content and faculty interaction. 
    Regular attendance ensures continuous exposure to concepts and reduces dependency on self-learning under pressure. 
    Your current attendance suggests {'excellent academic involvement' if att_index>=3 else 'partial engagement with potential gaps' if att_index>=1 else 'critical lack of academic exposure leading to weak continuity'}.
    """

    backlog_text = f"""
    Backlog status: {back}. The presence of backlogs indicates unresolved academic gaps and directly contributes to increased academic pressure in future semesters. 
    {'This reflects a stable academic record with no pending issues.' if back=='No' else 'This is a critical risk factor and requires immediate attention to prevent long-term academic decline.'}
    """

    resource_text = f"""
    Your primary study resource is: {res}. Resource selection significantly affects learning quality. Structured resources like textbooks and guided faculty support generally result in deeper understanding. 
    Your current choice suggests {'effective structured learning' if lib_index>=2 else 'partial reliance on informal or inconsistent sources which may limit depth of understanding'}.
    """

    st.markdown(f"<div class='section text-block'>{study_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section text-block'>{attendance_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section text-block'>{backlog_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section text-block'>{resource_text}</div>", unsafe_allow_html=True)

    # ---------------- IMPROVEMENT PLAN ---------------- #
    st.markdown("## 📘 Detailed Improvement Strategy")

    plan = f"""
    Based on the analysis of your academic behavior, several improvement areas have been identified.

    Study habits need {'significant improvement' if study<4 else 'better structuring and consistency' if study<7 else 'to be maintained at current level'}. 
    Increasing focused study hours and incorporating active revision techniques will improve retention and performance.

    Attendance should {'be prioritized immediately' if att_index<2 else 'be improved for consistency' if att_index<3 else 'be maintained at current level'} as it directly impacts conceptual clarity.

    {'Clearing backlogs should be treated as the highest priority as it affects overall academic progression.' if back=='Yes' else 'Maintaining a backlog-free record is a strong positive indicator of academic stability.'}

    Resource utilization should {'shift towards structured materials like textbooks and guided learning' if lib_index<2 else 'continue with current effective approach'} to enhance understanding.

    Overall, consistent incremental improvements in these areas can significantly elevate academic performance over time.
    """

    st.markdown(f"<div class='section text-block'>{plan}</div>", unsafe_allow_html=True)
