"""
HeartCare — CVD Risk Prediction App
Run: streamlit run app.py
Requires: model_xgb_only.pkl, feature_names.pkl, model_metadata.json

Changelog:
- [FIX] Model now uses SMOTE + scale_pos_weight=1 (no double correction).
        All metrics and threshold are read dynamically from model_metadata.json
        so no hardcoded numbers anywhere in this file.
- [FIX] SHAP table now reads from metadata (populated by NB3B) instead of
        hardcoded values that became stale after model was retrained.
- [FIX] sub-header updated to reflect Config B imbalance strategy.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="HeartCare — CVD Risk Predictor",
    page_icon="🫀", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size:2.2rem; font-weight:800; color:#C0392B; text-align:center; margin-bottom:0.2rem; }
    .sub-header  { font-size:1rem; color:#7F8C8D; text-align:center; margin-bottom:2rem; }
    .risk-high   { background:linear-gradient(135deg,#FADBD8,#F1948A); border-left:6px solid #C0392B;
                   padding:1.5rem; border-radius:10px; text-align:center; }
    .risk-low    { background:linear-gradient(135deg,#D5F5E3,#82E0AA); border-left:6px solid #27AE60;
                   padding:1.5rem; border-radius:10px; text-align:center; }
    .section-title { font-size:1.1rem; font-weight:700; color:#2C3E50; margin-top:1rem;
                     border-bottom:2px solid #E8634C; padding-bottom:0.3rem; }
    .stButton>button { width:100%; background:#C0392B; color:white; font-size:1.1rem;
                       font-weight:700; border:none; padding:0.8rem; border-radius:8px; }
    .stButton>button:hover { background:#E74C3C; }
    .footer { text-align:center; color:#BDC3C7; font-size:0.8rem; margin-top:3rem; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    mdl  = joblib.load("model_xgb_only.pkl")
    feat = joblib.load("feature_names.pkl")
    with open("model_metadata.json") as f:
        meta = json.load(f)
    return mdl, feat, meta

try:
    model, FEAT_NAMES, META = load_model()
    THRESHOLD   = META["threshold"]
    CONT_COLS   = META.get("scaler", {}).get("cont_cols",
                    ["Sleep_hours", "PhysHealth_days", "MentHealth_days"])
    BIN_COLS    = [c for c in FEAT_NAMES if c not in CONT_COLS]
    FINAL_ORDER = CONT_COLS + BIN_COLS   # ColumnTransformer output order
    # SHAP importance — read from metadata if available, else fallback defaults
    SHAP_TOP = META.get("shap_top10", [
        {"rank": 1, "feature": "Age_65_plus",       "shap": None},
        {"rank": 2, "feature": "Male",               "shap": None},
        {"rank": 3, "feature": "Sleep_hours",        "shap": None},
        {"rank": 4, "feature": "PhysHealth_days",    "shap": None},
        {"rank": 5, "feature": "GenHealth_excellent","shap": None},
    ])
except Exception as e:
    st.error(f"⚠️ Could not load model files: {e}")
    st.info("Make sure model_xgb_only.pkl, feature_names.pkl, model_metadata.json are in the same folder as app.py")
    st.stop()


st.markdown('<div class="main-header">🫀 HeartCare CVD Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">BRFSS 2022 · XGBoost + SMOTE (scale_pos_weight=1) · '
    'Powered by Big Data Pipeline</div>',
    unsafe_allow_html=True
)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Patient Information")
    st.markdown("Fill in all sections, then click **Predict**.")
    st.divider()

    st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
    sex        = st.radio("Sex", ["Female","Male"], horizontal=True)
    age_group  = st.selectbox("Age Group", ["18–24","25–34","35–44","45–54","55–64","65+"])
    race       = st.selectbox("Race / Ethnicity",
                    ["White","Black","American Indian","Asian","Pacific Islander","Hispanic"])
    education  = st.selectbox("Education Level",
                    ["No High School","High School Graduate","Some College","College Graduate"])
    employment = st.selectbox("Employment Status",
                    ["Employed","Self-employed","Unemployed >1yr","Unemployed <1yr",
                     "Homemaker","Student","Retired","Unable to work"])

    st.divider()
    st.markdown('<div class="section-title">🏃 Lifestyle</div>', unsafe_allow_html=True)
    sleep    = st.slider("Sleep Hours per Night", 1, 24, 7)
    exercise = st.radio("Regular Physical Activity?", ["Yes","No"], horizontal=True)
    alcohol  = st.radio("Alcohol Drinker?", ["Yes","No"], horizontal=True)
    smoking  = st.selectbox("Smoking Status",
                    ["Never smoked","Former smoker","Smoke some days","Smoke every day"])
    bmi_cat  = st.radio("BMI Category", ["Normal weight","Overweight / Obese"], horizontal=True)

    st.divider()
    st.markdown('<div class="section-title">🏥 Medical History</div>', unsafe_allow_html=True)
    stroke     = st.radio("Stroke history?",    ["No","Yes"], horizontal=True)
    diabetes   = st.selectbox("Diabetes status",
                    ["No","Pre-diabetes","Gestational (past)","Yes"])
    kidney     = st.radio("Kidney disease?",    ["No","Yes"], horizontal=True)
    copd       = st.radio("COPD / Emphysema?",  ["No","Yes"], horizontal=True)
    arthritis  = st.radio("Arthritis?",         ["No","Yes"], horizontal=True)
    depression = st.radio("Depression?",        ["No","Yes"], horizontal=True)
    cancer     = st.radio("Cancer history?",    ["No","Yes"], horizontal=True)

    st.divider()
    st.markdown('<div class="section-title">💊 General Health</div>', unsafe_allow_html=True)
    gen_health = st.select_slider("General Health",
        ["Poor","Fair","Good","Very Good","Excellent"], value="Good")
    phys_days  = st.slider("Poor Physical Health Days (last 30 days)", 0, 30, 0)
    ment_days  = st.slider("Poor Mental Health Days (last 30 days)", 0, 30, 0)

    st.divider()
    predict_btn = st.button("🔮 Predict CVD Risk", type="primary")


# ── Feature builder + predictor ───────────────────────────────
def build_and_predict():
    f = {name: 0 for name in FEAT_NAMES}

    f["Male"] = 1 if sex == "Male" else 0

    age_map = {"18–24":"Age_18_24","25–34":"Age_25_34","35–44":"Age_35_44",
               "45–54":"Age_45_54","55–64":"Age_55_64","65+":"Age_65_plus"}
    f[age_map[age_group]] = 1

    race_map = {"White":"Race_white","Black":"Race_black","American Indian":"Race_am_indian",
                "Asian":"Race_asian","Pacific Islander":"Race_pacific","Hispanic":"Race_hispanic"}
    f[race_map[race]] = 1

    edu_map = {"No High School":"Edu_no_hs","High School Graduate":"Edu_hs_grad",
               "Some College":"Edu_some_college","College Graduate":"Edu_college_grad"}
    f[edu_map[education]] = 1

    emp_map = {"Employed":"Emp_employed","Self-employed":"Emp_self",
               "Unemployed >1yr":"Emp_out_1yr","Unemployed <1yr":"Emp_out_less1yr",
               "Homemaker":"Emp_homemaker","Student":"Emp_student",
               "Retired":"Emp_retired","Unable to work":"Emp_unable"}
    f[emp_map[employment]] = 1

    f["Sleep_hours"]       = float(sleep)
    f["Physical_activity"] = 1 if exercise == "Yes" else 0
    f["Alcohol"]           = 1 if alcohol  == "Yes" else 0
    f["Overweight_obese"]  = 1 if bmi_cat  == "Overweight / Obese" else 0

    smoke_map = {"Never smoked":"Smoking_never","Former smoker":"Smoking_former",
                 "Smoke some days":"Smoking_someday","Smoke every day":"Smoking_everyday"}
    f[smoke_map[smoking]] = 1

    f["Stroke"]         = 1 if stroke     == "Yes" else 0
    f["Kidney_disease"] = 1 if kidney     == "Yes" else 0
    f["COPD"]           = 1 if copd       == "Yes" else 0
    f["Arthritis"]      = 1 if arthritis  == "Yes" else 0
    f["Depression"]     = 1 if depression == "Yes" else 0
    f["Cancer"]         = 1 if cancer     == "Yes" else 0

    diab_map = {"No":"Diabetes_no","Pre-diabetes":"Diabetes_pre",
                "Gestational (past)":"Diabetes_gest","Yes":"Diabetes_yes"}
    f[diab_map[diabetes]] = 1

    gh_map = {"Excellent":"GenHealth_excellent","Very Good":"GenHealth_very_good",
              "Good":"GenHealth_good","Fair":"GenHealth_fair","Poor":"GenHealth_poor"}
    f[gh_map[gen_health]] = 1
    f["PhysHealth_days"] = float(phys_days)
    f["MentHealth_days"] = float(ment_days)

    X = pd.DataFrame([f])[FINAL_ORDER]
    scaler = META.get("scaler", {})
    if scaler:
        for i, col in enumerate(scaler["cont_cols"]):
            X[col] = (X[col] - scaler["mean"][i]) / (scaler["std"][i] + 1e-8)

    prob = float(model.predict_proba(X.values)[0, 1])
    return prob


# ── Main layout ───────────────────────────────────────────────
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("### 📊 Model Performance (on BRFSS 2022 Test Set)")
    m = META["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sensitivity", f"{m['sensitivity']:.1%}", help="CVD patients correctly detected")
    c2.metric("Specificity", f"{m['specificity']:.1%}", help="Healthy patients correctly classified")
    c3.metric("G-mean",      f"{m['gmean']:.3f}",       help="Geometric mean of Sens & Spec")
    c4.metric("AUC",         f"{m['auc']:.3f}",         help="Area under ROC curve")

    st.divider()
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
1. Fill in all sections in the **sidebar** on the left
2. Click **Predict CVD Risk**
3. Review your risk assessment and key factors

> ⚠️ **Disclaimer:** This tool is for educational/research purposes only and is **not** a substitute for professional medical advice.
    """)

    st.markdown("### 🔬 About This Model")
    ti = META["training_info"]
    st.markdown(f"""
| Property | Value |
|---|---|
| Dataset | {ti['dataset']} ({ti['n_samples']:,} records) |
| Features | {ti['n_features']} engineered features |
| Model | {ti['model']} |
| Tuning | {ti['tuning']} |
| Decision Threshold | {THRESHOLD:.2f} (G-mean optimal) |
| CVD Prevalence | {ti['cvd_rate']:.1%} in dataset |
    """)

with col2:
    st.markdown("### 🔮 Risk Prediction")

    if predict_btn:
        with st.spinner("Analysing risk factors..."):
            prob     = build_and_predict()
            pred     = int(prob >= THRESHOLD)
            risk_pct = prob * 100

        if pred == 1:
            st.markdown(f"""
<div class="risk-high">
    <h2 style="color:#C0392B;margin:0">⚠️ HIGH RISK</h2>
    <h1 style="color:#C0392B;margin:0.3rem 0">{risk_pct:.1f}%</h1>
    <p style="color:#7F8C8D;margin:0">CVD Probability Score</p>
</div>""", unsafe_allow_html=True)
            st.error("This profile shows elevated CVD risk. Please consult a healthcare professional.")
        else:
            st.markdown(f"""
<div class="risk-low">
    <h2 style="color:#27AE60;margin:0">✅ LOW RISK</h2>
    <h1 style="color:#27AE60;margin:0.3rem 0">{risk_pct:.1f}%</h1>
    <p style="color:#7F8C8D;margin:0">CVD Probability Score</p>
</div>""", unsafe_allow_html=True)
            st.success("This profile shows low CVD risk. Maintain a healthy lifestyle.")

        st.markdown("#### Probability Breakdown")
        st.progress(prob,       text=f"CVD Risk: {risk_pct:.1f}%")
        st.progress(1.0 - prob, text=f"No CVD:   {(1-prob)*100:.1f}%")

        st.markdown("#### ⚡ Risk Factors Detected")
        factors = []
        if age_group in ["55–64","65+"]:     factors.append(f"🔴 Age {age_group} — high-risk group")
        if sex == "Male":                    factors.append("🔴 Male sex — elevated baseline risk")
        if stroke == "Yes":                  factors.append("🔴 Stroke history — strong CVD indicator")
        if copd == "Yes":                    factors.append("🔴 COPD — systemic inflammation")
        if diabetes == "Yes":               factors.append("🔴 Diabetes — metabolic risk factor")
        if kidney == "Yes":                  factors.append("🔴 Kidney disease — vascular risk")
        if smoking == "Smoke every day":     factors.append("🟠 Daily smoker — significant risk")
        if bmi_cat == "Overweight / Obese":  factors.append("🟠 Overweight/Obese")
        if phys_days > 14:                   factors.append(f"🟠 {phys_days} poor physical health days")
        if sleep < 6:                        factors.append(f"🟠 Short sleep ({sleep}h) — cardiovascular risk")
        if gen_health in ["Poor","Fair"]:    factors.append(f"🟠 {gen_health} general health")
        if exercise == "Yes":               factors.append("🟢 Regular exercise — protective factor")
        if gen_health == "Excellent":        factors.append("🟢 Excellent general health — protective")

        for rf in (factors or ["- No major risk factors detected"]):
            st.markdown(f"- {rf}")

        st.caption(
            f"Threshold: {THRESHOLD:.2f} (G-mean optimal) | "
            f"Sensitivity: {m['sensitivity']:.1%} | "
            f"Specificity: {m['specificity']:.1%}"
        )

    else:
        st.info("👈 Fill in patient information in the sidebar, then click **Predict CVD Risk**")

        # ── SHAP table — reads from metadata if available, else shows placeholder ──
        st.markdown("**Top CVD Risk Factors (from SHAP analysis):**")
        if SHAP_TOP and SHAP_TOP[0].get("shap") is not None:
            rows = "".join(
                f"| {s['rank']} | {s['feature']} | {s['shap']:.3f} |\n"
                for s in SHAP_TOP[:5]
            )
            st.markdown(
                "| Rank | Feature | Mean SHAP |\n"
                "|---|---|---|\n" + rows
            )
        else:
            st.markdown("""
| Rank | Feature | Mean SHAP |
|---|---|---|
| 1 | Age_65_plus | — |
| 2 | Male | — |
| 3 | Sleep_hours | — |
| 4 | PhysHealth_days | — |
| 5 | GenHealth_excellent | — |

*Run NB3B SHAP section and add `shap_top10` to model_metadata.json to populate this table.*
            """)


st.markdown("""
<div class="footer">
    HeartCare CVD Risk Predictor · BRFSS 2022 · XGBoost + SMOTE (scale_pos_weight=1) + Optuna<br>
    Big Data Pipeline: Hadoop HDFS + Apache Spark 3.5.1<br>
    Built for academic demonstration purposes only
</div>
""", unsafe_allow_html=True)
