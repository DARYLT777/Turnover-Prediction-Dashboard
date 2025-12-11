import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import hashlib

def hash_id(x):
    import hashlib
    return hashlib.sha256(str(x).encode('utf-8')).hexdigest()[:10]

def categorize_risk(prob):
    if prob >= 0.60:
        return "High"
    elif prob >= 0.35:
        return "Medium"
    else:
        return "Low"

def adverse_impact_ratio(group_pos, ref_pos):
    if ref_pos == 0:
        return None
    return group_pos / ref_pos

# Example synthetic data
data = pd.DataFrame({
    "employee_id": range(1, 101),
    "tenure_months": np.random.randint(1, 80, 100),
    "engagement_score": np.random.uniform(2.5, 4.8, 100),
    "absenteeism_90d": np.random.randint(0, 8, 100),
    "schedule_variability": np.random.uniform(0, 1.5, 100),
    "performance_rating": np.random.randint(1, 5, 100),
    "department": np.random.choice(["Front Desk", "Housekeeping", "Food & Bev", "Spa"], 100),
    "location": np.random.choice(["Temecula", "Carlsbad", "Palm Springs"], 100),
    "turnover_next90": np.random.binomial(1, 0.18, 100)
})

data["employee_id"] = data["employee_id"].apply(hash_id)

features = ["tenure_months", "engagement_score", "absenteeism_90d",
            "schedule_variability", "performance_rating"]

X = data[features]
y = data["turnover_next90"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

data["prediction_prob"] = model.predict_proba(X_scaled)[:, 1]
data["risk_flag"] = data["prediction_prob"].apply(categorize_risk)

st.title("AI-Enabled Predictive Turnover Dashboard (MVP)")
st.info(
    "Ethical Use Reminder: This dashboard is designed for proactive retention and well-being support, "
    "not for discipline or termination decisions. Always combine model output with human judgment."
)

dept_filter = st.selectbox("Department", ["All"] + sorted(data["department"].unique()))
loc_filter = st.selectbox("Location", ["All"] + sorted(data["location"].unique()))

df = data.copy()

if dept_filter != "All":
    df = df[df["department"] == dept_filter]

if loc_filter != "All":
    df = df[df["location"] == loc_filter]

if len(df) < 5:
    st.warning("For privacy reasons, groups with fewer than 5 employees cannot be displayed.")
else:
    st.subheader("Turnover Risk Overview")
    st.dataframe(df[["employee_id", "department", "location",
                     "prediction_prob", "risk_flag"]])

    st.subheader("Suggested Interventions (Theory-Based)")
    st.write("""
    - **Low Risk:** Maintain engagement through recognition and consistent scheduling (POS, commitment).
    - **Medium Risk:** Conduct a stay interview; review workload; improve role clarity.
    - **High Risk:** Escalate to HRBP; explore burnout and schedule volatility; co-create a retention plan.
    """)

st.sidebar.header("Fairness & Validity Panel")

tenure_groups = {
    "0-12": df[df["tenure_months"] <= 12],
    "13-36": df[(df["tenure_months"] > 12) & (df["tenure_months"] <= 36)],
    "37-60": df[(df["tenure_months"] > 36) & (df["tenure_months"] <= 60)],
    "60+": df[df["tenure_months"] > 60]
}

st.sidebar.subheader("Adverse Impact Ratio (AIR)")
ref_rate = tenure_groups["13-36"]["risk_flag"].value_counts(normalize=True).get("High", 0)

for tg, subset in tenure_groups.items():
    high_rate = subset["risk_flag"].value_counts(normalize=True).get("High", 0)
    air = adverse_impact_ratio(high_rate, ref_rate)
    if air is not None:
        st.sidebar.write(f"{tg}: AIR = {air:.2f}")
    else:
        st.sidebar.write(f"{tg}: N/A")

st.sidebar.subheader("Validity Check (Correlation)")
corr = df["prediction_prob"].corr(df["turnover_next90"])
st.sidebar.write(f"Correlation with actual turnover: r = {corr:.2f}")
