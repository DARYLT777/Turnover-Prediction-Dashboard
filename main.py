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
if "intervention_log" not in st.session_state:
    st.session_state["intervention_log"] = []
st.subheader("Log a Retention Intervention (Demo)")

with st.form("intervention_form"):
    target_group = st.selectbox(
        "Target Group",
        options=[
            "High-Risk employees in current filter",
            "Medium-Risk employees in current filter",
            "All employees in current filter",
        ],
    )
    intervention_type = st.selectbox(
        "Intervention Type",
        options=[
            "Stay interviews",
            "Recognition plan",
            "Schedule review",
            "Development plan",
            "Mentoring / coaching",
        ],
    )
    notes = st.text_area("Notes (optional)")
    submitted = st.form_submit_button("Log Intervention")

    if submitted:
        st.session_state["intervention_log"].append({
            "target_group": target_group,
            "intervention_type": intervention_type,
            "notes": notes,
        })
        st.success("Intervention logged for this session (demo).")
if st.session_state["intervention_log"]:
    st.markdown("### Intervention Log (Session Only)")
    st.dataframe(pd.DataFrame(st.session_state["intervention_log"]))
else:
    st.caption("No interventions logged yet in this session.")

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
    
    df = pd.read_csv("synthetic_turnover_data.csv")
# ---- Drift Monitoring Baseline (fake prior period) ----
BASELINE_ABSENTEEISM = 1.2     # Replace if needed
BASELINE_SCHED_VAR = 4.0       # Replace if needed
DRIFT_THRESHOLD = 0.5          # Threshold used to flag drift
st.title("Predictive Turnover Dashboard")

departments = df["department"].unique()
selected_dept = st.selectbox("Department", departments)

filtered = df[df["department"] == selected_dept]
# ---- Drift Monitoring Calculation ----
if len(filtered) >= PRIVACY_N:
    current_abs = filtered["absenteeism_90d"].mean()
    current_sched = filtered["schedule_variability"].mean()

    abs_diff = current_abs - BASELINE_ABSENTEEISM
    sched_diff = current_sched - BASELINE_SCHED_VAR

    st.subheader("Drift Monitor (Absenteeism & Schedule Variability)")

    col_dm1, col_dm2 = st.columns(2)
    with col_dm1:
        st.metric(
            "Avg Absences (90d)",
            f"{current_abs:.2f}",
            delta=f"{abs_diff:+.2f} vs. baseline"
        )
    with col_dm2:
        st.metric(
            "Avg Schedule Variability",
            f"{current_sched:.2f}",
            delta=f"{sched_diff:+.2f} vs. baseline"
        )

    if abs(abs_diff) > DRIFT_THRESHOLD or abs(sched_diff) > DRIFT_THRESHOLD:
        st.warning(
            "⚠️ **Potential Drift Detected:** Absenteeism or schedule variability has shifted materially "
            "from the baseline. Investigate whether this reflects real operational changes or data issues."
        )
    else:
        st.success("✅ Drift within expected range for absenteeism and schedule variability.")

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

st.title("AI-Enabled Predictive Turnover Dashboard (MVP)")st.caption(
    "Use the **Turnover Risk Explorer** for decisions, and scroll to **Fairness & Validity Snapshot** to review parity and model behavior."
)
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
    st.subheader("Turnover Risk Explorer")

st.warning(
    "⚠️ **Ethical Use Reminder:** These risk scores are designed for *proactive retention and support only*.\n\n"
    "- Do **not** use them for discipline or termination decisions.\n"
    "- Always combine this dashboard with manager judgment, context, and HR policy.\n"
)

explorer_cols = [
    "employee_id",
    "department",
    "location",
    "tenure_months",
    "engagement_score",
    "absenteeism_90d",
    "schedule_variability",
    "performance_rating",
    "prediction_prob",
    "risk_flag",
]
...
st.data_editor(
    explorer_df.sort_values("prediction_prob", ascending=False),
    use_container_width=True,
    disabled=True,
    column_config={
        "tenure_months": st.column_config.NumberColumn(
            "Tenure (months)",
            help="How long the employee has worked for the organization. Shorter tenure often predicts higher turnover risk."
        ),
        "engagement_score": st.column_config.NumberColumn(
            "Engagement Score",
            help="Composite engagement rating (1–5). Lower scores signal reduced motivation and attachment."
        ),
        "absenteeism_90d": st.column_config.NumberColumn(
            "Absences (90 days)",
            help="Number of missed shifts in the last 90 days. Higher absence counts can predict withdrawal."
        ),
        "schedule_variability": st.column_config.NumberColumn(
            "Schedule Variability",
            help="How inconsistent the schedule is (higher values = more unstable). High variability is linked to burnout and turnover."
        ),
        "performance_rating": st.column_config.NumberColumn(
            "Performance Rating",
            help="Overall performance assessment (1–5). Very low ratings can be associated with turnover risk."
        ),
        "prediction_prob": st.column_config.NumberColumn(
            "Turnover Risk (Prob.)",
            help="Model-estimated probability that this employee will leave in the next ~90 days (synthetic data)."
        ),
        "risk_flag": st.column_config.TextColumn(
            "Risk Flag",
            help="Categorical risk band derived from the probability: Low, Medium, or High."
        ),
    },
)



st.subheader("Fairness & Validity Snapshot (Tenure Bands)")

tmp = filtered.copy()
tmp["high_risk_flag"] = (tmp["risk_flag"] == "High").astype(int)
air_tenure = compute_air_by_group(tmp, "tenure_band", "high_risk_flag")

st.write("**Adverse Impact Ratio (AIR) on High-Risk Flags by Tenure Band**")
st.dataframe(air_tenure)

tenure_groups = {
    "0-12": df[df["tenure_months"] <= 12],
    "13-36": df[(df["tenure_months"] > 12) & (df["tenure_months"] <= 36)],
    "37-60": df[(df["tenure_months"] > 36) & (df["tenure_months"] <= 60)],
    "60+": df[df["tenure_months"] > 60]
}
st.sidebar.markdown("### 🔍 Fairness & Validity")
st.sidebar.caption("Scroll to the 'Fairness & Validity Snapshot' section in the main view to review AIR and correlation.")

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
