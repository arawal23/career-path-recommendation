
# app.py
# Streamlit UI for "Career Path Recommendation using Sequence Models"
# Run:
#   pip install streamlit plotly pandas numpy
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

st.set_page_config(
    page_title="Career Path Recommendation System",
    page_icon="🧭",
    layout="wide",
)

ROLE_CATALOG = [
    "Data Analyst","Business Analyst","BI Developer","Data Engineer",
    "Analytics Engineer","Machine Learning Engineer","Data Scientist",
    "Senior Data Scientist","Applied Scientist","ML Ops Engineer",
    "Product Analyst","Product Manager (Data)","Data Architect",
    "Engineering Manager (Data)",
]

SKILL_CATALOG = [
    "SQL","Python","R","Excel","Power BI","Tableau","Looker",
    "Statistics","Experimentation","A/B Testing","Time Series",
    "Data Modeling","ETL","Airflow","dbt","Spark","Databricks",
    "AWS","GCP","Azure","Docker","Kubernetes",
    "Machine Learning","Deep Learning","NLP","Recommender Systems",
    "Feature Engineering","MLOps","CI/CD","Git",
    "Communication","Stakeholder Management","Product Thinking",
]

def build_mock_recommendations(current_role):
    rng = np.random.default_rng(42)
    pool = [r for r in ROLE_CATALOG if r != current_role]
    rng.shuffle(pool)
    top_roles = pool[:5]
    conf = np.sort(rng.uniform(0.6, 0.95, size=5))[::-1]

    recs = []
    for i, r in enumerate(top_roles):
        recs.append({
            "rank": i+1,
            "recommended_role": r,
            "confidence": round(float(conf[i]),3),
            "estimated_months": int(rng.integers(3,12)),
            "key_skill_gaps": rng.choice(SKILL_CATALOG, size=3, replace=False).tolist()
        })
    return pd.DataFrame(recs)

def build_mock_career_history(current_role):
    seq = ["Intern","Junior Analyst",current_role]
    start_year = date.today().year - 3
    rows = []
    for i, role in enumerate(seq):
        rows.append({
            "role": role,
            "start": date(start_year+i,1,1),
            "end": date(start_year+i+1,1,1)
        })
    return pd.DataFrame(rows)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Landing / Input",
    "Career Journey Visualization",
    "Recommendation Dashboard",
])

if "current_role" not in st.session_state:
    st.session_state.current_role = "Data Analyst"
    st.session_state.recs = build_mock_recommendations("Data Analyst")
    st.session_state.history = build_mock_career_history("Data Analyst")

if page == "Landing / Input":
    st.title("Career Path Recommendation System")

    role = st.selectbox("Current Role", ROLE_CATALOG)
    skills = st.multiselect("Skills", SKILL_CATALOG)
    years = st.slider("Years of Experience",0,20,2)

    if st.button("Generate Recommendations"):
        st.session_state.current_role = role
        st.session_state.recs = build_mock_recommendations(role)
        st.session_state.history = build_mock_career_history(role)
        st.success("Recommendations generated!")

elif page == "Career Journey Visualization":
    st.title("Career Journey Visualization")
    df = st.session_state.history

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["start"],
        y=list(range(len(df))),
        mode="markers+text",
        text=df["role"],
        textposition="middle right"
    ))
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Recommendation Dashboard":
    st.title("Recommendation Dashboard")
    st.dataframe(st.session_state.recs, use_container_width=True)
