
# app.py
# Streamlit UI for "Career Path Recommendation using Sequence Models"
# Run:
#   pip install streamlit plotly pandas numpy
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

# -----------------------------
# Page setup + styling
# -----------------------------
st.set_page_config(
    page_title="Career Path Recommendation System",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      .kpi-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(6px);
      }
      .small-muted { color: rgba(49, 51, 63, 0.65); font-size: 0.9rem; }
      .section-title { font-size: 1.1rem; font-weight: 700; margin: 0.25rem 0 0.5rem; }
      .pill {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        border: 1px solid rgba(49, 51, 63, 0.25); margin-right: 6px; font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Mock data + helpers
# -----------------------------
ROLE_CATALOG = [
    "Data Analyst",
    "Business Analyst",
    "BI Developer",
    "Data Engineer",
    "Analytics Engineer",
    "Machine Learning Engineer",
    "Data Scientist",
    "Senior Data Scientist",
    "Applied Scientist",
    "ML Ops Engineer",
    "Product Analyst",
    "Product Manager (Data)",
    "Data Architect",
    "Engineering Manager (Data)",
]

SKILL_CATALOG = [
    "SQL", "Python", "R", "Excel", "Power BI", "Tableau", "Looker",
    "Statistics", "Experimentation", "A/B Testing", "Time Series",
    "Data Modeling", "ETL", "Airflow", "dbt", "Spark", "Databricks",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes",
    "Machine Learning", "Deep Learning", "NLP", "Recommender Systems",
    "Feature Engineering", "MLOps", "CI/CD", "Git",
    "Communication", "Stakeholder Management", "Product Thinking",
]

DOMAIN_CATALOG = ["General", "FinTech", "Healthcare", "E-commerce", "EdTech", "SaaS", "Media", "Logistics"]

def normalize_skills(skills):
    # simple normalizer for demo
    out = []
    for s in skills:
        ss = s.strip()
        if ss:
            out.append(ss)
    return sorted(list(dict.fromkeys(out)))

def seed_from_inputs(role, years, skills, target, domain):
    # deterministic-ish seed for repeatable mock recommendations
    key = f"{role}|{years}|{','.join(skills)}|{target}|{domain}"
    return abs(hash(key)) % (2**32)

def build_mock_recommendations(current_role, years, skills, target_role=None, domain="General"):
    rng = np.random.default_rng(seed_from_inputs(current_role, years, skills, target_role, domain))

    # candidate pool biased by "adjacent" roles
    adjacency = {
        "Data Analyst": ["BI Developer", "Product Analyst", "Data Scientist", "Analytics Engineer"],
        "Business Analyst": ["Product Analyst", "BI Developer", "Product Manager (Data)", "Data Analyst"],
        "BI Developer": ["Analytics Engineer", "Data Engineer", "Product Analyst", "Data Architect"],
        "Data Engineer": ["Analytics Engineer", "Data Architect", "ML Ops Engineer", "Machine Learning Engineer"],
        "Analytics Engineer": ["Data Engineer", "Data Scientist", "Data Architect", "Product Analyst"],
        "Data Scientist": ["Senior Data Scientist", "Machine Learning Engineer", "Applied Scientist", "Product Manager (Data)"],
        "Senior Data Scientist": ["Applied Scientist", "Engineering Manager (Data)", "Product Manager (Data)", "Machine Learning Engineer"],
        "Machine Learning Engineer": ["ML Ops Engineer", "Applied Scientist", "Engineering Manager (Data)", "Data Architect"],
        "ML Ops Engineer": ["Machine Learning Engineer", "Engineering Manager (Data)", "Data Architect", "Applied Scientist"],
        "Product Analyst": ["Product Manager (Data)", "Data Scientist", "Business Analyst", "Analytics Engineer"],
    }
    pool = adjacency.get(current_role, ROLE_CATALOG.copy())

    # Include target role in pool if provided
    if target_role and target_role in ROLE_CATALOG and target_role not in pool:
        pool = [target_role] + pool

    pool = [r for r in pool if r != current_role]
    pool = list(dict.fromkeys(pool))  # unique preserving order

    top_roles = pool[:8] if len(pool) >= 8 else pool
    rng.shuffle(top_roles)
    top_roles = top_roles[:5]

    # confidence scores (mock)
    raw = rng.uniform(0.55, 0.92, size=len(top_roles))
    raw = np.sort(raw)[::-1]
    conf = (raw / raw.sum()).round(3)

    # skill gaps (mock)
    role_skill_map = {
        "Data Engineer": ["ETL", "Airflow", "Spark", "AWS", "Data Modeling", "dbt"],
        "Analytics Engineer": ["dbt", "SQL", "Data Modeling", "Stakeholder Management"],
        "Data Scientist": ["Statistics", "Machine Learning", "Python", "Experimentation"],
        "Senior Data Scientist": ["Experimentation", "Product Thinking", "Communication", "Machine Learning"],
        "Machine Learning Engineer": ["Machine Learning", "Docker", "Kubernetes", "CI/CD", "MLOps"],
        "ML Ops Engineer": ["MLOps", "CI/CD", "Kubernetes", "Docker", "Monitoring"],
        "Applied Scientist": ["Deep Learning", "NLP", "Feature Engineering", "Recommender Systems"],
        "Product Analyst": ["Experimentation", "A/B Testing", "Communication", "Product Thinking"],
        "Product Manager (Data)": ["Stakeholder Management", "Product Thinking", "Communication", "Analytics"],
        "Data Architect": ["Data Modeling", "Cloud Architecture", "Governance", "Security"],
    }

    user_skill_set = set(skills)
    recs = []
    for i, r in enumerate(top_roles):
        needed = role_skill_map.get(r, ["SQL", "Python", "Communication"])
        gap = [x for x in needed if x not in user_skill_set]
        # timeline estimate (mock): depends on years + gaps
        base_months = rng.integers(3, 10)
        extra = min(10, len(gap) * 2)
        months = int(base_months + extra - min(4, years // 2))
        months = max(2, months)

        recs.append(
            {
                "rank": i + 1,
                "recommended_role": r,
                "confidence": float(conf[i]),
                "estimated_months": months,
                "key_skill_gaps": gap[:5],
            }
        )

    return pd.DataFrame(recs)

def build_mock_career_history(current_role):
    # simple timeline for demo
    # user can edit later; here we generate a plausible chain ending in current_role
    paths = {
        "Data Analyst": ["Intern", "Junior Analyst", "Data Analyst"],
        "Business Analyst": ["Intern", "Analyst", "Business Analyst"],
        "BI Developer": ["Analyst", "BI Analyst", "BI Developer"],
        "Data Engineer": ["Analyst", "ETL Developer", "Data Engineer"],
        "Analytics Engineer": ["Data Analyst", "BI Developer", "Analytics Engineer"],
        "Data Scientist": ["Data Analyst", "Junior Data Scientist", "Data Scientist"],
        "Senior Data Scientist": ["Data Analyst", "Data Scientist", "Senior Data Scientist"],
        "Machine Learning Engineer": ["Software Engineer", "Data Scientist", "Machine Learning Engineer"],
        "ML Ops Engineer": ["DevOps Engineer", "Machine Learning Engineer", "ML Ops Engineer"],
        "Product Analyst": ["Data Analyst", "Product Analyst"],
        "Product Manager (Data)": ["Product Analyst", "Product Manager (Data)"],
        "Data Architect": ["Data Engineer", "Senior Data Engineer", "Data Architect"],
        "Engineering Manager (Data)": ["Senior Data Engineer", "Engineering Manager (Data)"],
        "Applied Scientist": ["Data Scientist", "Senior Data Scientist", "Applied Scientist"],
    }
    seq = paths.get(current_role, ["Analyst", current_role])

    start_year = date.today().year - (len(seq) + 1)
    rows = []
    for i, role in enumerate(seq):
        s = date(start_year + i, 1, 1)
        e = date(start_year + i + 1, 1, 1)
        rows.append({"role": role, "start": s, "end": e})
    return pd.DataFrame(rows)

def plot_career_timeline(df_hist):
    # Plotly timeline-like view (nodes + edges)
    x = [d.toordinal() for d in df_hist["start"].tolist()] + [df_hist["end"].iloc[-1].toordinal()]
    # node positions
    node_x = [d.toordinal() for d in df_hist["start"].tolist()]
    node_y = list(range(len(node_x)))

    # edges connect sequential roles
    edge_x, edge_y = [], []
    for i in range(len(node_x) - 1):
        edge_x += [node_x[i], node_x[i + 1], None]
        edge_y += [node_y[i], node_y[i + 1], None]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            name="Transitions",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=df_hist["role"].tolist(),
            textposition="middle right",
            marker=dict(size=14),
            name="Roles",
        )
    )

    # format x axis as dates
    fig.update_xaxes(
        tickformat="%Y",
        title="Year",
        showgrid=True,
    )
    fig.update_yaxes(
        showticklabels=False,
        title="",
        showgrid=False,
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        title="Career Journey Timeline",
    )
    return fig

def plot_path_explorer_graph(center_role, rec_roles):
    # Simple interactive graph layout (center + recommended next steps + second hop)
    rng = np.random.default_rng(abs(hash(center_role)) % (2**32))

    nodes = [center_role] + rec_roles
    # add second hop expansions
    second_hop = []
    for r in rec_roles:
        picks = rng.choice([x for x in ROLE_CATALOG if x not in nodes], size=2, replace=False).tolist()
        second_hop += picks
    nodes += second_hop
    nodes = list(dict.fromkeys(nodes))

    # positions (radial-ish)
    pos = {}
    pos[center_role] = (0.0, 0.0)

    angle_step = 2 * np.pi / max(1, len(rec_roles))
    for i, r in enumerate(rec_roles):
        pos[r] = (np.cos(i * angle_step) * 1.2, np.sin(i * angle_step) * 1.2)

    # second hop around each rec
    idx = 0
    for r in rec_roles:
        children = [c for c in second_hop if c not in rec_roles and c != center_role][idx: idx + 2]
        idx += 2
        base_x, base_y = pos[r]
        for j, c in enumerate(children):
            pos[c] = (base_x + (j - 0.5) * 0.7, base_y + 0.6)

    # edges
    edges = []
    for r in rec_roles:
        edges.append((center_role, r))
    # connect rec -> two second hop nodes near it
    for r in rec_roles:
        children = [n for n in nodes if n not in [center_role] + rec_roles and abs(pos[n][0] - pos[r][0]) < 1.2 and abs(pos[n][1] - pos[r][1]) < 1.2]
        for c in children[:2]:
            edges.append((r, c))

    edge_x, edge_y = [], []
    for a, b in edges:
        ax, ay = pos[a]
        bx, by = pos[b]
        edge_x += [ax, bx, None]
        edge_y += [ay, by, None]

    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_text = nodes

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", name="Paths"))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="bottom center",
            marker=dict(size=14),
            name="Roles",
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        title="Career Path Explorer (Graph View)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

def plot_model_insights(recs_df):
    # mock transition probabilities + attention weights
    roles = recs_df["recommended_role"].tolist()
    probs = (recs_df["confidence"] / recs_df["confidence"].sum()).values

    fig_probs = go.Figure()
    fig_probs.add_trace(go.Bar(x=roles, y=probs, name="Transition Probability (Mock)"))
    fig_probs.update_layout(
        height=360,
        title="Transition Probabilities (Mock)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Next Role",
        yaxis_title="Probability",
    )

    # attention heatmap (mock): tokens = last 6 roles/skills "events"
    tokens = ["Role(t-3)", "Role(t-2)", "Role(t-1)", "Skill:SQL", "Skill:Python", "Skill:ML"]
    rng = np.random.default_rng(123)
    attn = rng.uniform(0, 1, size=(len(tokens), len(tokens)))
    attn = attn / attn.sum(axis=1, keepdims=True)

    fig_attn = go.Figure(
        data=go.Heatmap(
            z=attn,
            x=tokens,
            y=tokens,
            colorbar=dict(title="Weight"),
        )
    )
    fig_attn.update_layout(
        height=420,
        title="Transformer Attention Weights (Mock)",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig_probs, fig_attn

# -----------------------------
# State init
# -----------------------------
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "current_role": "Data Analyst",
        "years_exp": 2,
        "domain": "General",
        "target_role": "",
        "skills": ["SQL", "Python", "Excel"],
    }

if "career_history" not in st.session_state:
    st.session_state.career_history = build_mock_career_history(st.session_state.inputs["current_role"])

if "recs" not in st.session_state:
    st.session_state.recs = build_mock_recommendations(
        st.session_state.inputs["current_role"],
        st.session_state.inputs["years_exp"],
        st.session_state.inputs["skills"],
        st.session_state.inputs["target_role"] or None,
        st.session_state.inputs["domain"],
    )

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.markdown("### 🧭 Career Path Recommendation")
page = st.sidebar.radio(
    "Navigate",
    [
        "Landing / Input",
        "Career Journey Visualization",
        "Recommendation Dashboard",
        "Career Path Explorer",
        "Model Insights",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="small-muted">Tip: Start on Landing/Input, then explore the other views.</div>', unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("## Career Path Recommendation System")
st.markdown('<div class="small-muted">Streamlit UI prototype — clean, professional, and optimized for data professionals.</div>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Page: Landing / Input
# -----------------------------
if page == "Landing / Input":
    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown('<div class="section-title">Landing / Input Screen</div>', unsafe_allow_html=True)

        current_role = st.selectbox(
            "Current role",
            ROLE_CATALOG,
            index=ROLE_CATALOG.index(st.session_state.inputs["current_role"])
            if st.session_state.inputs["current_role"] in ROLE_CATALOG else 0,
        )

        years_exp = st.slider("Years of experience", 0, 20, int(st.session_state.inputs["years_exp"]))
        domain = st.selectbox("Domain (optional)", DOMAIN_CATALOG, index=DOMAIN_CATALOG.index(st.session_state.inputs["domain"]))

        target_role = st.selectbox(
            "Target role (optional)",
            [""] + ROLE_CATALOG,
            index=([""] + ROLE_CATALOG).index(st.session_state.inputs["target_role"]) if st.session_state.inputs["target_role"] in ([""] + ROLE_CATALOG) else 0,
            help="If selected, recommendations will be lightly biased toward this role (demo behavior).",
        )

        st.markdown("**Skills**")
        skills = st.multiselect(
            "Type to search (autocomplete) and select skills",
            options=SKILL_CATALOG,
            default=st.session_state.inputs["skills"],
            help="Autocomplete is provided by multiselect search; add your core skills.",
        )

        # Optional: allow custom skills not in catalog
        custom_skill = st.text_input("Add a custom skill (optional)")
        if custom_skill.strip():
            st.caption("Custom skills will be included for gap analysis.")
        final_skills = normalize_skills(skills + ([custom_skill.strip()] if custom_skill.strip() else []))

        run = st.button("Generate Recommendations", type="primary", use_container_width=True)

        if run:
            st.session_state.inputs = {
                "current_role": current_role,
                "years_exp": years_exp,
                "domain": domain,
                "target_role": target_role,
                "skills": final_skills,
            }
            st.session_state.career_history = build_mock_career_history(current_role)
            st.session_state.recs = build_mock_recommendations(
                current_role, years_exp, final_skills, target_role or None, domain
            )
            st.success("Recommendations updated. Use the sidebar to explore visualizations and insights.")

    with col2:
        st.markdown('<div class="section-title">Current Profile Snapshot</div>', unsafe_allow_html=True)

        inputs = st.session_state.inputs
        st.markdown(
            f"""
            <div class="kpi-card">
              <div><b>Role:</b> {inputs["current_role"]}</div>
              <div><b>Experience:</b> {inputs["years_exp"]} years</div>
              <div><b>Domain:</b> {inputs["domain"]}</div>
              <div><b>Target:</b> {inputs["target_role"] if inputs["target_role"] else "—"}</div>
              <div style="margin-top:10px;"><b>Skills:</b></div>
              <div>{" ".join([f'<span class="pill">{s}</span>' for s in inputs["skills"][:18]])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.info(
            "This prototype wires the full UI flow. Replace the mock recommendation logic with your Markov/LSTM/Transformer backend."
        )

# -----------------------------
# Page: Career Journey Visualization
# -----------------------------
elif page == "Career Journey Visualization":
    st.markdown('<div class="section-title">Career Journey Visualization Screen</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Interactive timeline using Plotly. Nodes = roles, edges = transitions.</div>', unsafe_allow_html=True)

    left, right = st.columns([0.75, 0.25], gap="large")
    with right:
        st.markdown("**Edit career history (optional)**")
        df_hist = st.session_state.career_history.copy()
        edited = st.data_editor(
            df_hist,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "role": st.column_config.TextColumn("Role"),
                "start": st.column_config.DateColumn("Start"),
                "end": st.column_config.DateColumn("End"),
            },
        )
        if st.button("Update Timeline", use_container_width=True):
            # basic sanity: sort by start date
            edited = edited.sort_values("start").reset_index(drop=True)
            st.session_state.career_history = edited
            st.success("Timeline updated.")

    with left:
        fig = plot_career_timeline(st.session_state.career_history)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Page: Recommendation Dashboard
# -----------------------------
elif page == "Recommendation Dashboard":
    st.markdown('<div class="section-title">Recommendation Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-muted">Top-5 next steps with confidence, skill gaps, and estimated transition timelines.</div>',
        unsafe_allow_html=True,
    )

    recs = st.session_state.recs.copy()
    inputs = st.session_state.inputs

    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="small-muted">Current Role</div><div style="font-size:1.2rem;"><b>{inputs["current_role"]}</b></div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="small-muted">Experience</div><div style="font-size:1.2rem;"><b>{inputs["years_exp"]} yrs</b></div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="small-muted">Domain</div><div style="font-size:1.2rem;"><b>{inputs["domain"]}</b></div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="small-muted">Target</div><div style="font-size:1.2rem;"><b>{inputs["target_role"] or "—"}</b></div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Table view with pretty fields
    display_df = recs.copy()
    display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.3f}")
    display_df["estimated_timeline"] = recs["estimated_months"].map(lambda m: f"{m} months")
    display_df["skill_gap"] = recs["key_skill_gaps"].map(lambda g: ", ".join(g) if g else "No major gaps")
    display_df = display_df[["rank", "recommended_role", "confidence", "estimated_timeline", "skill_gap"]]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Expanders per recommendation
    st.markdown("### Skill Gap Analysis")
    for _, row in recs.iterrows():
        with st.expander(f'#{int(row["rank"])} — {row["recommended_role"]} (confidence {row["confidence"]:.3f})', expanded=False):
            gaps = row["key_skill_gaps"]
            if not gaps:
                st.success("No major skill gaps detected (based on the demo role-skill map).")
            else:
                st.write("Recommended focus areas:")
                st.write("• " + "\n• ".join(gaps))
            st.write(f"Estimated transition timeline: **{int(row['estimated_months'])} months** (mock).")

# -----------------------------
# Page: Career Path Explorer
# -----------------------------
elif page == "Career Path Explorer":
    st.markdown('<div class="section-title">Career Path Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-muted">Interactive graph of multiple possible career paths from your current role.</div>',
        unsafe_allow_html=True,
    )

    inputs = st.session_state.inputs
    recs = st.session_state.recs
    rec_roles = recs["recommended_role"].tolist()

    left, right = st.columns([0.72, 0.28], gap="large")
    with right:
        st.markdown("**Explorer controls**")
        depth = st.slider("Exploration depth (demo)", 1, 3, 2)
        show_recs = st.multiselect("Highlight next-step roles", rec_roles, default=rec_roles[:3])
        st.caption("Depth is a UI control placeholder. Replace with real multi-hop inference later.")

        st.markdown("---")
        st.markdown("**Legend**")
        st.write("• Center node: current role")
        st.write("• First ring: recommended next steps")
        st.write("• Second ring: illustrative long-term paths (mock)")

    with left:
        fig = plot_path_explorer_graph(inputs["current_role"], rec_roles)
        st.plotly_chart(fig, use_container_width=True)

        if show_recs:
            st.markdown("#### Highlighted roles")
            st.write(", ".join(show_recs))

# -----------------------------
# Page: Model Insights
# -----------------------------
elif page == "Model Insights":
    st.markdown('<div class="section-title">Model Insights Panel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-muted">For advanced users: transition probabilities, attention weights, and confidence metrics.</div>',
        unsafe_allow_html=True,
    )

    recs = st.session_state.recs.copy()
    fig_probs, fig_attn = plot_model_insights(recs)

    c1, c2 = st.columns([0.55, 0.45], gap="large")
    with c1:
        st.plotly_chart(fig_probs, use_container_width=True)

        st.markdown("### Confidence Metrics (Mock)")
        metrics = pd.DataFrame(
            {
                "metric": ["Top-1 confidence", "Top-5 entropy", "Calibration error (ECE)"],
                "value": [
                    float(recs["confidence"].iloc[0]),
                    float(-(recs["confidence"] * np.log(recs["confidence"] + 1e-9)).sum()),
                    float(np.clip(np.random.default_rng(1).normal(0.06, 0.01), 0.01, 0.15)),
                ],
            }
        )
        st.dataframe(metrics, use_container_width=True, hide_index=True)

        st.caption("Replace these with real outputs: log-likelihood/perplexity, calibration, top-k accuracy, etc.")

    with c2:
        st.plotly_chart(fig_attn, use_container_width=True)
        st.caption("Attention shown is a mock heatmap. Swap in actual attention weights from your Transformer.")

    st.markdown("---")
    with st.expander("Backend integration notes (where to connect real models)"):
        st.write(
            """
- Replace `build_mock_recommendations(...)` with calls to your inference layer:
  - Markov / n-gram transition matrix
  - LSTM / Transformer next-role prediction (Top-K)
- Store outputs in `st.session_state.recs` with:
  - recommended_role, confidence, key_skill_gaps, estimated_months
- For Model Insights:
  - transition probabilities (Markov / softmax logits)
  - attention weights (Transformer)
  - confidence/calibration metrics
            """
        )
