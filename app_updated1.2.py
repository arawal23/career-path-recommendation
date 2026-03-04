
# app.py
# Streamlit UI (Interactive Demo) for Career Path Recommendation using Sequence Models
# Features:
# - Landing/Input with skill autocomplete (multiselect)
# - Career Journey Timeline (Plotly)
# - Recommendation Dashboard (Top-5, skill gaps, timelines)
# - Career Path Explorer (NEW): Multi-step Sankey + probability threshold slider
# - Model Insights (mock probabilities + attention heatmap)
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py

import math
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page setup + styling
# -----------------------------
st.set_page_config(page_title="Career Path Recommendation System", page_icon="🧭", layout="wide")

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
        border: 1px solid rgba(49, 51, 63, 0.25); margin-right: 6px; margin-bottom: 6px; font-size: 0.85rem;
      }
      .hint { font-size: 0.88rem; color: rgba(49,51,63,0.72); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Catalogs
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
MODEL_CATALOG = ["Markov (Baseline)", "LSTM (Deep)", "Transformer (Deep)"]

ROLE_SKILL_MAP = {
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

# -----------------------------
# Helpers
# -----------------------------
def normalize_skills(skills):
    out = []
    for s in skills:
        ss = str(s).strip()
        if ss:
            out.append(ss)
    return sorted(list(dict.fromkeys(out)))

def seed_from_inputs(role, years, skills, target, domain, model_name=""):
    key = f"{role}|{years}|{','.join(skills)}|{target}|{domain}|{model_name}"
    return abs(hash(key)) % (2**32)

def build_mock_career_history(current_role):
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
    node_x = [d.toordinal() for d in df_hist["start"].tolist()]
    node_y = list(range(len(node_x)))

    edge_x, edge_y = [], []
    for i in range(len(node_x) - 1):
        edge_x += [node_x[i], node_x[i + 1], None]
        edge_y += [node_y[i], node_y[i + 1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", name="Transitions"))
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
    fig.update_xaxes(tickformat="%Y", title="Year", showgrid=True)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), title="Career Journey Timeline")
    return fig

def build_mock_recommendations(current_role, years, skills, target_role=None, domain="General", model_name="Markov (Baseline)"):
    rng = np.random.default_rng(seed_from_inputs(current_role, years, skills, target_role, domain, model_name))

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
    if target_role and target_role in ROLE_CATALOG and target_role not in pool:
        pool = [target_role] + pool

    pool = [r for r in pool if r != current_role]
    pool = list(dict.fromkeys(pool))

    top_roles = pool[:8] if len(pool) >= 8 else pool
    rng.shuffle(top_roles)
    top_roles = top_roles[:5]

    raw = rng.uniform(0.55, 0.92, size=len(top_roles))
    raw = np.sort(raw)[::-1]
    conf = (raw / raw.sum()).round(3)

    user_skill_set = set(skills)
    recs = []
    for i, r in enumerate(top_roles):
        needed = ROLE_SKILL_MAP.get(r, ["SQL", "Python", "Communication"])
        gap = [x for x in needed if x not in user_skill_set]

        base_months = int(rng.integers(3, 10))
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

# -----------------------------
# Sankey-based multi-step explorer (mock transition function)
# -----------------------------
def transition_distribution(from_role, domain="General", model_name="Markov (Baseline)"):
    """
    Returns a list of (to_role, p) for a given from_role.
    MOCK distribution; replace with real model probabilities later.
    """
    temp = {"Markov (Baseline)": 1.0, "LSTM (Deep)": 0.85, "Transformer (Deep)": 0.7}.get(model_name, 1.0)
    rng = np.random.default_rng(seed_from_inputs(from_role, 0, [], "", domain, model_name))
    candidates = [r for r in ROLE_CATALOG if r != from_role]

    domain_bias = {
        "FinTech": ["Data Engineer", "ML Ops Engineer", "Machine Learning Engineer"],
        "Healthcare": ["Data Scientist", "Applied Scientist", "Data Engineer"],
        "E-commerce": ["Product Analyst", "Analytics Engineer", "Data Scientist"],
        "SaaS": ["Analytics Engineer", "Product Manager (Data)", "Data Engineer"],
        "Media": ["Product Analyst", "Data Scientist", "BI Developer"],
        "Logistics": ["Data Engineer", "Data Architect", "Analytics Engineer"],
        "EdTech": ["Data Scientist", "Product Analyst", "Business Analyst"],
        "General": [],
    }.get(domain, [])

    logits = rng.normal(0, 1, size=len(candidates))
    for i, c in enumerate(candidates):
        if c in domain_bias:
            logits[i] += 0.9

    logits = logits / max(1e-6, temp)
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    pairs = list(zip(candidates, probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs

def build_multistep_sankey(start_role, domain, model_name, max_steps=3, top_k=4, threshold=0.08, cap_links=80):
    """
    BFS expansion up to max_steps.
    Keeps edges with p >= threshold and limits to top_k outgoing per node.
    Returns plotly Sankey figure + a dataframe of top paths.
    """
    nodes = []
    node_index = {}

    def get_idx(label):
        if label not in node_index:
            node_index[label] = len(nodes)
            nodes.append(label)
        return node_index[label]

    queue = [(start_role, 0, [start_role], 1.0)]
    visited_expanded = set()
    links = {}
    path_records = []

    get_idx(start_role)

    while queue:
        role, step, path, path_prob = queue.pop(0)

        if step >= max_steps:
            path_records.append({"path": " → ".join(path), "cumulative_prob": path_prob})
            continue

        key = (role, step)
        if key in visited_expanded:
            continue
        visited_expanded.add(key)

        dist = transition_distribution(role, domain=domain, model_name=model_name)
        dist = dist[:max(1, top_k * 2)]
        filtered = [(r, p) for r, p in dist if p >= threshold][:top_k]

        if not filtered:
            path_records.append({"path": " → ".join(path) + " → (no edges over threshold)", "cumulative_prob": path_prob})
            continue

        for nxt, p in filtered:
            src = get_idx(role)
            tgt = get_idx(nxt)
            val = float(p)

            links[(src, tgt)] = links.get((src, tgt), 0.0) + val
            queue.append((nxt, step + 1, path + [nxt], path_prob * val))

        if len(links) >= cap_links:
            break

    sources, targets, values, hover = [], [], [], []
    for (s, t), v in sorted(links.items(), key=lambda x: x[1], reverse=True):
        sources.append(s)
        targets.append(t)
        values.append(v)
        hover.append(f"{nodes[s]} → {nodes[t]}<br>p ≈ {v:.3f}")

    fig = go.Figure(
    data=[
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=nodes,
                pad=18,
                thickness=16,
                color="rgba(30, 144, 255, 0.8)",  # Node color
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(160, 160, 160, 0.4)",  # Link color
            ),
        )
    ]
)

    fig.update_layout(
    font=dict(
        family="Arial",
        size=14,
        color="black"   # Text color
    ),
    title_font=dict(
        family="Arial",
        size=20,
        color="darkblue"
    )
)
    
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Career Path Explorer — {max_steps}-step expansion",
    )

    df_paths = pd.DataFrame(path_records)
    if not df_paths.empty:
        df_paths = df_paths.sort_values("cumulative_prob", ascending=False).head(15)
        df_paths["cumulative_prob"] = df_paths["cumulative_prob"].astype(float)
    return fig, df_paths

def plot_model_insights(recs_df):
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

    tokens = ["Role(t-3)", "Role(t-2)", "Role(t-1)", "Skill:SQL", "Skill:Python", "Skill:ML"]
    rng = np.random.default_rng(123)
    attn = rng.uniform(0, 1, size=(len(tokens), len(tokens)))
    attn = attn / attn.sum(axis=1, keepdims=True)

    fig_attn = go.Figure(
        data=go.Heatmap(z=attn, x=tokens, y=tokens, colorbar=dict(title="Weight"))
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
        "model_name": "Markov (Baseline)",
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
        st.session_state.inputs["model_name"],
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
st.sidebar.markdown('<div class="small-muted">Tip: Use the Sankey explorer for an impressive demo.</div>', unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("## Career Path Recommendation System")
st.markdown('<div class="small-muted">Interactive Streamlit UI prototype — optimized for data professionals.</div>', unsafe_allow_html=True)
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
        )

        model_name = st.selectbox(
            "Model (demo selector)",
            MODEL_CATALOG,
            index=MODEL_CATALOG.index(st.session_state.inputs.get("model_name", "Markov (Baseline)")),
        )

        st.markdown("**Skills**")
        skills = st.multiselect(
            "Type to search (autocomplete) and select skills",
            options=SKILL_CATALOG,
            default=st.session_state.inputs["skills"],
        )

        custom_skill = st.text_input("Add a custom skill (optional)")
        final_skills = normalize_skills(skills + ([custom_skill.strip()] if custom_skill.strip() else []))

        run = st.button("Generate Recommendations", type="primary", use_container_width=True)

        if run:
            st.session_state.inputs = {
                "current_role": current_role,
                "years_exp": years_exp,
                "domain": domain,
                "target_role": target_role,
                "skills": final_skills,
                "model_name": model_name,
            }
            st.session_state.career_history = build_mock_career_history(current_role)
            st.session_state.recs = build_mock_recommendations(
                current_role, years_exp, final_skills, target_role or None, domain, model_name
            )
            st.success("Updated. Now open the Sankey explorer from the sidebar.")

    with col2:
        st.markdown('<div class="section-title">Current Profile Snapshot</div>', unsafe_allow_html=True)
        inputs = st.session_state.inputs

        pills = " ".join([f'<span class="pill">{s}</span>' for s in inputs["skills"][:18]])
        st.markdown(
            f"""
            <div class="kpi-card">
              <div><b>Role:</b> {inputs["current_role"]}</div>
              <div><b>Experience:</b> {inputs["years_exp"]} years</div>
              <div><b>Domain:</b> {inputs["domain"]}</div>
              <div><b>Target:</b> {inputs["target_role"] if inputs["target_role"] else "—"}</div>
              <div><b>Model:</b> {inputs.get("model_name","—")}</div>
              <div style="margin-top:10px;"><b>Skills:</b></div>
              <div>{pills}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.info("Demo UI uses mock transitions. Swap in real model outputs later.")

# -----------------------------
# Page: Career Journey Visualization
# -----------------------------
elif page == "Career Journey Visualization":
    st.markdown('<div class="section-title">Career Journey Visualization Screen</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Interactive timeline using Plotly.</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="small-muted">Top-5 next steps with confidence, skill gaps, and estimated timelines.</div>', unsafe_allow_html=True)

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
        st.markdown(f'<div class="kpi-card"><div class="small-muted">Model</div><div style="font-size:1.2rem;"><b>{inputs.get("model_name","—")}</b></div></div>', unsafe_allow_html=True)

    st.markdown("")

    display_df = recs.copy()
    display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.3f}")
    display_df["estimated_timeline"] = recs["estimated_months"].map(lambda m: f"{m} months")
    display_df["skill_gap"] = recs["key_skill_gaps"].map(lambda g: ", ".join(g) if g else "No major gaps")
    display_df = display_df[["rank", "recommended_role", "confidence", "estimated_timeline", "skill_gap"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("### Skill Gap Analysis")
    for _, row in recs.iterrows():
        with st.expander(f'#{int(row["rank"])} — {row["recommended_role"]} (confidence {row["confidence"]:.3f})', expanded=False):
            gaps = row["key_skill_gaps"]
            if not gaps:
                st.success("No major skill gaps detected (demo role-skill map).")
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
        '<div class="small-muted">Multi-step transition exploration using an interactive Sankey diagram. '
        'Use the controls to change depth, prune low-probability edges, and explore alternatives.</div>',
        unsafe_allow_html=True,
    )

    inputs = st.session_state.inputs

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1.0, 1.0, 0.9], gap="large")
    with ctrl1:
        start_role = st.selectbox("Start role", ROLE_CATALOG, index=ROLE_CATALOG.index(inputs["current_role"]))
        model_name = st.selectbox("Model", MODEL_CATALOG, index=MODEL_CATALOG.index(inputs.get("model_name","Markov (Baseline)")))
    with ctrl2:
        max_steps = st.slider("Steps (depth)", 1, 5, 3)
        top_k = st.slider("Branching (Top-K per role)", 2, 8, 4)
    with ctrl3:
        threshold = st.slider("Probability threshold", 0.00, 0.30, 0.08, 0.01)
        cap_links = st.slider("Max links (performance)", 20, 200, 90, 10)
    with ctrl4:
        domain = st.selectbox("Domain", DOMAIN_CATALOG, index=DOMAIN_CATALOG.index(inputs["domain"]))
        st.markdown('<div class="hint">Tip: Increase threshold to simplify, increase depth to explore long-term.</div>', unsafe_allow_html=True)

    fig, df_paths = build_multistep_sankey(
        start_role=start_role,
        domain=domain,
        model_name=model_name,
        max_steps=max_steps,
        top_k=top_k,
        threshold=threshold,
        cap_links=cap_links,
    )

    left, right = st.columns([0.72, 0.28], gap="large")
    with left:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Hover links for probabilities. Drag nodes to rearrange.")

    with right:
        st.markdown("**Top multi-step trajectories (mock)**")
        if df_paths is None or df_paths.empty:
            st.write("No paths available (try lowering the threshold).")
        else:
            pretty = df_paths.copy()
            pretty["cumulative_prob"] = pretty["cumulative_prob"].map(lambda x: f"{x:.6f}")
            st.dataframe(pretty, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Integration notes**")
        st.write(
            "• Markov: use row-normalized transition probabilities.\n"
            "• LSTM/Transformer: use softmax(logits) for next-role probabilities.\n"
            "• Threshold + Top-K = pruning for interactivity."
        )

# -----------------------------
# Page: Model Insights
# -----------------------------
elif page == "Model Insights":
    st.markdown('<div class="section-title">Model Insights Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">For advanced users: probabilities, attention weights, and confidence metrics.</div>', unsafe_allow_html=True)

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
                    float(recs["confidence"].iloc[0]) if not recs.empty else 0.0,
                    float(-(recs["confidence"] * np.log(recs["confidence"] + 1e-9)).sum()) if not recs.empty else 0.0,
                    float(np.clip(np.random.default_rng(1).normal(0.06, 0.01), 0.01, 0.15)),
                ],
            }
        )
        st.dataframe(metrics, use_container_width=True, hide_index=True)

    with c2:
        st.plotly_chart(fig_attn, use_container_width=True)
        st.caption("Attention shown is a mock heatmap. Replace with actual Transformer attention weights.")
