"""
dashboard.py
------------
Streamlit web dashboard for VoiceDataCurator.

Reads dataset_manifest.csv and renders:
  - Language distribution pie chart
  - Quality score histogram
  - Accepted vs Rejected summary cards
  - Average duration per language bar chart
  - Per-file quality report table (sortable/filterable)

Run with:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import glob
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceDataCurator Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3350;
        text-align: center;
    }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #60a5fa; }
    .metric-label { font-size: 0.9rem; color: #94a3b8; margin-top: 4px; }
    .accepted-val { color: #34d399; }
    .rejected-val { color: #f87171; }
    h1, h2, h3 { color: #e2e8f0; }
    .stDataFrame { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_manifest(path: str) -> pd.DataFrame:
    """Load and parse dataset_manifest.csv."""
    df = pd.read_csv(path)
    # Ensure expected columns have correct types
    numeric_cols = ["duration", "snr_db", "silence_ratio", "clipping_ratio", "quality_score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def find_manifest_files() -> list:
    """Search common locations for manifest CSV files."""
    patterns = [
        "./output/dataset_manifest.csv",
        "./logs/dry_run_dataset_manifest.csv",
        "./dataset_manifest.csv",
    ]
    found = []
    for p in patterns:
        matches = glob.glob(p)
        found.extend(matches)
    return found


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microphone.png", width=80)
    st.title("VoiceDataCurator")
    st.caption("Speech Dataset Quality Dashboard")
    st.divider()

    # Manifest file selection
    st.subheader("📂 Data Source")
    auto_found = find_manifest_files()

    if auto_found:
        selected_file = st.selectbox(
            "Select manifest file",
            options=auto_found,
            help="Auto-detected manifest CSV files",
        )
    else:
        selected_file = st.text_input(
            "Manifest CSV path",
            value="./output/dataset_manifest.csv",
            help="Path to dataset_manifest.csv",
        )

    st.divider()

    # Filters
    st.subheader("🎛️ Filters")
    show_accepted = st.checkbox("Show Accepted", value=True)
    show_rejected = st.checkbox("Show Rejected", value=True)

    quality_range = st.slider(
        "Quality Score Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
    )

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ─── Main Content ─────────────────────────────────────────────────────────────
st.title("🎙️ VoiceDataCurator — Dataset Dashboard")
st.caption("Real-time quality metrics for your multilingual speech dataset")

# Load data
if not Path(selected_file).exists():
    st.warning(
        f"⚠️ No manifest found at `{selected_file}`.\n\n"
        "Run the pipeline first:\n```\npython main.py --input ./data/raw\n```"
    )
    st.stop()

df_full = load_manifest(selected_file)

# Apply filters
status_filter = []
if show_accepted:
    status_filter.append("accepted")
if show_rejected:
    status_filter.append("rejected")

df = df_full[
    df_full["status"].isin(status_filter) &
    df_full["quality_score"].between(*quality_range)
].copy()

if df.empty:
    st.info("No files match the current filter settings.")
    st.stop()

# ─── Summary Cards ───────────────────────────────────────────────────────────
st.subheader("📊 Dataset Overview")

total = len(df_full)
accepted = (df_full["status"] == "accepted").sum()
rejected = (df_full["status"] == "rejected").sum()
accept_rate = accepted / total * 100 if total > 0 else 0
avg_duration = df_full["duration"].mean() if "duration" in df_full.columns else 0
avg_snr = df_full["snr_db"].mean() if "snr_db" in df_full.columns else 0
num_languages = df_full["language"].nunique() if "language" in df_full.columns else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{total}</div>
        <div class="metric-label">Total Files</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value accepted-val">{accepted}</div>
        <div class="metric-label">✅ Accepted</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value rejected-val">{rejected}</div>
        <div class="metric-label">❌ Rejected</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{accept_rate:.1f}%</div>
        <div class="metric-label">Accept Rate</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{avg_duration:.1f}s</div>
        <div class="metric-label">Avg Duration</div>
    </div>""", unsafe_allow_html=True)

with c6:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{num_languages}</div>
        <div class="metric-label">Languages</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Charts Row 1 ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🌍 Language Distribution")
    if "language" in df.columns:
        lang_counts = df["language"].value_counts().reset_index()
        lang_counts.columns = ["language", "count"]
        fig_pie = px.pie(
            lang_counts,
            names="language",
            values="count",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend=dict(font=dict(color="#94a3b8")),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("📈 Quality Score Distribution")
    if "quality_score" in df.columns:
        fig_hist = px.histogram(
            df,
            x="quality_score",
            color="status",
            nbins=30,
            color_discrete_map={"accepted": "#34d399", "rejected": "#f87171"},
            barmode="overlay",
            opacity=0.8,
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,17,23,0.6)",
            font_color="#e2e8f0",
            xaxis=dict(title="Quality Score", gridcolor="#2d3350"),
            yaxis=dict(title="Count", gridcolor="#2d3350"),
            legend=dict(font=dict(color="#94a3b8")),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ─── Charts Row 2 ─────────────────────────────────────────────────────────────
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("⏱️ Average Duration per Language")
    if "language" in df.columns and "duration" in df.columns:
        avg_dur = (
            df.groupby("language")["duration"]
            .mean()
            .reset_index()
            .sort_values("duration", ascending=False)
        )
        fig_bar = px.bar(
            avg_dur,
            x="language",
            y="duration",
            color="duration",
            color_continuous_scale="Blues",
            text_auto=".1f",
        )
        fig_bar.update_traces(textfont_color="#ffffff")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,17,23,0.6)",
            font_color="#e2e8f0",
            xaxis=dict(title="Language", gridcolor="#2d3350"),
            yaxis=dict(title="Avg Duration (s)", gridcolor="#2d3350"),
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with col_right2:
    st.subheader("🔊 SNR Distribution by Language")
    if "language" in df.columns and "snr_db" in df.columns:
        fig_box = px.box(
            df,
            x="language",
            y="snr_db",
            color="status",
            color_discrete_map={"accepted": "#34d399", "rejected": "#f87171"},
            points="outliers",
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,17,23,0.6)",
            font_color="#e2e8f0",
            xaxis=dict(title="Language", gridcolor="#2d3350"),
            yaxis=dict(title="SNR (dB)", gridcolor="#2d3350"),
            legend=dict(font=dict(color="#94a3b8")),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ─── Per-File Report Table ─────────────────────────────────────────────────────
st.subheader("📋 Per-File Quality Report")
st.caption(f"Showing {len(df)} of {total} total files")

# Column config for richer display
col_config = {
    "status": st.column_config.TextColumn("Status", width="small"),
    "quality_score": st.column_config.ProgressColumn(
        "Quality Score",
        min_value=0.0,
        max_value=1.0,
        format="%.3f",
    ),
    "snr_db": st.column_config.NumberColumn("SNR (dB)", format="%.1f"),
    "silence_ratio": st.column_config.NumberColumn("Silence %", format="%.1%"),
    "duration": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
}

display_cols = [
    c for c in [
        "filename", "language", "duration", "snr_db",
        "silence_ratio", "clipping_ratio", "quality_score",
        "status", "rejection_reason",
    ]
    if c in df.columns
]

st.dataframe(
    df[display_cols].sort_values("quality_score", ascending=False),
    use_container_width=True,
    height=400,
    column_config=col_config,
)

# ─── Export ──────────────────────────────────────────────────────────────────
st.divider()
col_dl1, col_dl2, _ = st.columns([1, 1, 4])
with col_dl1:
    csv_data = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv_data,
        file_name="filtered_manifest.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col_dl2:
    accepted_df = df_full[df_full["status"] == "accepted"]
    csv_accepted = accepted_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="✅ Download Accepted Only",
        data=csv_accepted,
        file_name="accepted_manifest.csv",
        mime="text/csv",
        use_container_width=True,
    )
