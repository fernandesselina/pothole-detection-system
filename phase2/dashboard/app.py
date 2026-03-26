import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import folium
from streamlit_folium import st_folium
from PIL import Image

# ---------- PATHS & HELPERS ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "reports.json"
THEME = {
    "bg": "#0b1220",
    "panel": "#111a2b",
    "panel_alt": "#172338",
    "text": "#e5eef8",
    "muted": "#9fb0c3",
    "grid": "#2a3a52",
    "accent": "#38bdf8",
    "accent_light": "#7dd3fc",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "minor": "#60a5fa",
    "moderate": "#f59e0b",
    "severe": "#f87171",
}
SEVERITY_ORDER = ["Minor", "Moderate", "Severe"]
SEVERITY_WEIGHTS = {"Minor": 1, "Moderate": 2, "Severe": 3, "Unknown": 0}
SEVERITY_PALETTE = [THEME["minor"], THEME["moderate"], THEME["severe"]]
SEVERITY_MAP = dict(zip(SEVERITY_ORDER, SEVERITY_PALETTE))
STATUS_ORDER = ["Pending", "Resolved"]
STATUS_MAP = {"Pending": THEME["danger"], "Resolved": THEME["success"]}
REPORT_COLUMNS = ["report_id", "image_path", "latitude", "longitude", "confidence", "timestamp", "status", "severity"]


def load_reports():
    return json.loads(DATA_PATH.read_text()) if DATA_PATH.exists() else []


def save_reports(updated_reports):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(updated_reports, indent=2))


def classify_severity_from_ratio(area_ratio):
    # Tuned as image-area ratios so it adapts to different input resolutions.
    if area_ratio < 0.02:
        return "Minor"
    if area_ratio < 0.08:
        return "Moderate"
    return "Severe"


def get_detection_severity(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    img_h, img_w = result.orig_shape
    image_area = max(img_h * img_w, 1)
    severity_rank = {"Minor": 1, "Moderate": 2, "Severe": 3}
    max_rank = 0
    final_severity = "Minor"

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = max((x2 - x1), 0) * max((y2 - y1), 0)
        severity = classify_severity_from_ratio(area / image_area)
        if severity_rank[severity] > max_rank:
            max_rank = severity_rank[severity]
            final_severity = severity

    return final_severity


def mark_report_resolved(report_id):
    reports, updated = load_reports(), False
    for report in reports:
        if str(report.get("report_id")) == str(report_id):
            updated = report.get("status") != "Resolved"
            report["status"] = "Resolved"
            break
    if updated:
        save_reports(reports)
    return updated


def reports_df(reports):
    df = pd.DataFrame(reports) if reports else pd.DataFrame(columns=REPORT_COLUMNS)
    for col, default in {"status": "Pending", "severity": "Unknown", "timestamp": pd.Timestamp.now().isoformat()}.items():
        if col not in df.columns:
            df[col] = default
    df["status"] = df["status"].astype(str).str.title()
    df["severity"] = df["severity"].astype(str).str.title()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    num_potholes = df["num_potholes"] if "num_potholes" in df.columns else pd.Series(1, index=df.index)
    df["num_potholes"] = pd.to_numeric(num_potholes, errors="coerce").fillna(1)
    return df


def apply_global_theme():
    sns.set_theme(
        style="whitegrid",
        palette=[THEME["accent"], THEME["moderate"], THEME["danger"], THEME["success"]],
        rc={
            "axes.facecolor": THEME["panel"],
            "figure.facecolor": THEME["panel"],
            "axes.edgecolor": THEME["grid"],
            "axes.labelcolor": THEME["text"],
            "xtick.color": THEME["muted"],
            "ytick.color": THEME["muted"],
            "grid.color": THEME["grid"],
            "text.color": THEME["text"],
        },
    )

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, {THEME["bg"]} 0%, #0f172a 100%);
            color: {THEME["text"]};
        }}
        .stApp [data-testid="stAppViewContainer"] {{
            color: {THEME["text"]};
        }}
        .stApp [data-testid="stAppViewContainer"] p,
        .stApp [data-testid="stAppViewContainer"] span,
        .stApp [data-testid="stAppViewContainer"] label,
        .stApp [data-testid="stAppViewContainer"] div {{
            color: {THEME["text"]};
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }}
        section[data-testid="stSidebar"] * {{
            color: {THEME["text"]} !important;
        }}
        .block-container {{
            color: {THEME["text"]};
        }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(135deg, {THEME["panel"]} 0%, {THEME["panel_alt"]} 100%);
            border: 1px solid {THEME["grid"]};
            border-radius: 16px;
            padding: 14px;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28);
            color: {THEME["text"]} !important;
        }}
        div[data-testid="stMetric"] * {{
            color: {THEME["text"]} !important;
        }}
        div[data-testid="stDataFrame"], div[data-testid="stAlert"] {{
            border-radius: 16px;
            overflow: hidden;
            color: {THEME["text"]} !important;
        }}
        div[data-testid="stDataFrame"] *,
        div[data-testid="stAlert"] *,
        div[data-testid="stMarkdownContainer"] *,
        div[data-testid="stText"] *,
        div[data-testid="stCaptionContainer"] *,
        div[data-testid="stTable"] *,
        div[data-testid="stExpander"] * {{
            color: {THEME["text"]} !important;
        }}
        div[data-testid="stAlert"] {{
            background: {THEME["panel"]} !important;
            border: 1px solid {THEME["grid"]} !important;
        }}
        div[data-testid="stFileUploader"] {{
            background: {THEME["panel"]};
            border: 1px solid {THEME["grid"]};
            border-radius: 14px;
            padding: 0.5rem;
        }}
        div[data-testid="stFileUploader"] * {{
            color: {THEME["text"]} !important;
        }}
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"] {{
            background: {THEME["panel"]} !important;
            border-color: {THEME["grid"]} !important;
            color: {THEME["text"]} !important;
        }}
        div[data-baseweb="select"] span,
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] input,
        div[data-testid="stMultiSelect"] span {{
            color: {THEME["text"]} !important;
        }}
        div[data-baseweb="select"] svg,
        div[data-baseweb="input"] svg {{
            fill: {THEME["text"]} !important;
        }}
        input, textarea {{
            color: {THEME["text"]} !important;
            -webkit-text-fill-color: {THEME["text"]} !important;
        }}
        .stButton > button {{
            background: linear-gradient(135deg, {THEME["accent"]} 0%, #0ea5e9 100%);
            color: #08111f;
            border: none;
            border-radius: 10px;
            padding: 0.55rem 1.1rem;
            font-weight: 700;
        }}
        .stButton > button:hover {{
            background: linear-gradient(135deg, #7dd3fc 0%, {THEME["accent"]} 100%);
            color: #08111f;
        }}
        .stSelectbox label, .stMultiSelect label, .stNumberInput label, .stFileUploader label {{
            color: {THEME["text"]} !important;
            font-weight: 600;
        }}
        h1, h2, h3 {{
            color: {THEME["text"]};
            letter-spacing: 0.01em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_axis(ax, title=None):
    ax.set_facecolor(THEME["panel"])
    for spine in ax.spines.values():
        spine.set_color(THEME["grid"])
    ax.tick_params(colors=THEME["muted"])
    ax.xaxis.label.set_color(THEME["text"])
    ax.yaxis.label.set_color(THEME["text"])
    ax.grid(color=THEME["grid"], alpha=0.4, linestyle="--", linewidth=0.7)
    if title:
        ax.set_title(title, color=THEME["text"], fontsize=14, fontweight="bold", pad=12)


def annotate_bars(ax, fmt="{:.0f}", suffix=""):
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{fmt.format(height)}{suffix}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 5),
            textcoords="offset points",
            color=THEME["text"],
        )


def location_summary(df):
    if df.dropna(subset=["latitude", "longitude"]).empty:
        return pd.DataFrame()
    data = df.dropna(subset=["latitude", "longitude"]).copy()
    data["location_label"] = data["latitude"].round(3).astype(str) + ", " + data["longitude"].round(3).astype(str)
    return data.groupby("location_label", as_index=False)["weighted_impact"].sum().nlargest(6, "weighted_impact").sort_values("weighted_impact")


# ---------- CONFIG ----------
st.set_page_config(
    page_title="Pothole Management Dashboard",
    layout="wide"
)
apply_global_theme()

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Choose Interface",
    ["Pothole Detection", "Dashboard", "Analytics"]
)

if page == "Dashboard":

    st.title("🛣️ Smart Pothole Detection – Phase 2 Dashboard")

    df = reports_df(load_reports())

    # ---------- OVERVIEW METRICS ----------
    st.subheader("📊 System Overview")

    for col, (label, value) in zip(st.columns(3), {
        "Total Reports": len(df),
        "Pending Reports": len(df[df["status"].str.lower() == "pending"]),
        "Resolved Reports": len(df[df["status"].str.lower() == "resolved"]),
    }.items()):
        col.metric(label, value)

    # ---------- TABLE ----------
    st.subheader("📋 Pothole Reports")

    st.dataframe(df, use_container_width=True)

    # ---------- RESOLVE POTHOLE ----------
    st.subheader("Resolve Pothole")
    if not df.empty and "report_id" in df.columns:
        selected_report_id = st.selectbox("Select report_id", options=df["report_id"].tolist())

        if st.button("Mark as Resolved"):
            if mark_report_resolved(selected_report_id):
                st.success(f"Report {selected_report_id} marked as Resolved.")
                st.rerun()
            else:
                st.info(f"Report {selected_report_id} is already Resolved or was not found.")
    else:
        st.info("No reports available.")

    # ---------- MAP VIEW ----------
    st.subheader("🗺️ Pothole Location Map")

    if not df.empty:
        pothole_map = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=11)

        for _, row in df.iterrows():
            color = "red" if str(row["status"]).lower() == "pending" else "green"

            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=f"<b>Report ID:</b> {row['report_id']}<br><b>Status:</b> {row['status']}<br><b>Confidence:</b> {row['confidence']}<br><b>Timestamp:</b> {row['timestamp']}",
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(pothole_map)

        st_folium(pothole_map, width=1000, height=500)
    else:
        st.info("No location data available.")


    st.write("Looking for data at:", DATA_PATH.resolve())

elif page == "Pothole Detection":

    st.title("📸 Pothole Detection")

    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=0.0, format="%.6f")
    with col2:
        longitude = st.number_input("Longitude", value=0.0, format="%.6f")

    uploaded_file = st.file_uploader(
        "Upload Road Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Potholes"):

            from ultralytics import YOLO
            import tempfile

            model_candidates = [
    project_root / "best.pt",  # NEW (your moved model)
    project_root / "runs" / "detect" / "train4" / "weights" / "best.pt",
    project_root / "yolov8n.pt",
]

            model = YOLO(str(model_path))
            st.caption(f"Loaded model: {model_path}")

    # save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name

            results = model(temp_path, conf=0.25)

    # show detection result
            annotated_frame = results[0].plot()
            st.image(annotated_frame, caption="Detection Result", use_container_width=True)
            if len(results[0].boxes) == 0:
                st.warning("No detections found. This usually means the model is not trained for potholes, or confidence is too high.")
            else:
                st.success(f"Detected {len(results[0].boxes)} pothole(s).")
                severity = get_detection_severity(results[0])
                confidence = float(results[0].boxes.conf.max().item()) if results[0].boxes.conf is not None else 0.0
                existing_reports = load_reports()

                next_id = max((int(r.get("report_id", 0)) for r in existing_reports if str(r.get("report_id", "")).isdigit()), default=0) + 1

                existing_reports.append({
                    "report_id": next_id,
                    "image_path": uploaded_file.name,
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "confidence": round(confidence, 4),
                    "timestamp": datetime.now().isoformat(),
                    "status": "Pending",
                    "severity": severity or "Minor",
                })
                save_reports(existing_reports)
                st.info(f"Report #{next_id} saved with severity: {severity or 'Minor'}")
            
elif page == "Analytics":
    st.title("Pothole Analytics")

    reports = load_reports()
    if not reports:
        st.info("No analytics data available yet.")
    else:
        df = reports_df(reports)
        st.subheader("Analysis Controls")
        c1, c2 = st.columns(2)
        selected_statuses = c1.multiselect("Status Filter", options=STATUS_ORDER, default=STATUS_ORDER)
        selected_severities = c2.multiselect("Severity Filter", options=SEVERITY_ORDER, default=SEVERITY_ORDER)
        filtered_df = df[df["status"].isin(selected_statuses) & df["severity"].isin(selected_severities)].copy()

        if filtered_df.empty:
            st.warning("No records match the selected filters.")
            st.stop()

        status_counts = filtered_df["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
        sev_counts = filtered_df["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
        filtered_df["severity_score"] = filtered_df["severity"].map(SEVERITY_WEIGHTS).fillna(0)
        filtered_df = filtered_df.sort_values("timestamp", na_position="last").reset_index(drop=True)
        filtered_df["report_index"] = range(1, len(filtered_df) + 1)
        filtered_df["confidence_pct"] = pd.to_numeric(filtered_df["confidence"], errors="coerce").fillna(0) * 100
        filtered_df["weighted_impact"] = filtered_df["severity_score"] * filtered_df["confidence_pct"]

        total_reports = len(filtered_df)
        for col, (label, value) in zip(st.columns(4), {
            "Resolution Rate": f"{(status_counts['Resolved'] / total_reports * 100) if total_reports else 0:.1f}%",
            "Average Confidence": f"{filtered_df['confidence_pct'].mean():.1f}%",
            "Average Severity Score": f"{(filtered_df['severity_score'].replace(0, pd.NA).dropna().mean() or 0):.2f} / 3",
            "Hotspot Impact Score": f"{filtered_df['weighted_impact'].mean() if total_reports else 0:.1f}",
        }.items()):
            col.metric(label, value)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Report Status Share")
            fig1, ax1 = plt.subplots(figsize=(5.5, 5.5))
            fig1.patch.set_facecolor(THEME["panel"])
            wedges, texts, autotexts = ax1.pie(
                status_counts.values,
                labels=status_counts.index,
                autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
                startangle=90,
                colors=[STATUS_MAP[k] for k in status_counts.index],
                wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
                pctdistance=0.78,
            )
            for text in texts + autotexts:
                text.set_fontsize(10)
                text.set_color(THEME["text"])
            ax1.text(0, 0, f"{total_reports}\nReports", ha="center", va="center", fontsize=16, fontweight="bold", color=THEME["text"])
            ax1.axis("equal")
            st.pyplot(fig1)

        with chart_col2:
            st.subheader("Severity Distribution")
            sev_df = sev_counts.rename_axis("severity").reset_index(name="count")
            fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
            sns.barplot(data=sev_df, x="severity", y="count", palette=SEVERITY_PALETTE, ax=ax2)
            annotate_bars(ax2)
            ax2.set_xlabel("Severity")
            ax2.set_ylabel("Count")
            style_axis(ax2, "Severity Distribution")
            st.pyplot(fig2)

        st.subheader("High-Priority Reports")
        priority_df = (
            filtered_df.nlargest(min(len(filtered_df), 8), "weighted_impact")
            .sort_values("weighted_impact", ascending=True)
            .copy()
        )
        priority_df["report_label"] = priority_df["report_id"].astype(str).radd("Report #")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        sns.barplot(
            data=priority_df,
            x="weighted_impact",
            y="report_label",
            hue="severity",
            dodge=False,
            palette=SEVERITY_MAP,
            ax=ax3,
        )
        ax3.set_xlabel("Priority Score")
        ax3.set_ylabel("Report")
        style_axis(ax3, "High-Priority Reports")
        st.pyplot(fig3)

        st.subheader("Operational Pattern Analysis")
        ops_col1, ops_col2 = st.columns(2)

        with ops_col1:
            st.subheader("Confidence vs Severity")
            fig4, ax4 = plt.subplots(figsize=(7, 5.5))
            sns.scatterplot(
                data=filtered_df,
                x="report_index",
                y="confidence_pct",
                hue="status",
                size="severity_score",
                sizes=(80, 320),
                palette=STATUS_MAP,
                alpha=0.75,
                ax=ax4,
            )
            ax4.set_xlabel("Report Sequence")
            ax4.set_ylabel("Confidence (%)")
            ax4.set_ylim(0, 105)
            style_axis(ax4, "Confidence vs Severity")
            st.pyplot(fig4)

        with ops_col2:
            st.subheader("Average Confidence by Severity")
            confidence_summary = filtered_df.groupby("severity", as_index=False)["confidence_pct"].mean().set_index("severity").reindex(SEVERITY_ORDER).dropna().reset_index()
            fig5, ax5 = plt.subplots(figsize=(8, 5.5))
            sns.barplot(data=confidence_summary, x="severity", y="confidence_pct", palette=SEVERITY_PALETTE, ax=ax5)
            annotate_bars(ax5, "{:.1f}", "%")
            ax5.set_xlabel("Severity")
            ax5.set_ylabel("Average Confidence (%)")
            style_axis(ax5, "Average Confidence by Severity")
            st.pyplot(fig5)

        deep_col1, deep_col2 = st.columns(2)

        with deep_col1:
            st.subheader("Severity by Status Matrix")
            status_severity_matrix = filtered_df.groupby(["severity", "status"]).size().unstack(fill_value=0).reindex(index=SEVERITY_ORDER, columns=STATUS_ORDER, fill_value=0)
            fig6, ax6 = plt.subplots(figsize=(7, 5.5))
            sns.heatmap(
                status_severity_matrix,
                annot=True,
                fmt="d",
                cmap=sns.light_palette(THEME["accent"], as_cmap=True),
                linewidths=0.5,
                cbar_kws={"label": "Reports"},
                ax=ax6,
            )
            ax6.set_xlabel("Status")
            ax6.set_ylabel("Severity")
            style_axis(ax6, "Severity by Status Matrix")
            st.pyplot(fig6)

        with deep_col2:
            st.subheader("Geographic Risk Scatter")
            geo_df = filtered_df.dropna(subset=["latitude", "longitude"]).copy()
            if geo_df.empty:
                st.info("No geographic coordinates available.")
            else:
                fig7, ax7 = plt.subplots(figsize=(7, 5.5))
                sns.scatterplot(
                    data=geo_df,
                    x="longitude",
                    y="latitude",
                    hue="severity",
                    size="confidence_pct",
                    sizes=(70, 360),
                    palette=SEVERITY_MAP,
                    alpha=0.8,
                    ax=ax7,
                )
                ax7.set_xlabel("Longitude")
                ax7.set_ylabel("Latitude")
                style_axis(ax7, "Geographic Risk Scatter")
                st.pyplot(fig7)

        st.subheader("Priority Summary View")
        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            if (loc_summary := location_summary(filtered_df)).empty:
                st.info("No location-based summary available.")
            else:
                fig8, ax8 = plt.subplots(figsize=(8, 5.5))
                sns.barplot(data=loc_summary, x="weighted_impact", y="location_label", color=THEME["accent_light"], ax=ax8)
                ax8.set_xlabel("Total Priority Score")
                ax8.set_ylabel("Location")
                style_axis(ax8, "Top Risk Locations")
                st.pyplot(fig8)

        with summary_col2:
            st.subheader("Confidence Distribution by Severity")
            confidence_df = filtered_df[filtered_df["severity_score"] > 0].copy()
            if confidence_df.empty:
                st.info("No severity-tagged confidence data available.")
            else:
                fig9, ax9 = plt.subplots(figsize=(8, 5.5))
                sns.violinplot(
                    data=confidence_df,
                    x="severity",
                    y="confidence_pct",
                    order=SEVERITY_ORDER,
                    palette=SEVERITY_PALETTE,
                    inner="quartile",
                    cut=0,
                    ax=ax9,
                )
                ax9.set_xlabel("Severity")
                ax9.set_ylabel("Confidence (%)")
                ax9.set_ylim(0, 105)
                style_axis(ax9, "Confidence Distribution by Severity")
                st.pyplot(fig9)
