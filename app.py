"""
Streamlit ABSA Dashboard
==========================
Interactive dashboard for Aspect-Based Sentiment Analysis
on fintech lending apps (Google Play Store reviews).

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from preprocess import preprocess_text
from config import APPS, ASPECTS, LANG, COUNTRY, DATA_PROCESSED, MODELS_DIR

# ── PAGE CONFIG ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="ABSA Fintech Lending Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .metric-card h3 {
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 0.3rem;
    }

    .metric-card h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .metric-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    }

    .metric-trust {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }

    .metric-service {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        box-shadow: 0 4px 15px rgba(67, 233, 123, 0.3);
    }

    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .sentiment-positive { color: #43e97b; font-weight: 600; }
    .sentiment-negative { color: #f5576c; font-weight: 600; }
    .sentiment-neutral { color: #ffa726; font-weight: 600; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    div[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── SENTIMENT COLORS ──────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "Positive": "#43e97b",
    "Negative": "#f5576c",
    "Neutral": "#ffa726",
}

ASPECT_COLORS = {
    "risk": "#f5576c",
    "trust": "#4facfe",
    "service": "#43e97b",
}

ASPECT_LABELS = {
    "risk": "⚠️ Risk (Risiko)",
    "trust": "🛡️ Trust (Kepercayaan)",
    "service": "⚙️ Service Quality (Kualitas Layanan)",
}


# ── LOAD MODEL (CACHED) ──────────────────────────────────────────────
@st.cache_resource
def load_predictor(model_dir: str):
    from inference import ABSAPredictor
    return ABSAPredictor(model_dir)


# ── SCRAPING FUNCTION ─────────────────────────────────────────────────
def scrape_reviews(app_id: str, count: int = 200):
    """Scrape reviews from Google Play and preprocess them."""
    from google_play_scraper import Sort, reviews

    all_reviews = []
    batch_size = 200
    continuation_token = None

    with st.spinner(f"Scraping {count} reviews ..."):
        while len(all_reviews) < count:
            result, continuation_token = reviews(
                app_id, lang=LANG, country=COUNTRY,
                sort=Sort.NEWEST,
                count=min(batch_size, count - len(all_reviews)),
                continuation_token=continuation_token,
            )
            if not result:
                break
            all_reviews.extend(result)
            if continuation_token is None:
                break

    rows = []
    for r in all_reviews:
        if not r.get("content"):
            continue
        rows.append({
            "rating": r["score"],
            "review_text_raw": r["content"],
            "review_date": r["at"].strftime("%Y-%m-%d") if r.get("at") else "",
        })

    df = pd.DataFrame(rows)
    # Preprocess
    df["review_text"] = df["review_text_raw"].apply(preprocess_text)
    df = df[df["review_text"].str.split().str.len() >= 3].reset_index(drop=True)
    df["review_id"] = range(1, len(df) + 1)
    return df


# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Konfigurasi")

    # App selection
    app_options = {f"{name} ({app_id})": app_id for name, app_id in APPS.items()}
    app_options["Custom App ID"] = "custom"

    selected = st.selectbox("📱 Pilih Aplikasi", options=list(app_options.keys()))

    if app_options[selected] == "custom":
        app_id = st.text_input("Masukkan App ID", placeholder="com.example.app")
    else:
        app_id = app_options[selected]

    review_count = st.slider("📝 Jumlah Review", min_value=50, max_value=500, value=200, step=50)

    # Model selection
    model_options = {}
    for name in ["baseline", "lora", "retrained"]:
        model_path = MODELS_DIR / name / "model"
        if model_path.exists():
            model_options[f"IndoBERT ({name})"] = str(model_path)

    if not model_options:
        st.warning("⚠️ Belum ada model yang di-train. Jalankan training dulu.")
        selected_model = None
    else:
        selected_model_name = st.selectbox("🤖 Pilih Model", list(model_options.keys()))
        selected_model = model_options[selected_model_name]

    analyze_btn = st.button("🚀 Analyze!", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📖 Tentang")
    st.markdown(
        "Dashboard ini menganalisis sentimen ulasan pengguna "
        "pada aspek **Risk**, **Trust**, dan **Service Quality** "
        "menggunakan model IndoBERT dengan pendekatan ABSA."
    )

# ── HEADER ────────────────────────────────────────────────────────────
st.markdown('<p class="header-gradient">📊 ABSA Fintech Lending Dashboard</p>', unsafe_allow_html=True)
st.markdown("Analisis sentimen ulasan Google Play Store berdasarkan aspek **Risk**, **Trust**, dan **Service Quality**.")

# ── MAIN LOGIC ────────────────────────────────────────────────────────
if analyze_btn and app_id and app_id != "custom" and selected_model:
    # Load model
    predictor = load_predictor(selected_model)

    # Scrape reviews
    df = scrape_reviews(app_id, review_count)

    if df.empty:
        st.error("Tidak ada review yang ditemukan. Coba app ID lain.")
        st.stop()

    # Run ABSA inference
    with st.spinner("Menganalisis sentimen ..."):
        results = predictor.predict(df["review_text"].tolist())

    # Build results DataFrame
    for i, result in enumerate(results):
        for aspect in ASPECTS:
            if result[aspect]:
                df.loc[i, f"{aspect}_sentiment"] = result[aspect]["sentiment"]
                df.loc[i, f"{aspect}_confidence"] = result[aspect]["confidence"]

    st.success(f"✅ Berhasil menganalisis {len(df)} review!")

    # ── OVERVIEW CARDS ────────────────────────────────────────────────
    st.markdown("## 📋 Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Reviews</h3>
            <h1>{len(df)}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_rating = df["rating"].mean()
        st.markdown(f"""
        <div class="metric-card metric-risk">
            <h3>Rata-rata Rating</h3>
            <h1>⭐ {avg_rating:.1f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        pos_pct = 0
        for a in ASPECTS:
            col_name = f"{a}_sentiment"
            if col_name in df.columns:
                pos_pct += (df[col_name] == "Positive").sum()
        total_aspects = sum((f"{a}_sentiment" in df.columns and df[f"{a}_sentiment"].notna()).sum() for a in ASPECTS)
        pos_pct = pos_pct / max(total_aspects, 1) * 100

        st.markdown(f"""
        <div class="metric-card metric-trust">
            <h3>Sentimen Positif</h3>
            <h1>{pos_pct:.0f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        neg_pct = 0
        for a in ASPECTS:
            col_name = f"{a}_sentiment"
            if col_name in df.columns:
                neg_pct += (df[col_name] == "Negative").sum()
        neg_pct = neg_pct / max(total_aspects, 1) * 100

        st.markdown(f"""
        <div class="metric-card metric-service">
            <h3>Sentimen Negatif</h3>
            <h1>{neg_pct:.0f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── SENTIMENT DISTRIBUTION PER ASPECT ─────────────────────────────
    st.markdown("## 📊 Distribusi Sentimen per Aspek")
    chart_cols = st.columns(3)

    for idx, aspect in enumerate(ASPECTS):
        col_name = f"{aspect}_sentiment"
        if col_name not in df.columns:
            continue

        with chart_cols[idx]:
            counts = df[col_name].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                title=ASPECT_LABELS[aspect],
                color=counts.index,
                color_discrete_map=SENTIMENT_COLORS,
                hole=0.4,
            )
            fig.update_layout(
                font=dict(family="Inter"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=50, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── STACKED BAR CHART ─────────────────────────────────────────────
    st.markdown("## 📈 Perbandingan Sentimen Antar Aspek")

    bar_data = []
    for aspect in ASPECTS:
        col_name = f"{aspect}_sentiment"
        if col_name not in df.columns:
            continue
        for sentiment in ["Positive", "Neutral", "Negative"]:
            bar_data.append({
                "Aspect": ASPECT_LABELS[aspect],
                "Sentiment": sentiment,
                "Count": int((df[col_name] == sentiment).sum()),
            })

    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        fig = px.bar(
            bar_df, x="Aspect", y="Count", color="Sentiment",
            barmode="group",
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_layout(
            font=dict(family="Inter"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="Jumlah Review",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── RATING DISTRIBUTION ───────────────────────────────────────────
    st.markdown("## ⭐ Distribusi Rating")
    rating_counts = df["rating"].value_counts().sort_index()
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={"x": "Rating", "y": "Jumlah"},
        color=rating_counts.index,
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(
        font=dict(family="Inter"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── DETAIL TABLE ──────────────────────────────────────────────────
    st.markdown("## 📝 Detail Review")

    # Filter controls
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        filter_aspect = st.selectbox(
            "Filter Aspek",
            ["Semua"] + [ASPECT_LABELS[a] for a in ASPECTS],
        )
    with filter_col2:
        filter_sentiment = st.selectbox(
            "Filter Sentimen",
            ["Semua", "Positive", "Negative", "Neutral"],
        )

    display_df = df.copy()

    # Apply filters
    if filter_aspect != "Semua":
        aspect_key = [k for k, v in ASPECT_LABELS.items() if v == filter_aspect][0]
        col_name = f"{aspect_key}_sentiment"
        if col_name in display_df.columns:
            if filter_sentiment != "Semua":
                display_df = display_df[display_df[col_name] == filter_sentiment]
            else:
                display_df = display_df[display_df[col_name].notna()]

    elif filter_sentiment != "Semua":
        mask = pd.Series([False] * len(display_df))
        for a in ASPECTS:
            col_name = f"{a}_sentiment"
            if col_name in display_df.columns:
                mask |= display_df[col_name] == filter_sentiment
        display_df = display_df[mask]

    # Select display columns
    show_cols = ["review_id", "rating", "review_date", "review_text"]
    for a in ASPECTS:
        sent_col = f"{a}_sentiment"
        conf_col = f"{a}_confidence"
        if sent_col in display_df.columns:
            show_cols.append(sent_col)
        if conf_col in display_df.columns:
            show_cols.append(conf_col)

    available_cols = [c for c in show_cols if c in display_df.columns]
    st.dataframe(
        display_df[available_cols].head(100),
        use_container_width=True,
        height=400,
    )

    # ── SAMPLE REVIEWS PER ASPECT ─────────────────────────────────────
    st.markdown("## 🔍 Contoh Review per Aspek")

    for aspect in ASPECTS:
        col_name = f"{aspect}_sentiment"
        if col_name not in df.columns:
            continue

        with st.expander(ASPECT_LABELS[aspect], expanded=False):
            for sentiment in ["Positive", "Negative", "Neutral"]:
                subset = df[df[col_name] == sentiment]
                if subset.empty:
                    continue

                color = SENTIMENT_COLORS[sentiment]
                st.markdown(f"**<span style='color:{color}'>{sentiment}</span>** ({len(subset)} reviews):", unsafe_allow_html=True)

                samples = subset.head(3)
                for _, row in samples.iterrows():
                    st.markdown(f"> ⭐{row['rating']} — _{row['review_text'][:200]}_")
                st.markdown("")

elif not analyze_btn:
    # Show placeholder
    st.markdown("---")
    st.info("👈 Pilih aplikasi dan klik **Analyze!** di sidebar untuk memulai analisis.")

    # Show model comparison if available
    eval_path = DATA_PROCESSED / "evaluation" / "evaluation_summary.json"
    if eval_path.exists():
        import json
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        st.markdown("## 🏆 Model Performance Comparison")

        comparison_data = []
        for name in ["baseline", "lora", "retrained"]:
            if name in eval_data:
                d = eval_data[name]
                comparison_data.append({
                    "Model": name.capitalize(),
                    "Accuracy": f"{d['accuracy']:.4f}",
                    "F1 Macro": f"{d['f1_macro']:.4f}",
                    "F1 Weighted": f"{d['f1_weighted']:.4f}",
                    "Trainable Params": f"{d.get('trainable_params', 'N/A'):,}" if isinstance(d.get('trainable_params'), int) else "N/A",
                    "Train Time (s)": d.get("training_time_seconds", "N/A"),
                })

        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
