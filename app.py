"""
Streamlit Web Dashboard for Aspect-Based Sentiment Analysis (ABSA).
Provides live single-review analysis, batch CSV processing, interactive analytics,
model performance benchmarks, and academic viva preparation resources.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Configure page layout and title
st.set_page_config(
    page_title="Aspect-Based Sentiment Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern card styling, badges, and responsive tables
st.markdown("""
<style>
    /* Metric Card styling */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #eef2f6;
        margin-bottom: 16px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 4px;
    }
    
    /* Sentiment badges */
    .badge-positive {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-negative {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #f1f5f9;
        color: #475569;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-mixed {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Aspect cards */
    .aspect-box {
        background-color: #ffffff;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# Imports after configuration
from config import (
    METRICS_JSON_PATH,
    DEFAULT_DATASET_PATH,
    SENTIMENT_COLORS
)
from models.sentiment_model import HybridSentimentAnalyzer
from utils.dataset_loader import load_dataset
from utils.visualization import (
    plot_sentiment_distribution,
    plot_aspect_frequencies,
    plot_aspect_sentiment_matrix,
    plot_model_comparison_metrics,
    plot_confusion_matrix,
    plot_product_sentiment_distribution
)


@st.cache_resource
def get_analyzer():
    """Cache analyzer instance to avoid repeated disk reads."""
    return HybridSentimentAnalyzer()


@st.cache_data
def get_metrics_data():
    """Load pre-evaluated metrics JSON."""
    if METRICS_JSON_PATH.exists():
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_dataset():
    """Load benchmark dataset."""
    return load_dataset()


analyzer = get_analyzer()
metrics = get_metrics_data()

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.title("NLP ABSA System")
    st.caption("Aspect-Based Sentiment Analysis of Product Reviews")
    st.markdown("---")

    page = st.radio(
        "Navigation Menu",
        [
            "🔍 Single Review Analysis",
            "📁 Batch CSV Analysis",
            "📊 Visual Analytics",
            "🧠 Model Performance",
            "📘 University Report & Viva"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Settings")
    selected_model = st.selectbox(
        "Sentiment Classification Engine",
        ["logistic_regression", "naive_bayes", "vader"],
        format_func=lambda x: {
            "logistic_regression": "Logistic Regression (TF-IDF)",
            "naive_bayes": "Naive Bayes (MultinomialNB)",
            "vader": "VADER (Rule-based Lexicon)"
        }[x]
    )

    st.markdown("---")
    st.markdown("**Project Authors**:")
    st.caption("College Mini-Project | NLP & Machine Learning")


# Helper for rendering sentiment badges
def render_badge(sentiment: str) -> str:
    s = sentiment.lower()
    if s == "positive":
        return f'<span class="badge-positive">✓ {sentiment}</span>'
    elif s == "negative":
        return f'<span class="badge-negative">✗ {sentiment}</span>'
    elif s == "mixed":
        return f'<span class="badge-mixed">⚡ {sentiment}</span>'
    else:
        return f'<span class="badge-neutral">● {sentiment}</span>'


# =============================================================
# PAGE 1: SINGLE REVIEW ANALYSIS
# =============================================================
if page == "🔍 Single Review Analysis":
    st.title("🎯 Aspect-Based Sentiment Analysis")
    st.markdown(
        "Analyze customer reviews to extract individual product aspects (Camera, Battery, Display, etc.) "
        "and determine their granular sentiments independently."
    )

    # Preset Example Reviews
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
    sample_text = ""
    with col_ex1:
        if st.button("📱 Camera / Battery / Display", use_container_width=True):
            sample_text = "The phone camera is excellent, battery backup is poor, and the display quality is amazing."
    with col_ex2:
        if st.button("🎧 Audio / Customer Service", use_container_width=True):
            sample_text = "Sound quality is breathtaking with deep bass, however customer service refused my warranty claim."
    with col_ex3:
        if st.button("💻 Speed / Overpriced", use_container_width=True):
            sample_text = "The processor speed is blazing fast, but the price tag is completely overpriced and expensive."
    with col_ex4:
        if st.button("📦 Delivery / Hardware", use_container_width=True):
            sample_text = "Shipping was extremely fast and packaging was secure, but the device performance is very laggy."

    # User Input Text Area
    user_review = st.text_area(
        "Enter or Paste Customer Product Review:",
        value=sample_text if sample_text else "The phone camera is excellent, battery backup is poor, and the display quality is amazing.",
        height=110,
        placeholder="Type a product review here..."
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Aspects", type="primary", use_container_width=True)

    if user_review:
        with st.spinner("Executing NLP Preprocessing & Aspect-Based Sentiment Classification..."):
            result = analyzer.analyze_review_absa(user_review, model_name=selected_model)

        # Overview Header
        st.markdown("---")
        st.subheader("Analysis Summary")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Sentiment</div>
                <div class="metric-value">{render_badge(result["overall_sentiment"])}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Aspects Identified</div>
                <div class="metric-value">{len(result["aspects"])}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Confidence</div>
                <div class="metric-value">{result["overall_score"] * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Active Model</div>
                <div class="metric-value" style="font-size:1.1rem; padding-top:6px;">{selected_model.replace('_', ' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)

        # Aspect Breakdown Cards
        st.markdown("### 🧩 Granular Aspect Polarities")
        if result["aspects"]:
            cols = st.columns(min(len(result["aspects"]), 3))
            for i, aspect_data in enumerate(result["aspects"]):
                target_col = cols[i % len(cols)]
                with target_col:
                    sentiment = aspect_data["sentiment"]
                    conf = aspect_data["confidence"]
                    border_color = SENTIMENT_COLORS.get(sentiment, "#3b82f6")

                    st.markdown(f"""
                    <div class="aspect-box" style="border-left-color: {border_color};">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4 style="margin:0; color:#1e293b;">{aspect_data['aspect']}</h4>
                            {render_badge(sentiment)}
                        </div>
                        <p style="margin: 8px 0 4px 0; font-size: 0.9rem; color:#475569;">
                            <strong>Matched Expression:</strong> <code>{aspect_data['raw_term']}</code>
                        </p>
                        <p style="margin: 4px 0 8px 0; font-size: 0.85rem; color:#64748b;">
                            <strong>Isolated Context Clause:</strong><br>
                            <em>"{aspect_data['context_clause']}"</em>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"Estimated Class Probability: **{conf*100:.1f}%**")
                    st.progress(min(max(float(conf), 0.0), 1.0))
        else:
            st.info("No specific hardware/service aspect keywords detected in this review. The overall review was classified as a whole.")

        # Educational Preprocessing Stages Expander
        with st.expander("🔬 View NLP Preprocessing & Normalization Stages", expanded=False):
            stages = result.get("stages", {})
            st.markdown("Inspect how the raw text was normalized and converted through the NLP pipeline:")
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown("**1. Original Text:**")
                st.code(stages.get("original_text", ""), language="text")

                st.markdown("**2. Contractions Expanded:**")
                st.code(stages.get("expanded_contractions", ""), language="text")

                st.markdown("**3. Tokenization (NLTK):**")
                st.write(stages.get("raw_tokens", []))

            with p_col2:
                st.markdown("**4. Punctuation & Negation-Aware Stopword Removal:**")
                st.write(stages.get("tokens_no_stopwords", []))

                st.markdown("**5. POS-Tagged Lemmatization (WordNet):**")
                st.write(stages.get("lemmatized_tokens", []))

                st.markdown("**6. Final Normalized Representation (for TF-IDF):**")
                st.code(stages.get("final_processed_text", ""), language="text")


# =============================================================
# PAGE 2: BATCH CSV ANALYSIS
# =============================================================
elif page == "📁 Batch CSV Analysis":
    st.title("📁 Batch Aspect-Based Sentiment Analysis")
    st.markdown("Upload any CSV dataset containing customer reviews, or evaluate the built-in benchmark dataset.")

    data_source = st.radio(
        "Select Dataset Source:",
        ["Use Built-in Benchmark Dataset (Amazon Electronics Reviews)", "Upload Custom CSV File"],
        horizontal=True
    )

    batch_df = None
    if data_source == "Upload Custom CSV File":
        uploaded_file = st.file_uploader("Upload Product Reviews CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded CSV with {len(batch_df)} rows!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    else:
        batch_df = get_dataset()
        st.info(f"Loaded benchmark dataset containing {len(batch_df)} product reviews.")

    if batch_df is not None:
        # Detect text columns
        text_cols = [c for c in batch_df.columns if batch_df[c].dtype == object]
        default_col_idx = 0
        for i, c in enumerate(text_cols):
            if c.lower() in ["review_text", "review", "text", "comments", "body"]:
                default_col_idx = i
                break

        selected_text_col = st.selectbox("Select Review Text Column:", text_cols, index=default_col_idx)

        num_rows = st.slider("Select Number of Reviews to Analyze:", min_value=5, max_value=min(len(batch_df), 300), value=min(len(batch_df), 50), step=5)

        if st.button("⚡ Run Batch ABSA Pipeline", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            processed_rows = []
            aspect_frequency_map = {}
            aspect_sentiment_records = []

            subset_df = batch_df.head(num_rows).copy()

            for idx, text in enumerate(subset_df[selected_text_col]):
                res = analyzer.analyze_review_absa(str(text), model_name=selected_model)
                
                aspect_names = [a["aspect"] for a in res["aspects"]]
                aspect_polarities = [f"{a['aspect']}: {a['sentiment']}" for a in res["aspects"]]

                for a in res["aspects"]:
                    asp = a["aspect"]
                    sent = a["sentiment"]
                    aspect_frequency_map[asp] = aspect_frequency_map.get(asp, 0) + 1
                    aspect_sentiment_records.append({"Aspect": asp, "Sentiment": sent})

                processed_rows.append({
                    "Review Index": idx + 1,
                    "Review Text": str(text),
                    "Overall Sentiment": res["overall_sentiment"],
                    "Aspects Detected": ", ".join(aspect_names) if aspect_names else "None",
                    "Granular Polarities": " | ".join(aspect_polarities) if aspect_polarities else "None",
                    "Confidence": f"{res['overall_score']*100:.1f}%"
                })

                progress_bar.progress((idx + 1) / num_rows)
                status_text.text(f"Processing review {idx + 1} of {num_rows}...")

            status_text.text("Batch Processing Complete!")
            results_df = pd.DataFrame(processed_rows)

            st.markdown("---")
            st.subheader("Batch ABSA Results Table")
            
            # Interactive Filter
            filter_sentiment = st.multiselect(
                "Filter by Overall Sentiment:",
                ["All", "Positive", "Negative", "Neutral", "Mixed"],
                default=["All"]
            )

            display_df = results_df
            if "All" not in filter_sentiment and filter_sentiment:
                display_df = display_df[display_df["Overall Sentiment"].isin(filter_sentiment)]

            st.dataframe(display_df, use_container_width=True)

            # Download CSV
            csv_data = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Analyzed Results as CSV",
                data=csv_data,
                file_name="absa_analyzed_results.csv",
                mime="text/csv"
            )

            # Batch Visuals
            st.markdown("### 📊 Batch Analytics Overview")
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                st.plotly_chart(plot_sentiment_distribution(results_df, col="Overall Sentiment"), use_container_width=True)
            with vcol2:
                if aspect_frequency_map:
                    st.plotly_chart(plot_aspect_frequencies(aspect_frequency_map), use_container_width=True)


# =============================================================
# PAGE 3: VISUAL ANALYTICS
# =============================================================
elif page == "📊 Visual Analytics":
    st.title("📊 Dataset & Aspect Visual Analytics")
    st.markdown("Exploratory data analysis across customer reviews, aspect mentions, and product categories.")

    df = get_dataset()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Reviews", f"{len(df):,}")
    with c2:
        st.metric("Product Categories", df["product_category"].nunique() if "product_category" in df.columns else "N/A")
    with c3:
        st.metric("Unique Products", df["product_name"].nunique() if "product_name" in df.columns else "N/A")
    with c4:
        st.metric("Avg Review Rating", f"{df['rating'].mean():.2f} / 5.0" if "rating" in df.columns else "N/A")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_sentiment_distribution(df, col="sentiment"), use_container_width=True)
    with col2:
        if "product_category" in df.columns:
            st.plotly_chart(plot_product_sentiment_distribution(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Aspect-Wise Sentiment Distribution")
    
    # Compute Aspect Sentiment Matrix
    aspect_records = []
    if "aspect_sentiments" in df.columns:
        for val in df["aspect_sentiments"].dropna():
            try:
                aspect_dict = json.loads(val)
                for asp, sent in aspect_dict.items():
                    aspect_records.append({"Aspect": asp, "Sentiment": sent})
            except Exception:
                pass

    if aspect_records:
        asp_df = pd.DataFrame(aspect_records)
        asp_matrix = asp_df.groupby(["Aspect", "Sentiment"]).size().unstack(fill_value=0).reset_index()
        st.plotly_chart(plot_aspect_sentiment_matrix(asp_matrix), use_container_width=True)


# =============================================================
# PAGE 4: MODEL PERFORMANCE & BENCHMARKS
# =============================================================
elif page == "🧠 Model Performance":
    st.title("🧠 Empirical Model Evaluation & Benchmarks")
    st.markdown(
        "Quantitative comparison between **Logistic Regression (TF-IDF)**, **Multinomial Naive Bayes (TF-IDF)**, "
        "and **VADER (Rule-based Lexicon)** on an identical 80/20 stratified test split."
    )

    if metrics:
        summary = metrics["dataset_summary"]
        lr = metrics["logistic_regression"]
        nb = metrics["naive_bayes"]
        vd = metrics["vader"]
        asp = metrics["aspect_extraction"]

        # High-level Metrics Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Logistic Regression Acc", f"{lr['accuracy']*100:.2f}%")
        with m2:
            st.metric("Naive Bayes Acc", f"{nb['accuracy']*100:.2f}%")
        with m3:
            st.metric("VADER Baseline Acc", f"{vd['accuracy']*100:.2f}%")
        with m4:
            st.metric("Aspect Extraction F1", f"{asp['f1_score']*100:.2f}%")

        st.markdown("---")
        st.subheader("Model Comparison Metrics")
        st.plotly_chart(plot_model_comparison_metrics(metrics), use_container_width=True)

        st.markdown("---")
        st.subheader("Confusion Matrices (Test Set)")
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.plotly_chart(plot_confusion_matrix(lr["confusion_matrix"], summary["classes"], "Logistic Regression Confusion Matrix"), use_container_width=True)
        with cm_col2:
            st.plotly_chart(plot_confusion_matrix(nb["confusion_matrix"], summary["classes"], "Naive Bayes Confusion Matrix"), use_container_width=True)

        st.markdown("---")
        st.subheader("Aspect Extraction Rigorous Benchmark")
        st.markdown(
            f"""
            Aspect extraction performance evaluated against labeled multi-aspect ground truth test cases:
            - **Aspect Precision**: **{asp['precision']*100:.2f}%**
            - **Aspect Recall**: **{asp['recall']*100:.2f}%**
            - **Aspect F1-Score**: **{asp['f1_score']*100:.2f}%**
            - **True Positives**: {asp['true_positives']} | **False Positives**: {asp['false_positives']} | **False Negatives**: {asp['false_negatives']}
            """
        )

        st.markdown("---")
        st.subheader("Detailed Classification Reports")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("**Logistic Regression Report**")
            st.dataframe(pd.DataFrame(lr["classification_report"]).transpose(), use_container_width=True)
        with r_col2:
            st.markdown("**Naive Bayes Report**")
            st.dataframe(pd.DataFrame(nb["classification_report"]).transpose(), use_container_width=True)

    else:
        st.warning("Model metrics file not found. Run model_evaluation.py to populate metrics.")


# =============================================================
# PAGE 5: UNIVERSITY REPORT & VIVA
# =============================================================
elif page == "📘 University Report & Viva":
    st.title("📘 University Mini-Project Report & Viva Guide")
    st.markdown("Comprehensive academic documentation and viva voce examination preparation.")

    tab_doc, tab_viva, tab_arch = st.tabs(["📄 Academic Report Chapters", "🎓 Viva Voce Q&A", "🏗️ System Architecture"])

    with tab_doc:
        academic_report_path = Path("reports/academic_report.md")
        if academic_report_path.exists():
            with open(academic_report_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.markdown("Academic report document will be displayed here.")

    with tab_viva:
        viva_path = Path("reports/viva_questions.md")
        if viva_path.exists():
            with open(viva_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.markdown("Viva questions document will be displayed here.")

    with tab_arch:
        st.markdown("""
        ### End-to-End System Architecture

        ```
        +-----------------------------------------------------------------------------------+
        |                                Customer Product Review                            |
        |  "The phone camera is excellent, battery backup is poor, and display is amazing." |
        +-----------------------------------------+-----------------------------------------+
                                                  |
                                                  v
        +-----------------------------------------------------------------------------------+
        |                              NLP Preprocessing Engine                             |
        |  - Lowercasing & Contraction Expansion ("don't" -> "do not")                      |
        |  - Punctuation & Special Character Cleaning                                       |
        |  - Negation-Aware Stop-word Removal (Preserves "not", "no", "never", "barely")    |
        |  - POS-Informed Lemmatization (WordNet Lemmatizer)                                |
        +-----------------------------------------+-----------------------------------------+
                                                  |
                                                  v
        +-----------------------------------------------------------------------------------+
        |                         Aspect Extraction & Segmentation                          |
        |  - Clause Partitioning (Coordinating Conjunctions: but, however, and, yet)        |
        |  - Dependency Parsing & Noun Chunks (spaCy: nsubj, dobj, amod, acomp)             |
        |  - Aspect Taxonomy & Synonym Mapping (Longest matching compound priority)         |
        |                                                                                   |
        |    [Camera]  --> Clause: "The phone camera is excellent"                          |
        |    [Battery] --> Clause: "battery backup is poor"                                 |
        |    [Display] --> Clause: "and the display quality is amazing"                     |
        +-----------------------------------------+-----------------------------------------+
                                                  |
                                                  v
        +-----------------------------------------------------------------------------------+
        |                       Sentiment Classification Engine                             |
        |  - TF-IDF N-Gram Vectorizer (1, 2)                                                |
        |  - Classifiers: Logistic Regression (Balanced) | Naive Bayes | VADER              |
        |                                                                                   |
        |    Camera Context  --> [Positive] (Confidence: 85%)                               |
        |    Battery Context --> [Negative] (Confidence: 89%)                               |
        |    Display Context --> [Positive] (Confidence: 91%)                               |
        +-----------------------------------------+-----------------------------------------+
                                                  |
                                                  v
        +-----------------------------------------------------------------------------------+
        |                           Overall Sentiment Synthesis                             |
        |  - Multi-Polarity Resolution Logic:                                               |
        |    Positive Aspects + Negative Aspects ==> Overall: [MIXED]                       |
        +-----------------------------------------+-----------------------------------------+
                                                  |
                                                  v
        +-----------------------------------------------------------------------------------+
        |                       Interactive Streamlit Web Dashboard                         |
        |  - Live Single Review ABSA | Batch CSV Processing | Analytics | Benchmarks        |
        +-----------------------------------------------------------------------------------+
        ```
        """)

