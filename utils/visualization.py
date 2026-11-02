"""
Visualization Module for Aspect-Based Sentiment Analysis.
Provides high-quality interactive Plotly figures and Seaborn/Matplotlib heatmaps
for aspect breakdowns, model metrics comparisons, and dataset analytics.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from config import SENTIMENT_COLORS, SENTIMENT_LABELS


def plot_sentiment_distribution(df: pd.DataFrame, col: str = "sentiment") -> go.Figure:
    """Creates an interactive donut chart of sentiment distributions."""
    counts = df[col].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    
    color_map = {
        "Positive": SENTIMENT_COLORS["Positive"],
        "Negative": SENTIMENT_COLORS["Negative"],
        "Neutral": SENTIMENT_COLORS["Neutral"],
        "Mixed": SENTIMENT_COLORS["Mixed"]
    }

    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        hole=0.45,
        color="Sentiment",
        color_discrete_map=color_map,
        title="Overall Sentiment Distribution"
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color="#ffffff", width=2))
    )
    fig.update_layout(
        font=dict(family="sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=30)
    )
    return fig


def plot_aspect_frequencies(aspect_counts: dict[str, int]) -> go.Figure:
    """Horizontal bar chart of most frequently mentioned product aspects."""
    sorted_items = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)
    df_aspects = pd.DataFrame(sorted_items, columns=["Aspect", "Mentions"])

    fig = px.bar(
        df_aspects,
        x="Mentions",
        y="Aspect",
        orientation="h",
        color="Mentions",
        color_continuous_scale="Viridis",
        title="Most Frequently Mentioned Product Aspects"
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Number of Mentions",
        yaxis_title="Product Aspect",
        margin=dict(l=20, r=20, t=50, b=30)
    )
    return fig


def plot_aspect_sentiment_matrix(aspect_sentiment_df: pd.DataFrame) -> go.Figure:
    """
    Stacked/grouped bar chart showing aspect-level sentiment breakdown
    (Positive, Negative, Neutral) per aspect category.
    """
    fig = go.Figure()

    for sent in ["Positive", "Neutral", "Negative"]:
        if sent in aspect_sentiment_df.columns:
            fig.add_trace(go.Bar(
                name=sent,
                x=aspect_sentiment_df["Aspect"],
                y=aspect_sentiment_df[sent],
                marker_color=SENTIMENT_COLORS.get(sent, "#999999")
            ))

    fig.update_layout(
        barmode="stack",
        title="Aspect-Wise Sentiment Distribution",
        xaxis_title="Product Aspect",
        yaxis_title="Review Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=30)
    )
    return fig


def plot_model_comparison_metrics(metrics: dict) -> go.Figure:
    """Grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models."""
    models = ["Logistic Regression", "Naive Bayes", "VADER"]
    metric_names = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]

    lr = metrics["logistic_regression"]
    nb = metrics["naive_bayes"]
    vd = metrics["vader"]

    data = {
        "Metric": metric_names * 3,
        "Model": ["Logistic Regression"] * 4 + ["Naive Bayes"] * 4 + ["VADER"] * 4,
        "Score": [
            lr["accuracy"], lr["precision_macro"], lr["recall_macro"], lr["f1_macro"],
            nb["accuracy"], nb["precision_macro"], nb["recall_macro"], nb["f1_macro"],
            vd["accuracy"], vd["precision_macro"], vd["recall_macro"], vd["f1_macro"]
        ]
    }

    df_plot = pd.DataFrame(data)

    fig = px.bar(
        df_plot,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".2%",
        title="Classification Benchmark Comparison (Test Set)",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1.1], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=50)
    )
    return fig


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str], title: str = "Confusion Matrix") -> go.Figure:
    """Generates an annotated heatmap for model confusion matrix using Plotly."""
    z = np.array(cm)
    x = class_names
    y = class_names

    # Format text annotations
    text_annotations = [[str(val) for val in row] for row in z]

    fig = ff_create_heatmap(z, x, y, text_annotations, title)
    return fig


def ff_create_heatmap(z: np.ndarray, x: list[str], y: list[str], annotations: list[list[str]], title: str) -> go.Figure:
    """Helper to render clean heatmap with cell values."""
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale="Blues",
        showscale=True,
        text=annotations,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=30, r=30, t=50, b=40)
    )
    return fig


def plot_product_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart showing sentiment breakdown by product category or name."""
    col = "product_category" if "product_category" in df.columns else "product_name"
    if col not in df.columns:
        col = "rating"

    grouped = df.groupby([col, "sentiment"]).size().unstack(fill_value=0).reset_index()

    fig = go.Figure()
    for sent in ["Positive", "Neutral", "Negative"]:
        if sent in grouped.columns:
            fig.add_trace(go.Bar(
                name=sent,
                x=grouped[col],
                y=grouped[sent],
                marker_color=SENTIMENT_COLORS.get(sent, "#999999")
            ))

    fig.update_layout(
        barmode="group",
        title=f"Sentiment Distribution by {col.replace('_', ' ').title()}",
        xaxis_title=col.replace('_', ' ').title(),
        yaxis_title="Number of Reviews",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=30)
    )
    return fig

