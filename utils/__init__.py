"""Utility package for dataset loading and interactive visualizations."""
from utils.dataset_loader import load_dataset, preprocess_and_cache_dataset
from utils.visualization import (
    plot_sentiment_distribution,
    plot_aspect_sentiment_matrix,
    plot_model_comparison_metrics,
    plot_confusion_matrix,
    plot_aspect_frequencies
)

__all__ = [
    "load_dataset",
    "preprocess_and_cache_dataset",
    "plot_sentiment_distribution",
    "plot_aspect_sentiment_matrix",
    "plot_model_comparison_metrics",
    "plot_confusion_matrix",
    "plot_aspect_frequencies",
]

