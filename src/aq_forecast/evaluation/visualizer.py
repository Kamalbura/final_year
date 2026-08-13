"""
visualizer.py — Visualization utilities for the AQI prediction pipeline.

Generates:
    1. Training/validation loss curves
    2. Actual vs. predicted scatter and line plots
    3. Metric comparison tables and heatmaps
    4. Feature importance bar charts
    5. Attention weight heatmaps
    6. Residual distribution plots
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

from src.aq_forecast.config import PLOTS_DIR, AQI_CATEGORIES

logger = logging.getLogger(__name__)

# Consistent style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_training_history(history: Dict[str, List[float]],
                          model_name: str,
                          save_dir: str = None) -> str:
    """
    Plot training and validation loss curves.

    Args:
        history: Dict with 'train_loss' and 'val_loss' lists.
        model_name: Name for the plot title and filename.
        save_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    save_dir = save_dir or PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], label="Train Loss",
             linewidth=2, color="#2196F3")
    ax1.plot(epochs, history["val_loss"], label="Val Loss",
             linewidth=2, color="#F44336")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"{model_name} — Training & Validation Loss", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Learning rate schedule
    if "learning_rate" in history:
        ax2.plot(epochs, history["learning_rate"],
                 linewidth=2, color="#4CAF50")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Learning Rate", fontsize=12)
        ax2.set_title(f"{model_name} — Learning Rate Schedule", fontsize=14)
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_training_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved training history plot: {path}")
    return path


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     model_name: str, n_points: int = 500,
                     save_dir: str = None) -> str:
    """
    Plot actual vs. predicted AQI values (line + scatter).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        model_name: Name for title/filename.
        n_points: Max number of points to plot.
        save_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    save_dir = save_dir or PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    y_true = y_true.flatten()[:n_points]
    y_pred = y_pred.flatten()[:n_points]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Time series comparison
    ax1.plot(y_true, label="Actual AQI", linewidth=1.5,
             color="#2196F3", alpha=0.8)
    ax1.plot(y_pred, label="Predicted AQI", linewidth=1.5,
             color="#F44336", alpha=0.8)
    ax1.set_xlabel("Time Step", fontsize=12)
    ax1.set_ylabel("AQI", fontsize=12)
    ax1.set_title(f"{model_name} — Actual vs Predicted AQI", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add AQI category bands
    for cat, (lo, hi) in AQI_CATEGORIES.items():
        if lo < max(max(y_true), max(y_pred)):
            color_map = {
                "Good": "#4CAF50", "Satisfactory": "#8BC34A",
                "Moderate": "#FFC107", "Poor": "#FF9800",
                "Very Poor": "#F44336", "Severe": "#9C27B0"
            }
            ax1.axhspan(lo, min(hi, max(max(y_true), max(y_pred)) + 10),
                        alpha=0.08, color=color_map.get(cat, "gray"),
                        label=f"_{cat}")

    # Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.4, s=10, color="#673AB7")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--",
             linewidth=2, label="Perfect Prediction")
    ax2.set_xlabel("Actual AQI", fontsize=12)
    ax2.set_ylabel("Predicted AQI", fontsize=12)
    ax2.set_title(f"{model_name} — Scatter: Actual vs Predicted", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved prediction plot: {path}")
    return path


def plot_metric_heatmap(results: Dict[str, Dict[str, float]],
                        save_dir: str = None) -> str:
    """
    Create a heatmap comparing metrics across all models.

    Args:
        results: Dict of {model_name: {metric_name: value}}.
        save_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    save_dir = save_dir or PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Build DataFrame
    df = pd.DataFrame(results).T  # models as rows, metrics as columns

    # Select core metrics for the heatmap
    core_metrics = ["RMSE", "MAE", "R2", "MAPE", "F1_weighted", "F1_macro"]
    available = [m for m in core_metrics if m in df.columns]
    df_core = df[available].astype(float)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df_core) * 0.8)))

    sns.heatmap(
        df_core, annot=True, fmt=".4f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Model Performance Comparison — Metric Heatmap",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12)
    ax.set_xlabel("Metric", fontsize=12)

    plt.tight_layout()
    path = os.path.join(save_dir, "metric_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved metric heatmap: {path}")
    return path


def plot_metric_comparison_bars(results: Dict[str, Dict[str, float]],
                                save_dir: str = None) -> str:
    """
    Create grouped bar charts comparing key metrics across models.

    Args:
        results: Dict of {model_name: {metric_name: value}}.
        save_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    save_dir = save_dir or PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    metrics_to_plot = ["RMSE", "MAE", "R2", "F1_weighted"]
    available = [m for m in metrics_to_plot
                 if all(m in v for v in results.values())]

    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    colors = sns.color_palette("husl", len(results))

    for i, metric in enumerate(available):
        models = list(results.keys())
        values = [results[m].get(metric, 0) for m in models]

        bars = axes[i].barh(models, values, color=colors)
        axes[i].set_xlabel(metric, fontsize=12)
        axes[i].set_title(metric, fontsize=14, fontweight="bold")
        axes[i].grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_width() + 0.01 * max(values),
                         bar.get_y() + bar.get_height() / 2,
                         f"{val:.4f}", va="center", fontsize=9)

    plt.suptitle("Model Performance Comparison", fontsize=16,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "metric_comparison_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved metric comparison bars: {path}")
    return path


def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str,
                               save_dir: str = None) -> str:
    """
    Plot the distribution of prediction residuals.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        model_name: Model name for title.
        save_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    save_dir = save_dir or PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    residuals = y_true.flatten() - y_pred.flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(residuals, bins=50, edgecolor="black", alpha=0.7,
             color="#2196F3")
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax1.set_xlabel("Residual (Actual - Predicted)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(f"{model_name} — Residual Distribution", fontsize=14)

    # Q-Q type: residuals vs index
    ax2.scatter(range(len(residuals[:500])), residuals[:500],
                alpha=0.4, s=5, color="#F44336")
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Sample Index", fontsize=12)
    ax2.set_ylabel("Residual", fontsize=12)
    ax2.set_title(f"{model_name} — Residuals Over Time", fontsize=14)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_residuals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved residual plot: {path}")
    return path


def plot_feature_importance(importance: Dict[str, float],
                            model_name: str, top_n: int = 20,
                            save_dir: str = None) -> str:
    """
    Plot feature importance from gradient-boosting models.

    Args:
        importance: {feature_name: importance_score}.
        model_name: Model name for title.
        top_n: Number of top features to show.
        save_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    save_dir = save_dir or PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Sort and take top N
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top = sorted_imp[:top_n]
    names, values = zip(*top)

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    ax.barh(range(len(names)), values, color=sns.color_palette("viridis",
                                                                len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"{model_name} — Top {top_n} Feature Importance",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved feature importance plot: {path}")
    return path


def generate_results_table(results: Dict[str, Dict[str, float]],
                           save_dir: str = None) -> str:
    """
    Generate and save a formatted results comparison table.

    Args:
        results: Dict of {model_name: {metric_name: value}}.
        save_dir: Save directory.

    Returns:
        Path to saved CSV.
    """
    save_dir = save_dir or PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(results).T
    df.index.name = "Model"
    df = df.round(4)

    # Sort by RMSE (lower is better)
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE")

    path = os.path.join(save_dir, "results_comparison.csv")
    df.to_csv(path)

    logger.info(f"Saved results table: {path}")
    logger.info(f"\n{df.to_string()}")
    return path
