import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.model import mc_predict

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
FIGSIZE = (8, 6)


def full_evaluation(
    model,
    X_test: np.ndarray,
    y_test,
    feature_names: list[str],
    history=None,
    lr_tracker=None,
    results_dir: Optional[Path] = None,
) -> dict:

    results_dir = results_dir or Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    y_test_arr = np.array(y_test)

    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    mc_mean, mc_std = mc_predict(model, X_test, n_samples=50)
    mc_mean = mc_mean.ravel()
    mc_std = mc_std.ravel()
    mc_pred = (mc_mean >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test_arr, y_pred)),
        "precision": float(precision_score(y_test_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test_arr, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test_arr, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_arr, y_proba)),
        "mc_accuracy": float(accuracy_score(y_test_arr, mc_pred)),
        "mc_mean_uncertainty": float(mc_std.mean()),
    }

    report = classification_report(
        y_test_arr, y_pred, target_names=["Healthy", "Heart Disease"]
    )
    logger.info("\n%s", report)

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved -> %s", metrics_path)

    _plot_confusion_matrix(y_test_arr, y_pred, results_dir)
    _plot_roc_curve(y_test_arr, y_proba, metrics["roc_auc"], results_dir)
    _plot_pr_curve(y_test_arr, y_proba, results_dir)

    if history is not None:
        _plot_training_history(history, lr_tracker, results_dir)

    _plot_uncertainty(mc_std, y_test_arr, mc_pred, results_dir)

    return metrics


def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, results_dir: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Disease"],
        yticklabels=["Healthy", "Disease"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved confusion_matrix.png")


def _plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, roc_auc_val: float, results_dir: Path
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(results_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Saved roc_curve.png")


def _plot_pr_curve(
    y_true: np.ndarray, y_proba: np.ndarray, results_dir: Path
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(recall, precision, linewidth=2, label=f"PR (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(results_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Saved precision_recall_curve.png")


def _plot_training_history(history, lr_tracker, results_dir: Path) -> None:
    h = history.history
    n_plots = 3 if (lr_tracker and lr_tracker.lrs) else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    axes[0].plot(h["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(h["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(h["accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(h["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if n_plots == 3:
        axes[2].plot(lr_tracker.lrs, label="LR (Exp Decay)", color="purple", linewidth=2)
        axes[2].set_title("Learning Rate Schedule")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    fig.suptitle("Training Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved training_curves.png")


def _plot_uncertainty(
    mc_std: np.ndarray, y_true: np.ndarray, mc_pred: np.ndarray, results_dir: Path
) -> None:
    correct = mc_pred == y_true
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(mc_std[correct], bins=20, alpha=0.6, label="Correct", color="green")
    ax.hist(mc_std[~correct], bins=20, alpha=0.6, label="Incorrect", color="red")
    ax.axvline(0.15, color="black", linestyle="--", label="Review Threshold (0.15)")
    ax.set_xlabel("MC Dropout Std Dev")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty Distribution - Correct vs. Incorrect Predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "uncertainty_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved uncertainty_distribution.png")
