"""
visualize.py — Plot generation for training analysis and model evaluation.

All functions accept numpy arrays / plain dicts so they can be called
independently of the training state.

Usage (standalone):
    python visualize.py --metrics outputs/metrics.csv --output outputs/plots/

Called programmatically from train.py and evaluate.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def plot_training_curves(
    history: dict[str, list],
    output_path: Path,
) -> None:
    """
    Plot training / validation loss and validation macro F1 over epochs.

    Args:
        history: Dict returned by run_training(). Expected keys:
                 epoch, train_loss, val_loss, val_macro_f1, val_macro_auc.
        output_path: Directory where training_curves.png is saved.
    """
    epochs = history.get("epoch", [])
    if not epochs:
        logger.warning("History is empty; skipping training curves plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves.
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"],   label="Val loss",   marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Macro F1 curve.
    ax = axes[1]
    ax.plot(epochs, history["val_macro_f1"], label="Val macro F1", color="green", marker="o", markersize=3)
    if "val_macro_auc" in history:
        ax.plot(epochs, history["val_macro_auc"], label="Val macro AUC", color="orange", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation metrics over epochs")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_path / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalize: bool = True,
) -> None:
    """
    Plot a confusion matrix heatmap.

    Args:
        cm: Integer confusion matrix shape (num_classes, num_classes).
        class_names: Ordered list of class labels.
        output_path: Directory where confusion_matrix.png is saved.
        normalize: If True, normalize rows to sum to 1 (shows recall per class).
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums == 0, 0, cm / row_sums)
        fmt = ".2f"
        title = "Confusion matrix (normalised)"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion matrix (counts)"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm_plot.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_plot[i, j]
            text = f"{val:{fmt}}" if fmt == "d" else f"{val:.2f}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=11,
            )

    fig.tight_layout()
    out = output_path / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """
    Plot one-vs-rest ROC curves for all classes with AUC in the legend.

    Args:
        labels: int array shape (N,) — ground-truth class indices.
        probs:  float array shape (N, num_classes) — softmax probabilities.
        class_names: Ordered list of class labels.
        output_path: Directory where roc_curves.png is saved.
    """
    num_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, cls in enumerate(class_names):
        if labels_bin.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f"Class '{cls}' (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_path / "roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_class_distribution(
    segments: list,
    output_path: Path,
    title: str = "Class distribution",
) -> None:
    """
    Bar chart of segment counts per class.

    Args:
        segments: List of Segment objects.
        output_path: Directory where class_distribution.png is saved.
        title: Chart title.
    """
    from collections import Counter
    counts = Counter(seg.label_name for seg in segments)
    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_path / "class_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Re-generate plots from saved metrics CSV")
    parser.add_argument("--metrics", type=str, required=True, help="Path to outputs/metrics.csv")
    parser.add_argument("--output", type=str, required=True, help="Output directory for plots")
    args = parser.parse_args()

    import csv
    from collections import defaultdict

    history: dict[str, list] = defaultdict(list)
    with open(args.metrics, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                try:
                    history[key].append(float(val))
                except ValueError:
                    history[key].append(val)

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    plot_training_curves(dict(history), out_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
