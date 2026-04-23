"""
src/evaluate.py — Inference runner and metrics for the SED pipeline.

Two levels of evaluation are provided:

Frame-level (evaluate_frame_split):
  Per-frame multi-class precision/recall/F1 for each event class (background
  excluded).  Predictions are argmax of softmax outputs.  Uses DataLoader
  batches directly; absolute timestamps are not needed.

Event-level (evaluate_events):
  Each recording is processed independently via sliding-window inference
  (run_full_recording_inference).  Predicted events are compared to
  ground-truth annotations using a collar-based matching rule: a prediction
  is a True Positive if it overlaps a same-class GT event by at least 50 %
  of the shorter event's duration.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

from src.model import BowelSoundSEDModel
from src.postprocess import run_inference_and_postprocess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame-level inference and metrics
# ---------------------------------------------------------------------------

def run_frame_inference(
    model: BowelSoundSEDModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float | list[float] = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect per-frame predictions across a DataLoader.

    Args:
        model: Trained BowelSoundSEDModel.
        loader: DataLoader yielding {"input_values", "labels"} batches.
        device: CUDA or CPU.
        threshold: Per-class sigmoid threshold (float or list of floats).

    Returns:
        Tuple (all_labels, all_preds, all_probs):
          - all_labels: float32 array shape (N_frames, C) — binary ground truth.
          - all_preds:  int32  array shape (N_frames, C) — thresholded predictions.
          - all_probs:  float32 array shape (N_frames, C) — sigmoid probabilities.
    """
    model.eval()
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"]  # (B, T, C) float32 — keep on CPU

            logits = model(input_values)
            T = min(logits.size(1), labels.size(1))
            logits = logits[:, :T, :]
            labels = labels[:, :T, :]

            probs = torch.sigmoid(logits).cpu().numpy()  # (B, T, C)

            B, T, C = probs.shape
            thresh_arr = np.array(
                threshold if isinstance(threshold, list) else [threshold] * C,
                dtype=np.float32,
            )
            preds = (probs >= thresh_arr).astype(np.int32)  # (B, T, C)

            all_labels.append(labels.numpy().reshape(-1, C))
            all_preds.append(preds.reshape(-1, C))
            all_probs.append(probs.reshape(-1, C))

    return (
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_probs, axis=0),
    )


def compute_frame_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Compute per-class binary frame-level metrics for multi-label sigmoid predictions.

    Args:
        labels: float32 array shape (N, C) — binary ground truth (0.0 or 1.0).
        preds:  int32   array shape (N, C) — thresholded binary predictions.
        probs:  float32 array shape (N, C) — sigmoid probabilities.
        class_names: Ordered list of event class names, e.g. ["b", "mb", "h", "noise"].

    Returns:
        Dict with:
          - "frame_macro_f1": float (macro average over all event classes)
          - "per_class": dict[class_name → {"precision", "recall", "f1"}]
    """
    per_class = {}
    f1s = []
    for cls_idx, cls in enumerate(class_names):
        y_true = labels[:, cls_idx].astype(int)
        y_pred = preds[:, cls_idx].astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        per_class[cls] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    return {"frame_macro_f1": macro_f1, "per_class": per_class}


def evaluate_frame_split(
    model: BowelSoundSEDModel,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    split_name: str = "test",
) -> dict:
    """
    Frame-level evaluation on one dataset split.

    Args:
        model: Trained BowelSoundSEDModel.
        loader: DataLoader (no augmentation, no shuffle).
        class_names: Ordered list of class names.
        device: CUDA or CPU.
        split_name: Label for log messages.

    Returns:
        Dict with "frame_metrics", "labels", "preds", "probs".
    """
    logger.info("--- Frame-level evaluation on %s set ---", split_name)
    labels_np, preds_np, probs_np = run_frame_inference(model, loader, device)
    frame_metrics = compute_frame_metrics(labels_np, preds_np, probs_np, class_names)

    logger.info(
        "[%s] frame macro F1=%.4f", split_name, frame_metrics["frame_macro_f1"]
    )
    for cls, m in frame_metrics["per_class"].items():
        logger.info(
            "[%s] %s  P=%.4f  R=%.4f  F1=%.4f",
            split_name, cls, m["precision"], m["recall"], m["f1"],
        )

    return {
        "frame_metrics": frame_metrics,
        "labels": labels_np,
        "preds": preds_np,
        "probs": probs_np,
    }


# ---------------------------------------------------------------------------
# Event-level (collar-based) metrics
# ---------------------------------------------------------------------------

def _overlap_ratio(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """Fraction of the shorter event that is covered by the intersection."""
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    shorter = min(pred_end - pred_start, gt_end - gt_start)
    return inter / shorter if shorter > 0 else 0.0


def compute_event_metrics(
    pred_events: list[dict],
    gt_events: list[dict],
    class_names: list[str],
    overlap_threshold: float = 0.5,
) -> dict:
    """
    Collar-based event-level precision, recall, and F1.

    A predicted event is a True Positive if there exists an unmatched
    ground-truth event of the same class that overlaps it by at least
    `overlap_threshold` of the shorter event's duration.

    Each GT event can be matched at most once (greedy, sorted by overlap ratio).

    Args:
        pred_events: List of {"start", "end", "label"} dicts (predicted).
        gt_events:   List of {"start", "end", "label"} dicts (ground truth).
        class_names: Ordered list of class names.
        overlap_threshold: Minimum overlap ratio to count as a match.

    Returns:
        Dict with:
          - "event_macro_f1": float
          - "per_class": dict[class_name → {"precision", "recall", "f1", "tp", "fp", "fn"}]
    """
    per_class: dict[str, dict] = {}

    for cls in class_names:
        preds_cls = [e for e in pred_events if e["label"] == cls]
        gts_cls = [e for e in gt_events if e["label"] == cls]

        matched_gt: set[int] = set()
        tp = 0
        for pred in preds_cls:
            best_ratio = 0.0
            best_gt_idx = -1
            for gi, gt in enumerate(gts_cls):
                if gi in matched_gt:
                    continue
                ratio = _overlap_ratio(
                    pred["start"], pred["end"], gt["start"], gt["end"]
                )
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_gt_idx = gi
            if best_ratio >= overlap_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)

        fp = len(preds_cls) - tp
        fn = len(gts_cls) - len(matched_gt)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    return {"event_macro_f1": macro_f1, "per_class": per_class}


def evaluate_events(
    model: BowelSoundSEDModel,
    waveforms: dict[str, np.ndarray],
    all_segments: list,
    recording_ranges: dict[str, tuple[float, float]],
    processor,
    device: torch.device,
    class_names: list[str],
    split_name: str = "test",
    chunk_duration: float = 10.0,
    chunk_hop: float = 5.0,
    thresholds: list[float] | float = 0.5,
    median_frames: int = 5,
    min_event_frames_per_class: list[int] | int = 3,
    sr: int = 16_000,
    compute_n_frames_fn=None,
) -> tuple[dict, list[dict]]:
    """
    Event-level evaluation via full per-recording inference.

    Each recording in `recording_ranges` is processed independently with
    the sliding-window inference, so predicted event timestamps are correct
    absolute seconds within that recording.

    GT events are taken from `all_segments` restricted to each recording's
    time range.

    Args:
        model: Trained model (any backbone).
        waveforms: Dict of filename → full float32 waveform.
        all_segments: All parsed Segment objects.
        recording_ranges: Dict of filename → (start_sec, end_sec) for this split.
        processor: Wav2Vec2Processor or None (CNN14).
        device: CUDA or CPU.
        class_names: Ordered list of class names.
        split_name: Label for log messages.
        chunk_duration / chunk_hop / thresholds / median_frames /
        min_event_frames_per_class / sr: Post-processing parameters.
        compute_n_frames_fn: Callable(n_samples) → n_frames (backbone-specific).

    Returns:
        Tuple of (event_metrics_dict, list of all predicted event dicts).
    """
    logger.info("--- Event-level evaluation on %s set ---", split_name)

    all_pred_events: list[dict] = []
    all_gt_events: list[dict] = []

    for fname, (range_start, range_end) in recording_ranges.items():
        if fname not in waveforms:
            continue

        waveform = waveforms[fname]
        # Restrict inference to the split time range only.
        start_sample = int(range_start * sr)
        end_sample = min(int(range_end * sr), len(waveform))
        waveform_slice = waveform[start_sample:end_sample]

        _, pred_events = run_inference_and_postprocess(
            model=model,
            waveform=waveform_slice,
            processor=processor,
            device=device,
            class_names=class_names,
            chunk_duration=chunk_duration,
            chunk_hop=chunk_hop,
            thresholds=thresholds,
            median_frames=median_frames,
            min_event_frames_per_class=min_event_frames_per_class,
            sr=sr,
            compute_n_frames_fn=compute_n_frames_fn,
        )

        # Shift predicted times by range_start to get absolute recording times.
        for ev in pred_events:
            ev["start"] = round(ev["start"] + range_start, 4)
            ev["end"] = round(ev["end"] + range_start, 4)
        all_pred_events.extend(pred_events)

        # Collect GT events for this recording's split range.
        # Assign by where the event starts to ensure each event belongs to
        # exactly one split (no double-counting at boundaries).
        for seg in all_segments:
            if seg.source != fname:
                continue
            if seg.start >= range_start and seg.start < range_end:
                all_gt_events.append(
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "label": seg.label_name,
                    }
                )

    event_metrics = compute_event_metrics(all_pred_events, all_gt_events, class_names)
    logger.info(
        "[%s] event macro F1=%.4f", split_name, event_metrics["event_macro_f1"]
    )
    for cls, m in event_metrics["per_class"].items():
        logger.info(
            "[%s] %s  P=%.4f  R=%.4f  F1=%.4f  (TP=%d FP=%d FN=%d)",
            split_name, cls, m["precision"], m["recall"], m["f1"],
            m["tp"], m["fp"], m["fn"],
        )

    return event_metrics, all_pred_events
