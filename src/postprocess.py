"""
src/postprocess.py — Inference post-processing for the SED pipeline.

Converts the raw per-frame softmax probabilities from BowelSoundSEDModel
into a clean list of (start_time, end_time, label) events and saves them
as a tab-separated CSV matching the original annotation format.

Pipeline:
  1. Sliding-window inference over a full recording.
  2. Merge overlapping chunk predictions by averaging (softmax space).
  3. Per-class median filtering to smooth flickering predictions.
  4. Per-class threshold: frame active for class c if prob[c] >= threshold[c].
  5. Overlap resolution: if multiple classes active at same frame, keep highest.
  6. Extract connected active regions per class → connected components.
  7. Minimum-length noise gate (suppress very short detections).
  8. Export to tab-separated CSV.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.signal import medfilt
from scipy.ndimage import label as ndimage_label
from transformers import Wav2Vec2Processor

from src.dataset import compute_n_frames

logger = logging.getLogger(__name__)

# Effective wav2vec2-base stride in samples: product of all CNN strides.
_FRAME_STRIDE = 320


# ---------------------------------------------------------------------------
# Full-recording sliding-window inference
# ---------------------------------------------------------------------------

def run_full_recording_inference(
    model: torch.nn.Module,
    waveform: np.ndarray,
    processor: Wav2Vec2Processor | None,
    device: torch.device,
    chunk_duration: float = 10.0,
    chunk_hop: float = 5.0,
    sr: int = 16_000,
    compute_n_frames_fn=None,
) -> np.ndarray:
    """
    Run inference over an entire recording with overlapping chunks.

    Each position is covered by ceil(chunk_duration / chunk_hop) chunks;
    overlapping predictions are averaged.  The returned probability array
    has the same temporal resolution as the model output (~50 fps).

    Args:
        model: Trained model in eval mode — any backbone.
        waveform: Full float32 recording array shape (n_samples,).
        processor: Wav2Vec2Processor for normalisation, or None for CNN14
                   (the model handles feature extraction from raw waveform).
        device: CUDA or CPU device.
        chunk_duration: Length of each inference chunk in seconds.
        chunk_hop: Hop between consecutive chunks in seconds.
        sr: Sample rate.
        compute_n_frames_fn: Callable(n_samples) → n_frames.  Defaults to
                             the wav2vec2 CNN formula.

    Returns:
        Float32 array of shape (T_total, num_classes) with softmax
        probabilities averaged over overlapping chunks.
    """
    _n_frames = compute_n_frames_fn if compute_n_frames_fn is not None else compute_n_frames

    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(chunk_hop * sr)
    n_total = len(waveform)
    n_frames_total = _n_frames(n_total)

    # probs_acc is initialised lazily on the first chunk so n_classes is
    # inferred from the model output rather than being hardcoded.
    probs_acc: np.ndarray | None = None
    count_acc = np.zeros(n_frames_total, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for start_sample in range(0, n_total, hop_samples):
            end_sample = start_sample + chunk_samples
            chunk = waveform[start_sample : min(end_sample, n_total)].copy()

            # Zero-pad the last (possibly short) chunk.
            if len(chunk) < chunk_samples:
                chunk = np.concatenate(
                    [chunk, np.zeros(chunk_samples - len(chunk), dtype=np.float32)]
                )

            if processor is not None:
                processed = processor(chunk, sampling_rate=sr, return_tensors="pt")
                input_values = processed.input_values.to(device)
            else:
                input_values = torch.from_numpy(chunk).unsqueeze(0).to(device)

            logits = model(input_values)                                        # (1, T_chunk, C)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)          # (T_chunk, C)

            # Global frame offset: approximate linear mapping via stride 320.
            frame_offset = start_sample // _FRAME_STRIDE
            frame_end = frame_offset + probs.shape[0]

            if frame_offset >= n_frames_total:
                break
            if frame_end > n_frames_total:
                probs = probs[: n_frames_total - frame_offset]
                frame_end = n_frames_total

            if probs_acc is None:
                probs_acc = np.zeros((n_frames_total, probs.shape[1]), dtype=np.float64)

            probs_acc[frame_offset:frame_end] += probs
            count_acc[frame_offset:frame_end] += 1

    # Average overlapping windows; guard against zero-coverage gaps.
    if probs_acc is None:
        # Empty waveform — return zero array with 0 classes.
        return np.zeros((n_frames_total, 0), dtype=np.float32)
    count_safe = np.maximum(count_acc[:, np.newaxis], 1.0)
    return (probs_acc / count_safe).astype(np.float32)


# ---------------------------------------------------------------------------
# Thresholding, filtering, and event extraction
# ---------------------------------------------------------------------------

def extract_events(
    probs: np.ndarray,
    thresholds: list[float] | float = 0.5,
    median_frames: int = 5,
    min_event_frames_per_class: list[int] | int = 3,
) -> list[tuple[int, int, int]]:
    """
    Convert per-frame sigmoid probabilities into a list of detected events.

    Each class is thresholded independently (multi-label: multiple classes can
    be active on the same frame).

    Steps:
      1. Per-class median filter to smooth sigmoid traces.
      2. Per-class threshold → binary active mask.
      3. Connected-component labelling per class.
      4. Per-class minimum-length noise gate.

    Args:
        probs: Float32 array shape (T, num_classes) — sigmoid probabilities.
               Class indices: 0=b, 1=mb, 2=h, 3=noise.
        thresholds: Per-class threshold. List of length num_classes or a
                    single float applied to all classes.
        median_frames: Median filter kernel size in frames (must be odd).
        min_event_frames_per_class: Per-class minimum consecutive active
                    frames. List of length num_classes or a single int.

    Returns:
        List of (start_frame, end_frame_inclusive, class_idx) tuples
        sorted by start_frame.
    """
    kernel = median_frames if median_frames % 2 == 1 else median_frames + 1
    num_classes = probs.shape[1]

    if isinstance(thresholds, (int, float)):
        thresholds = [float(thresholds)] * num_classes
    if isinstance(min_event_frames_per_class, int):
        min_event_frames_per_class = [min_event_frames_per_class] * num_classes

    # Step 1: smooth all class traces.
    smoothed = probs.copy()
    if kernel > 1:
        for c in range(num_classes):
            smoothed[:, c] = medfilt(smoothed[:, c].astype(np.float64), kernel_size=kernel)

    # Steps 2–4: per-class threshold → connected components → length gate.
    events: list[tuple[int, int, int]] = []
    for cls_idx in range(num_classes):
        thresh = thresholds[cls_idx] if cls_idx < len(thresholds) else thresholds[-1]
        min_frames = (
            min_event_frames_per_class[cls_idx]
            if cls_idx < len(min_event_frames_per_class)
            else min_event_frames_per_class[-1]
        )
        active = (smoothed[:, cls_idx] >= thresh).astype(np.int32)
        labeled, n_components = ndimage_label(active)
        for comp_id in range(1, n_components + 1):
            frames = np.where(labeled == comp_id)[0]
            if len(frames) >= min_frames:
                events.append((int(frames[0]), int(frames[-1]), cls_idx))

    events.sort(key=lambda e: e[0])
    return events


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def events_to_seconds(
    events: list[tuple[int, int, int]],
    class_names: list[str],
    frame_stride: int = _FRAME_STRIDE,
    sr: int = 16_000,
) -> list[dict]:
    """
    Convert (start_frame, end_frame, class_idx) → (start_s, end_s, label).

    Args:
        events: List of (start_frame, end_frame_inclusive, class_idx).
        class_names: Ordered list of event class names, e.g.
                     ["b", "mb", "h", "other"].  class_names[cls_idx] gives
                     the label string.
        frame_stride: Samples per frame (default 320 for wav2vec2-base).
        sr: Sample rate.

    Returns:
        List of dicts with keys "start", "end", "label".
    """
    result = []
    for start_frame, end_frame, cls_idx in events:
        result.append(
            {
                "start": round(start_frame * frame_stride / sr, 4),
                "end": round((end_frame + 1) * frame_stride / sr, 4),
                "label": class_names[cls_idx],
            }
        )
    return result


def save_events_csv(
    events: list[dict],
    output_path: Path,
) -> None:
    """
    Save a list of event dicts to a tab-separated CSV.

    Output format matches the original annotation files:
        start_time<TAB>end_time<TAB>label

    Args:
        events: List of dicts with keys "start", "end", "label".
        output_path: Destination .csv or .txt path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for ev in events:
            writer.writerow([ev["start"], ev["end"], ev["label"]])
    logger.info("Saved %d events → %s", len(events), output_path)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_inference_and_postprocess(
    model: torch.nn.Module,
    waveform: np.ndarray,
    processor: Wav2Vec2Processor | None,
    device: torch.device,
    class_names: list[str],
    chunk_duration: float = 10.0,
    chunk_hop: float = 5.0,
    thresholds: list[float] | float = 0.5,
    median_frames: int = 5,
    min_event_frames_per_class: list[int] | int = 3,
    sr: int = 16_000,
    compute_n_frames_fn=None,
) -> tuple[np.ndarray, list[dict]]:
    """
    End-to-end: full-recording inference → post-processing → event list.

    Args:
        model: Trained model (any backbone).
        waveform: Full recording float32 array.
        processor: Wav2Vec2Processor, or None for CNN14.
        device: Target device.
        class_names: Ordered list of class name strings.
        chunk_duration: Chunk length for inference (seconds).
        chunk_hop: Sliding window hop (seconds).
        thresholds: Per-class detection threshold(s).  List of length
                    num_classes or a single float (applied to all classes).
        median_frames: Median filter window size (frames).
        min_event_frames_per_class: Per-class minimum event duration (frames).
                    List of length num_classes or a single int.
        sr: Sample rate.
        compute_n_frames_fn: Callable(n_samples) → n_frames (backbone-specific).

    Returns:
        Tuple of:
          - probs: (T_total, num_classes) float32 probability array.
          - events: List of {"start": float, "end": float, "label": str}.
    """
    probs = run_full_recording_inference(
        model, waveform, processor, device,
        chunk_duration=chunk_duration,
        chunk_hop=chunk_hop,
        sr=sr,
        compute_n_frames_fn=compute_n_frames_fn,
    )
    raw_events = extract_events(
        probs,
        thresholds=thresholds,
        median_frames=median_frames,
        min_event_frames_per_class=min_event_frames_per_class,
    )
    events = events_to_seconds(raw_events, list(class_names), sr=sr)
    logger.info("Detected %d events in %.0f s recording.", len(events), len(waveform) / sr)
    return probs, events
