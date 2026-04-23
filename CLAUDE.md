# CLAUDE.md — Codebase guide for AI assistants

## What this project does

This is a **frame-level Sound Event Detection (SED)** pipeline for bowel sounds.
Input: an arbitrary-length `.wav` recording.
Output: a sequence of `(start_time, end_time, label)` tuples written to a tab-separated CSV.

The pipeline is **not** a clip classifier. Do not suggest mean-pooling, single-label cross-entropy, or per-segment inference. The architecture runs Wav2Vec2 as a frame-level feature extractor, attaches a per-frame linear head, and produces one probability per class per ~20 ms frame.

---

## Repository layout

```
/home/digihealth project/
├── config.py           # All hyperparameters — single source of truth
├── train.py            # Training entry point
├── evaluate.py         # Standalone evaluation on a saved checkpoint
├── requirements.txt
├── data/
│   ├── AS_1.wav        # ~2517 s, 16 kHz
│   ├── AS_1.txt        # Tab-separated annotations: start  end  label
│   ├── 23M74M.wav      # ~519 s, 48 kHz → resampled to 16 kHz
│   └── 23M74M.txt
└── src/
    ├── dataset.py      # BowelSoundSEDDataset, time_based_split, frame label construction
    ├── model.py        # BowelSoundSEDModel (Wav2Vec2 + per-frame Linear head)
    ├── postprocess.py  # Sliding-window inference, median filter, event extraction, CSV export
    ├── trainer.py      # BCEWithLogitsLoss, train/val loops, early stopping, checkpointing
    ├── evaluate.py     # Frame-level and event-level metrics
    └── augmentation.py # Waveform augmentation (time-stretch, pitch-shift, masking)
```

---

## Classes and functions

| Symbol | File | Purpose |
|---|---|---|
| `SEDConfig` | `config.py` | Chunk duration, hop, threshold, median filter, min event frames |
| `DataConfig` | `config.py` | Data directory, sample rate, label map, class names |
| `TrainingConfig` | `config.py` | Batch size (4), LR (1e-5), epochs, early stopping patience |
| `BowelSoundSEDDataset` | `src/dataset.py` | Chunks recordings into fixed-length windows; builds `(T_frames,)` int64 single-label targets |
| `time_based_split` | `src/dataset.py` | Splits each recording by time fraction, returns `{fname: (start_s, end_s)}` dicts |
| `compute_n_frames` | `src/dataset.py` | Exact CNN output length given n_samples; mirrors the 7-layer wav2vec2-base extractor |
| `BowelSoundSEDModel` | `src/model.py` | `forward()` returns `(B, T_frames, num_classes)` raw logits — no pooling |
| `compute_class_weights` | `src/trainer.py` | `weight[c] = total_frames / (num_classes × frames_c)` — inverse-frequency weights for CE |
| `run_training` | `src/trainer.py` | Full loop: CE loss, AdamW + warmup scheduler, gradient clipping, early stopping on val macro F1 |
| `run_full_recording_inference` | `src/postprocess.py` | Sliding-window, averages overlapping chunk predictions, returns `(T_total, num_classes)` probs |
| `extract_events` | `src/postprocess.py` | medfilt → threshold → ndimage.label → overlap resolution → `[(start_frame, end_frame, cls)]` |
| `run_inference_and_postprocess` | `src/postprocess.py` | End-to-end wrapper: waveform → event list of `{"start", "end", "label"}` dicts |
| `evaluate_frame_split` | `src/evaluate.py` | DataLoader-based per-frame P/R/F1 per class |
| `evaluate_events` | `src/evaluate.py` | Per-recording sliding-window inference + collar-based event matching (50 % overlap threshold) |

---

## Key constants and numbers

- **Frame stride**: 320 samples at 16 kHz → 1 frame ≈ 20 ms → ~50 fps
- **CNN layers** (wav2vec2-base): `[(10,5),(3,2),(3,2),(3,2),(3,2),(2,2),(2,2)]`
- **Classes**: `b`=0, `mb`=1, `h`=2.  No background class.
- **Label aliases**: `sb`, `sbs` → class 0 (`b`); `n`, `v` → ignored
- **Unannotated frames**: label `-1` — excluded from loss via `ignore_index=-1`
- **Training chunks**: 10 s, non-overlapping (hop = 10 s)
- **Inference chunks**: 10 s, 5 s hop (overlapping); `5.0 s × 16000 = 80000 = 250 × 320` → exact frame alignment
- **Batch size**: 4 (10 s chunks are large; batch 16 would OOM on a single GPU)
- **Learning rate**: 1e-5 (CNN feature extractor frozen; only transformer + head trained)

---

## Data split strategy

Splits are **time-based within each recording**:

| Recording | Duration | Train | Val | Test |
|---|---|---|---|---|
| AS_1.wav | ~2517 s | 0 – 1762 s | 1762 – 2139 s | 2139 – 2517 s |
| 23M74M.wav | ~519 s | 0 – 363 s | 363 – 441 s | 441 – 519 s |

**Never use per-segment random splits** — temporal autocorrelation in biosignals means a random split leaks context from the test period into training.

---

## Loss function

`CrossEntropyLoss` with per-class `weight`. The weight for class `c` is:

```
weight[c] = total_frames / (num_classes * frames_of_class_c)
```

This is standard inverse-frequency weighting. Rarer classes (event classes) receive higher weight so they contribute equally to the total loss. Background (class 0) is heavily down-weighted because it dominates frame counts.

There is no background class. Event classes are 0=b, 1=mb, 2=h. The model outputs 3-class softmax; argmax on frames above the confidence threshold gives the predicted event class. Frames below the threshold produce no event.

---

## Shape alignment

The wav2vec2 CNN stride is 320 but integer arithmetic can produce `T_model` vs `T_labels` differing by ±1. Every forward pass uses:

```python
T = min(logits.size(1), labels.size(1))
logits, labels = logits[:, :T, :], labels[:, :T, :]
```

This is applied in `_align_logits_labels` (trainer) and in `run_frame_inference` (evaluate).

---

## Post-processing pipeline

```
(T_total, 4) softmax probs  (0=bg, 1=b, 2=mb, 3=h)
  → per-class medfilt(kernel=5 frames = 100 ms)
  → argmax per frame → single class index
  → low-confidence gate: max_prob < 0.5 → force to bg
  → per non-bg class: ndimage.label → connected components
  → drop events shorter than 3 frames (60 ms)
  → convert frames to seconds: start_s = start_frame × 320 / 16000
  → save as tab-separated CSV: start  end  label
```

No overlap resolution needed — argmax is already mutually exclusive.

---

## Evaluation: two levels

1. **Frame-level** (`evaluate_frame_split`): uses the DataLoader directly. Per-frame binary P/R/F1 for each class. Absolute timestamps are irrelevant; multiple recordings can be batched.

2. **Event-level** (`evaluate_events`): processes each recording independently. Slices the waveform to the split's time range, runs sliding-window inference, shifts predicted timestamps by `range_start` to get absolute times, then applies collar-based matching (≥50 % overlap of the shorter event's duration = TP). Each GT event matched at most once (greedy, best-overlap-first).

**Do not merge these two.** An earlier version concatenated DataLoader chunks into a single probability trace, which scrambled absolute timestamps across recordings.

---

## Running the pipeline

```bash
# Train (50 epochs, early stopping on val macro F1)
python train.py

# Quick smoke test
python train.py --epochs 3

# Disable augmentation (baseline run)
python train.py --no-augment

# Evaluate a saved checkpoint
python evaluate.py --checkpoint outputs/best_model.pt
python evaluate.py --checkpoint outputs/best_model.pt --split val
python evaluate.py --checkpoint outputs/best_model.pt --predict   # also saves per-recording CSVs
```

Outputs land in `outputs/`:
- `best_model.pt` — best checkpoint by val macro F1
- `last_model.pt` — most recent checkpoint
- `test_metrics.json` — frame + event metrics on the test split
- `metrics.csv` — per-epoch training curves
- `history.json` — full training history
- `predictions/*.txt` — per-recording event CSVs (tab-separated)

---

## What NOT to do

- Do not add mean pooling to `BowelSoundSEDModel.forward()`.
- Do not use `BCEWithLogitsLoss` or per-class sigmoid thresholding — `CrossEntropyLoss(weight=..., ignore_index=-1)` is the current design.
- Do not use per-segment stratified splits — use `time_based_split`.
- Do not concatenate chunk predictions from multiple recordings into a single array before extracting events.
- Do not skip `_align_logits_labels` — the ±1 frame mismatch will cause shape errors under BCE.
- Do not change the CNN layer list in `compute_n_frames` without updating `_WAV2VEC2_CNN_LAYERS` in `dataset.py` to match the actual checkpoint's architecture.
