# Bowel Sound Event Detection

Frame-level Sound Event Detection (SED) for bowel sounds using a Wav2Vec2 backbone. Given a raw `.wav` recording of any length, the pipeline outputs a sequence of `(start_time, end_time, label)` events saved as a tab-separated CSV.

---

## Overview

Bowel sounds carry clinical information about intestinal motility. Manually annotating long recordings is tedious and subjective. This project trains a deep learning model that can automatically segment and classify four types of bowel acoustic events in continuous recordings.

**It is not a clip classifier.** The model predicts one probability per class per ~20 ms frame across the full recording, then post-processing converts the probability traces into discrete event segments. This allows it to handle events of variable duration and back-to-back occurrences without any fixed segmentation.

---

## Approach

### Architecture

```
Raw waveform (any length)
        │
        ▼  sliding 2-second windows (1.7 s hop during inference)
Wav2Vec2Processor  ─── zero-mean / unit-variance normalisation
        │
        ▼
Wav2Vec2Model (CNN feature extractor → 12-layer transformer encoder)
        │  last_hidden_state  (B, T_frames, 768)  ~50 frames / second
        ▼
Dropout(0.1) → Linear(768, 4)
        │  logits  (B, T_frames, 4)
        ▼
sigmoid per class  ──  multi-label: multiple classes can be active on the same frame
        │
        ▼  post-processing
per-class median filter (5 frames = 100 ms)
per-class threshold  →  binary active mask
connected-component labelling per class
minimum-length noise gate
        │
        ▼
Events: [(start_s, end_s, label), ...]  →  CSV
```

**No mean pooling.** Background is implicit: a frame with all classes below threshold produces no event.

### Event classes

| Index | Label | Description |
|-------|-------|-------------|
| 0 | `b` | Single burst |
| 1 | `mb` | Multi-burst sequence |
| 2 | `h` | High-pitched sound |
| 3 | `noise` | Voice / environmental noise |

### Loss function

Binary Focal Loss with per-class positive weights:

```
FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)
```

- γ = 2.0 (down-weights easy/confident predictions)
- α weights derived from dataset frame ratios via `sqrt(neg_frames / pos_frames)`, normalised to `mb = 1.0`

| Class | α (pos_weight) | Rationale |
|-------|---------------|-----------|
| b     | 2.0 | Rare (~4.5 % of recording time) |
| mb    | 1.0 | Baseline (~15 % of recording time) |
| h     | 2.5 | Very rare (~2.8 % of recording time) |
| noise | 0.1 | Ubiquitous (~94 % of annotated time) |

### Training strategy

- **Time-based splits** within each recording (70 / 15 / 15) — no temporal leakage
- **Hard negative mining**: equal sampling of event-containing and background-only chunks
- CNN feature extractor frozen; only transformer layers + linear head are trained
- AdamW with linear warmup (10 % of total steps), gradient clipping at 1.0
- Early stopping on validation frame macro F1 (patience = 20)

---

## Data

Two abdominal auscultation recordings sampled at 16 kHz (23M74M resampled from 48 kHz), with tab-separated annotation files:

```
start_time  end_time  label
```

| Recording | Duration | Train | Val | Test |
|-----------|----------|-------|-----|------|
| AS_1.wav | 2212 s | 0–1549 s | 1549–1881 s | 1881–2212 s |
| 23M74M.wav | 301 s | 0–210 s | 210–255 s | 255–301 s |

A 200–2500 Hz bandpass filter is applied on load to suppress very low-frequency motion artefacts and high-frequency electronic noise.

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: PyTorch ≥ 2.1, torchaudio, transformers ≥ 4.37, librosa, scikit-learn, scipy.

Set `TRANSFORMERS_OFFLINE=1` after the first run to skip HuggingFace Hub version checks:

```bash
export TRANSFORMERS_OFFLINE=1
```

---

## Usage

### Train

```bash
# Default run (50 epochs, early stopping, augmentation on)
python train.py

```


| File | Contents |
|------|----------|
| `best_model.pt` | Checkpoint with best val macro F1 |
| `last_model.pt` | Most recent checkpoint |
| `test_metrics.json` | Frame + event metrics on the test split |
| `metrics.csv` | Per-epoch training curves |
| `predictions/*.txt` | Per-recording event CSVs |

### Evaluate a saved checkpoint

```bash
python evaluate.py --checkpoint outputs/best_model.pt
```

### Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | 50 | Number of training epochs |
| `--patience N` | 8 | Early stopping patience |
| `--batch-size N` | 4 | Batch size |
| `--lr F` | 1e-5 | Learning rate |
| `--chunk-duration F` | 10.0 | Chunk length in seconds |
| `--chunk-hop-train F` | 10.0 | Training chunk hop in seconds |
| `--chunk-hop-infer F` | 5.0 | Inference chunk hop in seconds |
| `--threshold F,...` | 0.7,0.7,0.3,0.5 | Per-class detection threshold |
| `--min-event-frames N,...` | 3,8,3,3 | Per-class minimum event length (frames) |
| `--pos-weight F,...` | 2.0,1.0,2.5,0.1 | Per-class focal loss α |
| `--focal-gamma F` | 2.0 | Focal loss γ |
| `--no-augment` | — | Disable waveform augmentation |
| `--backbone` | wav2vec2 | Feature extractor: `wav2vec2`, `cnn14`, `cnn14_dlmax` |
| `--recordings` | all | Comma-separated subset of `.wav` files to use |

---

## Results

Baseline run: no augmentation, 2-second chunks, batch size 50, patience 20. Stopped at epoch 36.

### Frame-level metrics (test set)

Per-frame binary precision / recall / F1. Each frame independently thresholded; multiple classes can be simultaneously active.

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| b | 0.506 | 0.238 | 0.324 |
| mb | 0.672 | 0.538 | 0.598 |
| h | 0.929 | 0.323 | 0.479 |
| noise | 1.000 | 0.975 | 0.987 |
| **macro** | | | **0.597** |

### Event-level metrics (test set)

Collar-based matching: a predicted event is a True Positive if it overlaps a same-class ground-truth event by ≥ 50 % of the shorter event's duration. Each GT event matched at most once.

| Class | Precision | Recall | F1 | TP | FP | FN |
|-------|-----------|--------|----|----|----|----|
| b | 0.500 | 0.018 | 0.034 | 1 | 1 | 56 |
| mb | 0.667 | 0.182 | 0.286 | 6 | 3 | 27 |
| h | 0.667 | 0.333 | 0.444 | 4 | 2 | 8 |
| noise | 1.000 | 0.063 | 0.118 | 1 | 0 | 15 |
| **macro** | | | **0.220** | | | |

### Interpreting the gap between frame-level and event-level F1

The frame-level and event-level results tell two different stories:

**What the model learned well:**
- `noise` at frame level (F1 = 0.99): the model has almost perfectly learned when noise/voice is present. High precision (1.0) means it never fires spuriously; high recall (0.97) means it almost never misses noise frames.
- `h` precision (0.93): when the model predicts a high-pitched event frame, it is almost always correct.
- `mb` frame-level balance (P=0.67, R=0.54): the most common event class shows the healthiest precision/recall trade-off.

**Where it struggles:**
- **`b` recall is critically low** (0.24 at frame level, 0.02 at event level): single bursts are short (~120 ms average) and the 2-second chunks with threshold 0.7 miss most of them. The model detects their frames only sporadically.
- **The frame → event gap for `noise`**: frame F1 = 0.99 but event F1 = 0.12. The model treats noise as a near-constant background state rather than segmenting it into distinct episodes, so the 15 GT noise events are merged into just 1 detected event in the test segment.
- **Overall event recall is the bottleneck** across all classes: the model's probability traces are activating at the right regions, but thresholding + minimum-frame gating is too conservative to yield enough connected-component events.




