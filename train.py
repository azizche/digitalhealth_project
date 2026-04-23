"""
train.py — Main training entry point for the bowel sound SED pipeline.

Usage:
    python train.py
    python train.py --no-augment           # disable augmentation (baseline)
    python train.py --output-dir /tmp/out  # override output directory
    python train.py --epochs 10            # quick smoke-test run

Steps:
    1.  Set random seed for reproducibility.
    2.  Load both .wav files into memory (resampling 23M74M to 16 kHz).
    3.  Parse all annotations.
    4.  Time-based split (70 / 15 / 15) — no temporal leakage.
    5.  Build BowelSoundSEDDataset × 3.
    6.  Create DataLoaders.
    7.  Instantiate BowelSoundSEDModel.
    8.  Train with run_training().
    9.  Load best checkpoint, evaluate on test set (frame + event level).
    10. Run full-recording inference on each recording → save prediction CSVs.
    11. Save test metrics JSON and training curves CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

from config import get_config
from src.dataset import (
    BowelSoundSEDDataset,
    build_segments,
    compute_n_frames,
    compute_n_frames_mel,
    load_audio_files,
    time_based_split,
)
from src.evaluate import evaluate_frame_split, evaluate_events
from src.model import build_model
from src.postprocess import run_inference_and_postprocess, save_events_csv
from src.trainer import load_checkpoint, run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bowel sound SED model")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation (useful for a baseline run).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory from config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs (e.g. 3 for a smoke test).",
    )
    parser.add_argument(
        "--unfreeze-cnn-epoch",
        type=int,
        default=None,
        help="Unfreeze the CNN feature extractor after this epoch number.",
    )
    def _int_list(s: str) -> list[int]:
        """Accept '3' or '3,8,3'."""
        return [int(x.strip()) for x in s.split(",")]

    def _float_list(s: str) -> list[float]:
        """Accept '0.7' or '0.7,0.7,0.3'."""
        return [float(x.strip()) for x in s.split(",")]

    parser.add_argument("--min-event-frames", type=_int_list, default=None,
        help="Per-class minimum event frames, e.g. '3,8,3' or a single int applied to all classes.")
    parser.add_argument("--threshold", type=_float_list, default=None,
        help="Per-class detection threshold, e.g. '0.7,0.7,0.3' or a single float applied to all classes.")
    parser.add_argument("--bandpass-low", type=float, default=None,
        help="Bandpass filter lower cutoff in Hz (default 200). Set to 0 to disable bandpass.")
    parser.add_argument("--bandpass-high", type=float, default=None,
        help="Bandpass filter upper cutoff in Hz (default 2500). Set to 0 to disable bandpass.")
    parser.add_argument("--pos-weight", type=_float_list, default=None,
        help="Focal loss positive-class weights for [b,mb,h,noise], "
             "e.g. '1.0,0.5,2.0,1.0'. Higher = positives of that class weighted more.")
    parser.add_argument("--focal-gamma", type=float, default=None,
        help="Focal loss focusing exponent γ (default 2.0). 0 = standard weighted CE.")
    parser.add_argument("--hard-neg-ratio", type=float, default=None,
        help="Background chunks per event chunk for hard negative mining (default 1.0).")
    parser.add_argument("--patience", type=int, default=None,
        help="Override training.early_stopping_patience (epochs without improvement before stopping).")
    parser.add_argument("--lr", type=float, default=None,
        help="Override training.learning_rate.")
    parser.add_argument("--lowpass-cutoff", type=float, default=None,
        help="Apply a low-pass Butterworth filter at this cutoff frequency (Hz) before training.")
    parser.add_argument("--freeze-feature-extractor", type=lambda x: x.lower() != "false",
        default=None, metavar="BOOL",
        help="Freeze CNN feature extractor (default True). Pass False to unfreeze.")
    parser.add_argument("--backbone", type=str, default=None,
        choices=["wav2vec2", "cnn14", "cnn14_dlmax"],
        help="Feature extraction backbone (default wav2vec2).")
    parser.add_argument("--panns-checkpoint", type=str, default=None,
        help="Path to pretrained CNN14 .pth file (only used with --backbone cnn14).")
    parser.add_argument("--recordings", type=str, default=None,
        help="Comma-separated list of .wav filenames to use (e.g. 'AS_1.wav'). "
             "Omit to use all recordings.")
    parser.add_argument("--chunk-duration", type=float, default=None,
        help="Duration of each audio chunk in seconds (default 10.0).")
    parser.add_argument("--chunk-hop-train", type=float, default=None,
        help="Hop between training chunks in seconds (default = chunk-duration, non-overlapping).")
    parser.add_argument("--chunk-hop-infer", type=float, default=None,
        help="Hop between inference chunks in seconds (default chunk-duration / 2).")
    parser.add_argument("--batch-size", type=int, default=None,
        help="Override training.batch_size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()

    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.no_augment:
        cfg.augmentation.enabled = False
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.min_event_frames is not None:
        n = len(cfg.data.class_names)
        vals = args.min_event_frames
        cfg.sed.min_event_frames_per_class = vals if len(vals) > 1 else vals * n
    if args.threshold is not None:
        n = len(cfg.data.class_names)
        vals = args.threshold
        cfg.sed.thresholds = vals if len(vals) > 1 else vals * n
    if args.pos_weight is not None:
        cfg.training.pos_weight = args.pos_weight
    if args.focal_gamma is not None:
        cfg.training.focal_gamma = args.focal_gamma
    if args.hard_neg_ratio is not None:
        cfg.training.hard_neg_ratio = args.hard_neg_ratio
    if args.patience is not None:
        cfg.training.early_stopping_patience = args.patience
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.freeze_feature_extractor is not None:
        cfg.model.freeze_feature_extractor = args.freeze_feature_extractor
    if args.backbone is not None:
        cfg.model.backbone = args.backbone
    if args.panns_checkpoint is not None:
        cfg.model.panns_checkpoint = args.panns_checkpoint
    if args.lowpass_cutoff is not None:
        cfg.data.lowpass_cutoff_hz = args.lowpass_cutoff
    if args.bandpass_low is not None:
        cfg.data.bandpass_low_hz = args.bandpass_low if args.bandpass_low > 0 else None
    if args.bandpass_high is not None:
        cfg.data.bandpass_high_hz = args.bandpass_high if args.bandpass_high > 0 else None
    if args.chunk_duration is not None:
        cfg.sed.chunk_duration = args.chunk_duration
    if args.chunk_hop_train is not None:
        cfg.sed.chunk_hop_train = args.chunk_hop_train
    if args.chunk_hop_infer is not None:
        cfg.sed.chunk_hop_infer = args.chunk_hop_infer
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.training.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load audio
    # ------------------------------------------------------------------
    data_dir = Path(cfg.data.data_dir)
    logger.info("Loading audio files from %s …", data_dir)
    waveforms = load_audio_files(
        data_dir,
        target_sr=cfg.data.sample_rate,
        lowpass_cutoff_hz=cfg.data.lowpass_cutoff_hz,
        bandpass_low_hz=cfg.data.bandpass_low_hz,
        bandpass_high_hz=cfg.data.bandpass_high_hz,
    )

    # ------------------------------------------------------------------
    # 2. Parse annotations
    # ------------------------------------------------------------------
    logger.info("Parsing annotations …")
    all_segments = build_segments(data_dir, cfg.data)

    # ------------------------------------------------------------------
    # Optional: restrict to a subset of recordings
    # ------------------------------------------------------------------
    if args.recordings is not None:
        keep = {r.strip() for r in args.recordings.split(",")}
        missing = keep - set(waveforms)
        if missing:
            raise ValueError(f"--recordings names not found in data dir: {missing}")
        waveforms = {k: v for k, v in waveforms.items() if k in keep}
        all_segments = [s for s in all_segments if s.source in keep]
        logger.info("Restricted to recordings: %s", sorted(keep))

    # ------------------------------------------------------------------
    # 3. Time-based split
    # ------------------------------------------------------------------
    logger.info("Performing time-based train/val/test split …")
    train_ranges, val_ranges, test_ranges = time_based_split(
        waveforms,
        sr=cfg.data.sample_rate,
        train_frac=cfg.training.train_split,
        val_frac=cfg.training.val_split,
    )

    # ------------------------------------------------------------------
    # 4. Processor + frame-count function (backbone-dependent)
    # ------------------------------------------------------------------
    if cfg.model.backbone == "wav2vec2":
        logger.info("Loading Wav2Vec2Processor (%s) …", cfg.model.base_model)
        processor = Wav2Vec2Processor.from_pretrained(cfg.model.base_model)
        n_frames_fn = compute_n_frames
    else:
        logger.info("CNN14 backbone — skipping Wav2Vec2Processor (mel extracted in model).")
        processor = None
        n_frames_fn = compute_n_frames_mel

    # ------------------------------------------------------------------
    # 5. Datasets
    # ------------------------------------------------------------------
    use_augment = cfg.augmentation.enabled

    train_dataset = BowelSoundSEDDataset(
        waveforms=waveforms,
        segments=all_segments,
        recording_ranges=train_ranges,
        processor=processor,
        cfg_data=cfg.data,
        cfg_sed=cfg.sed,
        cfg_aug=cfg.augmentation,
        augment=use_augment,
        compute_n_frames_fn=n_frames_fn,
    )
    val_dataset = BowelSoundSEDDataset(
        waveforms=waveforms,
        segments=all_segments,
        recording_ranges=val_ranges,
        processor=processor,
        cfg_data=cfg.data,
        cfg_sed=cfg.sed,
        cfg_aug=cfg.augmentation,
        augment=False,
        compute_n_frames_fn=n_frames_fn,
    )
    test_dataset = BowelSoundSEDDataset(
        waveforms=waveforms,
        segments=all_segments,
        recording_ranges=test_ranges,
        processor=processor,
        cfg_data=cfg.data,
        cfg_sed=cfg.sed,
        cfg_aug=cfg.augmentation,
        augment=False,
        compute_n_frames_fn=n_frames_fn,
    )

    logger.info(
        "Chunks — train: %d | val: %d | test: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )

    # ------------------------------------------------------------------
    # 6. DataLoaders
    # ------------------------------------------------------------------
    train_sampler = train_dataset.make_hard_neg_sampler(
        hard_neg_ratio=cfg.training.hard_neg_ratio
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,          # replaces shuffle=True
        num_workers=cfg.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ------------------------------------------------------------------
    # 7. Model
    # ------------------------------------------------------------------
    logger.info("Instantiating model (backbone=%s) …", cfg.model.backbone)
    model = build_model(cfg.model).to(device)
    param_counts = model.count_parameters()
    logger.info(
        "Parameters — trainable: %s / total: %s",
        f"{param_counts['trainable']:,}",
        f"{param_counts['total']:,}",
    )

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )

    # ------------------------------------------------------------------
    # 9. Evaluate on test set using best checkpoint
    # ------------------------------------------------------------------
    best_ckpt = output_dir / "best_model.pt"
    if best_ckpt.exists():
        load_checkpoint(model, best_ckpt, device=device)
        logger.info("Loaded best checkpoint for test evaluation.")

    frame_results = evaluate_frame_split(
        model=model,
        loader=test_loader,
        class_names=cfg.data.class_names,
        device=device,
        split_name="test",
    )
    event_metrics, pred_events = evaluate_events(
        model=model,
        waveforms=waveforms,
        all_segments=all_segments,
        recording_ranges=test_ranges,
        processor=processor,
        device=device,
        class_names=cfg.data.class_names,
        split_name="test",
        chunk_duration=cfg.sed.chunk_duration,
        chunk_hop=cfg.sed.chunk_hop_infer,
        thresholds=cfg.sed.thresholds,
        median_frames=cfg.sed.median_filter_frames,
        min_event_frames_per_class=cfg.sed.min_event_frames_per_class,
        sr=cfg.data.sample_rate,
        compute_n_frames_fn=n_frames_fn,
    )

    # Save serialisable test metrics.
    serialisable = {
        "frame_metrics": frame_results["frame_metrics"],
        "event_metrics": event_metrics,
    }
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Test metrics saved → %s", output_dir / "test_metrics.json")

    # ------------------------------------------------------------------
    # 10. Full-recording inference → prediction CSVs
    # ------------------------------------------------------------------
    logger.info("Running full-recording inference to generate prediction CSVs …")
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(exist_ok=True)

    for fname, waveform in waveforms.items():
        logger.info("  Inferring %s …", fname)
        _, events = run_inference_and_postprocess(
            model=model,
            waveform=waveform,
            processor=processor,
            device=device,
            class_names=cfg.data.class_names,
            chunk_duration=cfg.sed.chunk_duration,
            chunk_hop=cfg.sed.chunk_hop_infer,
            thresholds=cfg.sed.thresholds,
            median_frames=cfg.sed.median_filter_frames,
            min_event_frames_per_class=cfg.sed.min_event_frames_per_class,
            sr=cfg.data.sample_rate,
            compute_n_frames_fn=n_frames_fn,
        )
        stem = fname.replace(".wav", "")
        csv_path = preds_dir / f"{stem}_predictions.txt"
        save_events_csv(events, csv_path)

    logger.info("All done. Results in: %s", output_dir)


if __name__ == "__main__":
    main()
