"""
evaluate.py — Standalone evaluation script for a trained SED checkpoint.

Loads a saved checkpoint, rebuilds the dataset splits with the same config,
evaluates on the requested split (frame-level + event-level), and optionally
runs full-recording inference to produce prediction CSV files.

Usage:
    python evaluate.py --checkpoint outputs/best_model.pt
    python evaluate.py --checkpoint outputs/best_model.pt --split val
    python evaluate.py --checkpoint outputs/best_model.pt --predict
    python evaluate.py --checkpoint outputs/best_model.pt --split test --predict --output-dir /tmp/eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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
from src.trainer import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained bowel sound SED checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pt checkpoint file (e.g. outputs/best_model.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Also run full-recording inference and save prediction CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files. Defaults to checkpoint parent directory.",
    )
    parser.add_argument(
        "--lowpass-cutoff",
        type=float,
        default=None,
        help="Apply a low-pass Butterworth filter at this cutoff (Hz) before evaluation.",
    )
    parser.add_argument(
        "--bandpass-low",
        type=float,
        default=None,
        help="Bandpass filter lower cutoff in Hz (default 200). Pass 0 to disable.",
    )
    parser.add_argument(
        "--bandpass-high",
        type=float,
        default=None,
        help="Bandpass filter upper cutoff in Hz (default 2500). Pass 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cfg = get_config()
    if args.lowpass_cutoff is not None:
        cfg.data.lowpass_cutoff_hz = args.lowpass_cutoff
    if args.bandpass_low is not None:
        cfg.data.bandpass_low_hz = args.bandpass_low if args.bandpass_low > 0 else None
    if args.bandpass_high is not None:
        cfg.data.bandpass_high_hz = args.bandpass_high if args.bandpass_high > 0 else None
    output_dir = (
        Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load audio and annotations
    # ------------------------------------------------------------------
    logger.info("Loading audio files …")
    data_dir = Path(cfg.data.data_dir)
    waveforms = load_audio_files(
        data_dir,
        target_sr=cfg.data.sample_rate,
        lowpass_cutoff_hz=cfg.data.lowpass_cutoff_hz,
        bandpass_low_hz=cfg.data.bandpass_low_hz,
        bandpass_high_hz=cfg.data.bandpass_high_hz,
    )

    logger.info("Parsing annotations …")
    all_segments = build_segments(data_dir, cfg.data)

    # ------------------------------------------------------------------
    # Processor and frame-count function (backbone-dependent)
    # ------------------------------------------------------------------
    if cfg.model.backbone == "wav2vec2":
        logger.info("Loading Wav2Vec2Processor (%s) …", cfg.model.base_model)
        processor = Wav2Vec2Processor.from_pretrained(cfg.model.base_model)
        n_frames_fn = compute_n_frames
    else:
        logger.info("CNN14 backbone — skipping Wav2Vec2Processor.")
        processor = None
        n_frames_fn = compute_n_frames_mel

    # ------------------------------------------------------------------
    # Rebuild splits with the same fractions used during training
    # ------------------------------------------------------------------
    train_ranges, val_ranges, test_ranges = time_based_split(
        waveforms,
        sr=cfg.data.sample_rate,
        train_frac=cfg.training.train_split,
        val_frac=cfg.training.val_split,
    )
    ranges_map = {
        "train": train_ranges,
        "val": val_ranges,
        "test": test_ranges,
    }
    split_ranges = ranges_map[args.split]

    split_dataset = BowelSoundSEDDataset(
        waveforms=waveforms,
        segments=all_segments,
        recording_ranges=split_ranges,
        processor=processor,
        cfg_data=cfg.data,
        cfg_sed=cfg.sed,
        cfg_aug=cfg.augmentation,
        augment=False,
        compute_n_frames_fn=n_frames_fn,
    )
    split_loader = DataLoader(
        split_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
    )

    # ------------------------------------------------------------------
    # Load model and checkpoint
    # ------------------------------------------------------------------
    logger.info("Loading model (backbone=%s) …", cfg.model.backbone)
    model = build_model(cfg.model).to(device)
    load_checkpoint(model, checkpoint_path, device=device)

    # ------------------------------------------------------------------
    # Evaluate: frame-level + event-level
    # ------------------------------------------------------------------
    frame_results = evaluate_frame_split(
        model=model,
        loader=split_loader,
        class_names=cfg.data.class_names,
        device=device,
        split_name=args.split,
    )
    event_metrics, pred_events = evaluate_events(
        model=model,
        waveforms=waveforms,
        all_segments=all_segments,
        recording_ranges=split_ranges,
        processor=processor,
        device=device,
        class_names=cfg.data.class_names,
        split_name=args.split,
        chunk_duration=cfg.sed.chunk_duration,
        chunk_hop=cfg.sed.chunk_hop_infer,
        thresholds=cfg.sed.thresholds,
        median_frames=cfg.sed.median_filter_frames,
        min_event_frames_per_class=cfg.sed.min_event_frames_per_class,
        sr=cfg.data.sample_rate,
        compute_n_frames_fn=n_frames_fn,
    )

    # Save metrics JSON.
    serialisable = {
        "split": args.split,
        "frame_metrics": frame_results["frame_metrics"],
        "event_metrics": event_metrics,
    }
    out_json = output_dir / f"{args.split}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Metrics saved → %s", out_json)

    # Save predicted events from this split as a CSV.
    if pred_events:
        pred_csv = output_dir / f"{args.split}_pred_events.txt"
        save_events_csv(pred_events, pred_csv)

    # ------------------------------------------------------------------
    # Optional: full-recording inference → per-recording prediction CSVs
    # ------------------------------------------------------------------
    if args.predict:
        logger.info("Running full-recording inference …")
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

    logger.info("Done. Results in: %s", output_dir)


if __name__ == "__main__":
    main()
