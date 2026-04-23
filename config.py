"""
config.py — Centralised hyperparameters for the bowel-sound SED pipeline.

All tunable knobs live here. Modify get_config() or individual dataclass fields
directly rather than scattering magic numbers across the codebase.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: str = "/home/digihealth project/data"
    sample_rate: int = 16_000
    # Maps raw annotation strings → integer class index.
    # Unannotated frames are treated as background (all-zero BCE target).
    # sb / sbs are alternative spellings of single burst found in 23M74M.txt.
    # n (noise) and v (voice) are ignored entirely (no segment created).
    label_map: dict = field(
        default_factory=lambda: {
            "b": 0, "sb": 0, "sbs": 0,
            "mb": 1,
            "h": 2,
            "n": 3, "v": 3,
        }
    )
    ignore_labels: set = field(default_factory=lambda: set())
    # 4 event classes (background is implicit — all-zero row).
    class_names: list = field(default_factory=lambda: ["b", "mb", "h", "noise"])
    # Low-pass Butterworth filter applied once after loading (None = disabled).
    lowpass_cutoff_hz: float = None
    # Bandpass Butterworth filter (200–2500 Hz by default; set either to None to disable).
    # Applied after lowpass (if both are set), but typically use one or the other.
    bandpass_low_hz: float = 200.0
    bandpass_high_hz: float = 2500.0


@dataclass
class SEDConfig:
    """Sound Event Detection chunking and post-processing parameters."""
    # Chunk duration in seconds for sliding-window processing.
    chunk_duration: float = 10.0
    # Hop between consecutive chunks during training (non-overlapping).
    chunk_hop_train: float = 10.0
    # Hop during inference (overlapping for better temporal resolution).
    chunk_hop_infer: float = 5.0
    # Per-class detection thresholds (index = class index: 0=b, 1=mb, 2=h, 3=noise).
    # Higher threshold → fewer but more confident detections (better precision).
    # Lower threshold → more detections (better recall).
    thresholds: list = field(default_factory=lambda: [0.7, 0.7, 0.3, 0.5])
    # Median filter kernel size in frames (1 frame ≈ 20 ms at 16 kHz).
    # Must be odd; if even, it is incremented by 1 internally.
    median_filter_frames: int = 5
    # Per-class minimum consecutive active frames to keep an event (noise gate).
    # At 50 fps: 3 frames = 60 ms, 5 = 100 ms, 8 = 160 ms, 10 = 200 ms.
    # mb uses a higher minimum (8 = 160 ms) to suppress spurious short bursts.
    min_event_frames_per_class: list = field(default_factory=lambda: [3, 8, 3, 3])


@dataclass
class AugmentationConfig:
    enabled: bool = True
    # Per-augmentation kill switches (only take effect when enabled=True).
    use_pitch_shift: bool = True
    use_gaussian_noise: bool = True
    use_gain_jitter: bool = True
    use_background_noise: bool = True
    # Pitch shift in semitones.
    pitch_shift_range: tuple = (-2.0, 2.0)
    p_pitch_shift: float = 0.5
    # Additive Gaussian noise std range (amplitude units, 0-peak normalised audio).
    # 0.005–0.02 ≈ light-to-moderate sensor noise.
    gaussian_noise_std_range: tuple = (0.005, 0.02)
    p_gaussian_noise: float = 0.5
    # Gain jitter: multiply amplitude by a random factor in this range.
    # 0.5–2.0 simulates loose/tight stethoscope contact variation.
    gain_jitter_range: tuple = (0.5, 2.0)
    p_gain_jitter: float = 0.5
    # Background noise mixing: SNR in dB (higher = cleaner signal).
    # 10–30 dB covers moderately noisy to faintly noisy conditions.
    background_noise_snr_db_range: tuple = (10.0, 30.0)
    p_background_noise: float = 0.5


@dataclass
class ModelConfig:
    # "wav2vec2", "cnn14", or "cnn14_dlmax"
    backbone: str = "wav2vec2"
    base_model: str = "facebook/wav2vec2-base"
    # Path to a pretrained CNN14 .pth checkpoint from the PANNS repo.
    # Leave empty to train CNN14 from scratch (or fine-tune from random init).
    panns_checkpoint: str = ""
    # 4 event classes: 0=b, 1=mb, 2=h, 3=noise. Background is implicit (all-zero row).
    num_classes: int = 4
    dropout: float = 0.1
    # Freeze CNN feature extractor; only fine-tune the top layers + head.
    freeze_feature_extractor: bool = True


@dataclass
class TrainingConfig:
    output_dir: str = "/home/digihealth project/outputs"
    # Reduced batch size vs clip-level: 10-second chunks are ~5× larger.
    batch_size: int = 4
    num_epochs: int = 50
    # Smaller LR than clip-level: we are fine-tuning per-frame predictions.
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 8
    train_split: float = 0.70
    val_split: float = 0.15
    # test_split is implicitly 1 - train_split - val_split
    random_seed: int = 42
    # Hard negative mining: background chunks sampled per event chunk each epoch.
    # 1.0 = equal number of background and event chunks.
    # Set to a larger value to include more background context.
    hard_neg_ratio: float = 1.0
    num_workers: int = 4
    fp16: bool = False
    log_every_n_steps: int = 10
    gradient_accumulation_steps: int = 1
    # Binary focal loss positive-class weights (alpha): [b, mb, h, noise].
    # Derived from dataset frame counts via sqrt(neg/pos), normalised to mb=1.0.
    # Empirical frame ratios: b neg/pos≈21.5, mb≈5.8, h≈35.0, noise≈0.07.
    # Noise covers ~94 % of annotated time → very low alpha to avoid false-firing.
    pos_weight: list = field(default_factory=lambda: [2.0, 1.0, 2.5, 0.1])
    # Focal loss focusing exponent γ. 0 = standard weighted BCE. 2 is typical.
    focal_gamma: float = 2.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    sed: SEDConfig = field(default_factory=SEDConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def get_config() -> Config:
    """Return the default Config. Modify fields directly for experiments."""
    return Config()
