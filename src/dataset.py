"""
src/dataset.py — Data loading, annotation parsing, and PyTorch Dataset for SED.

Design decisions:
- Both .wav files are loaded fully into RAM as float32 numpy arrays once.
  At 16 kHz float32: AS_1 (~2517 s) ≈ 160 MB, 23M74M resampled ≈ 33 MB.
- Splits are time-based (first 70 % of each recording = train, etc.) to
  prevent temporal leakage between splits.
- BowelSoundSEDDataset slices each recording into fixed-length chunks and
  builds a (T_frames, num_classes) float32 binary target for each chunk,
  where T_frames is the wav2vec2 CNN output length (~50 fps at 16 kHz).
- Frame labels are 1.0 for active event frames, 0.0 for background.
  All frames carry a valid training signal for BCEWithLogitsLoss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

from config import AugmentationConfig, DataConfig, SEDConfig
from src.augmentation import augment_waveform

logger = logging.getLogger(__name__)

# Mapping from audio filename to its paired annotation file.
ANNOTATION_MAP: dict[str, str] = {
    "AS_1.wav": "AS_1.txt",
    "23M74M.wav": "23M74M.txt",
}

# Native sample rates per file (before resampling to 16 kHz).
NATIVE_SAMPLE_RATES: dict[str, int] = {
    "AS_1.wav": 16_000,
    "23M74M.wav": 48_000,
}

# wav2vec2-base CNN feature extractor layer parameters: (kernel_size, stride).
# Total effective stride = 5×2×2×2×2×2×2 = 320 samples at 16 kHz (≈ 20 ms/frame).
_WAV2VEC2_CNN_LAYERS: list[tuple[int, int]] = [
    (10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)
]


def compute_n_frames(n_samples: int) -> int:
    """
    Compute the number of wav2vec2-base output frames for n_samples input.

    Applies the CNN feature extractor layer-by-layer using the exact
    kernel sizes and strides from the wav2vec2-base architecture.

    Args:
        n_samples: Number of raw audio samples.

    Returns:
        Number of output frames from the CNN feature extractor.
    """
    n = n_samples
    for kernel, stride in _WAV2VEC2_CNN_LAYERS:
        n = (n - kernel) // stride + 1
    return max(n, 0)


def compute_n_frames_mel(n_samples: int, hop_length: int = 320) -> int:
    """
    Approximate number of mel spectrogram frames for n_samples input.

    Matches torchaudio.transforms.MelSpectrogram with center=True (default):
    each side is padded by n_fft//2 before the STFT, giving
    n_frames = n_samples // hop_length + 1.

    Args:
        n_samples: Number of raw audio samples.
        hop_length: STFT hop in samples (default 320 → ~50 fps at 16 kHz).

    Returns:
        Number of mel spectrogram frames.
    """
    return max(0, n_samples // hop_length + 1)


@dataclass
class Segment:
    start: float       # seconds
    end: float         # seconds
    label: int         # encoded integer class index (0=b, 1=mb, 2=h, 3=noise)
    source: str        # key into the waveforms dict, e.g. "AS_1.wav"
    label_name: str    # human-readable class name

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_annotation_file(
    txt_path: Path,
    label_map: dict[str, int],
    ignore_labels: set[str],
) -> list[Segment]:
    """
    Parse a tab-separated annotation file into a list of Segment objects.

    Handles Windows \\r\\n line endings via splitlines().
    Labels are stripped of whitespace before lookup.
    Rows whose label is in ignore_labels or absent from label_map are skipped.

    Args:
        txt_path: Path to the .txt annotation file.
        label_map: Maps raw label strings to integer class indices.
        ignore_labels: Label strings to skip entirely (e.g. "n", "v").

    Returns:
        List of Segment objects sorted by start time.
    """
    source_key = txt_path.stem + ".wav"
    segments: list[Segment] = []

    raw = txt_path.read_text(encoding="utf-8", errors="replace")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue

        raw_label = parts[2].strip()
        if raw_label in ignore_labels:
            continue
        if raw_label not in label_map:
            continue

        label_int = label_map[raw_label]
        canonical = {0: "b", 1: "mb", 2: "h", 3: "noise"}.get(label_int, raw_label)

        segments.append(
            Segment(
                start=start,
                end=end,
                label=label_int,
                source=source_key,
                label_name=canonical,
            )
        )

    segments.sort(key=lambda s: s.start)
    return segments


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def apply_lowpass_filter(
    waveform: np.ndarray,
    cutoff_hz: float,
    sr: int,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter to a waveform."""
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2.0
    if cutoff_hz >= nyq:
        return waveform
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, waveform).astype(np.float32)


def apply_bandpass_filter(
    waveform: np.ndarray,
    low_hz: float,
    high_hz: float,
    sr: int,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to a waveform."""
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2.0
    low = low_hz / nyq
    high = high_hz / nyq
    # Clamp to a valid range (butter requires 0 < Wn < 1).
    low = max(low, 1e-6)
    high = min(high, 1.0 - 1e-6)
    if low >= high:
        logger.warning("Bandpass low (%.1f Hz) >= high (%.1f Hz) — filter skipped.", low_hz, high_hz)
        return waveform
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, waveform).astype(np.float32)


def load_audio_files(
    data_dir: Path,
    target_sr: int = 16_000,
    lowpass_cutoff_hz: float | None = None,
    bandpass_low_hz: float | None = None,
    bandpass_high_hz: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Load all paired .wav files into memory as float32 arrays at target_sr.

    Files not at target_sr are resampled once (kaiser_best quality).
    Waveforms are peak-normalised to [-1, 1].

    Args:
        data_dir: Directory containing the .wav files.
        target_sr: Target sample rate (default 16 000 Hz).

    Returns:
        Dict mapping filename key (e.g. "AS_1.wav") to float32 waveform
        shape (n_samples,).
    """
    waveforms: dict[str, np.ndarray] = {}

    for wav_name in ANNOTATION_MAP:
        wav_path = data_dir / wav_name
        if not wav_path.exists():
            logger.warning("Audio file not found, skipping: %s", wav_path)
            continue

        logger.info("Loading %s …", wav_name)
        audio, native_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if native_sr != target_sr:
            logger.info(
                "  Resampling %s from %d Hz → %d Hz", wav_name, native_sr, target_sr
            )
            audio = librosa.resample(
                audio, orig_sr=native_sr, target_sr=target_sr, res_type="kaiser_best"
            )

        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        if bandpass_low_hz is not None and bandpass_high_hz is not None:
            logger.info(
                "  Applying %.0f–%.0f Hz bandpass filter to %s",
                bandpass_low_hz, bandpass_high_hz, wav_name,
            )
            audio = apply_bandpass_filter(audio, bandpass_low_hz, bandpass_high_hz, target_sr)

        if lowpass_cutoff_hz is not None:
            logger.info(
                "  Applying %.0f Hz low-pass filter to %s", lowpass_cutoff_hz, wav_name
            )
            audio = apply_lowpass_filter(audio, lowpass_cutoff_hz, target_sr)

        waveforms[wav_name] = audio.astype(np.float32)
        logger.info(
            "  Loaded %s: %.1f s, %d samples", wav_name, len(audio) / target_sr, len(audio)
        )

    return waveforms


# ---------------------------------------------------------------------------
# Segment collection
# ---------------------------------------------------------------------------

def build_segments(
    data_dir: Path,
    cfg: DataConfig,
) -> list[Segment]:
    """
    Parse all annotation files and return the combined segment list.

    Args:
        data_dir: Directory containing annotation (.txt) files.
        cfg: DataConfig with label_map and ignore_labels.

    Returns:
        Combined list of Segment objects.
    """
    from collections import Counter

    all_segments: list[Segment] = []
    for wav_name, txt_name in ANNOTATION_MAP.items():
        txt_path = data_dir / txt_name
        if not txt_path.exists():
            logger.warning("Annotation file not found, skipping: %s", txt_path)
            continue
        segs = parse_annotation_file(txt_path, cfg.label_map, cfg.ignore_labels)
        logger.info("  %s: %d target segments parsed", txt_name, len(segs))
        all_segments.extend(segs)

    logger.info("Total segments: %d", len(all_segments))
    counts = Counter(s.label_name for s in all_segments)
    for cls, cnt in sorted(counts.items()):
        logger.info("  %s: %d", cls, cnt)

    return all_segments


# ---------------------------------------------------------------------------
# Time-based split
# ---------------------------------------------------------------------------

def time_based_split(
    waveforms: dict[str, np.ndarray],
    sr: int,
    train_frac: float,
    val_frac: float,
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, tuple[float, float]],
    dict[str, tuple[float, float]],
]:
    """
    Split each recording into train / val / test time ranges.

    For each recording the first `train_frac` of its duration is training,
    the next `val_frac` is validation, and the remainder is test.  This
    avoids temporal leakage that would occur with per-segment random splits.

    Args:
        waveforms: Dict of filename → float32 waveform array.
        sr: Sample rate.
        train_frac: Fraction of recording duration for training (e.g. 0.70).
        val_frac: Fraction of recording duration for validation (e.g. 0.15).

    Returns:
        Three dicts, each mapping filename → (start_sec, end_sec):
        (train_ranges, val_ranges, test_ranges)
    """
    train_ranges: dict[str, tuple[float, float]] = {}
    val_ranges: dict[str, tuple[float, float]] = {}
    test_ranges: dict[str, tuple[float, float]] = {}

    for fname, waveform in waveforms.items():
        duration = len(waveform) / sr
        t_train = duration * train_frac
        t_val = t_train + duration * val_frac
        train_ranges[fname] = (0.0, t_train)
        val_ranges[fname] = (t_train, t_val)
        test_ranges[fname] = (t_val, duration)
        logger.info(
            "  %s (%.0f s): train 0–%.0f s | val %.0f–%.0f s | test %.0f–%.0f s",
            fname, duration, t_train, t_train, t_val, t_val, duration,
        )

    return train_ranges, val_ranges, test_ranges


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BowelSoundSEDDataset(Dataset):
    """
    PyTorch Dataset for frame-level bowel sound event detection.

    Backbone-agnostic: works with both wav2vec2 and CNN14 (PANNS).

    Args:
        waveforms: Dict of filename → float32 waveform array (full recording).
        segments: All parsed Segment objects (the dataset will filter by
                  source filename and time range).
        recording_ranges: Dict of filename → (start_sec, end_sec) for this
                          split.
        processor: Wav2Vec2Processor for waveform normalisation, or None.
                   When None (CNN14 backbone) the raw waveform tensor is
                   returned directly — the model handles feature extraction.
        cfg_data: DataConfig.
        cfg_sed: SEDConfig controlling chunk size and hop.
        cfg_aug: AugmentationConfig (used only when augment=True).
        augment: Whether to apply waveform augmentation in __getitem__.
        hop: Override the chunk hop in seconds.  If None, uses
             cfg_sed.chunk_hop_train.
        compute_n_frames_fn: Callable(n_samples) → n_frames.  Defaults to
                             the wav2vec2 CNN formula.  Pass
                             compute_n_frames_mel for the CNN14 backbone.
    """

    def __init__(
        self,
        waveforms: dict[str, np.ndarray],
        segments: list[Segment],
        recording_ranges: dict[str, tuple[float, float]],
        processor: Wav2Vec2Processor | None,
        cfg_data: DataConfig,
        cfg_sed: SEDConfig,
        cfg_aug: AugmentationConfig,
        augment: bool = False,
        hop: float | None = None,
        frame_stride: int = 320,
        compute_n_frames_fn=None,
    ) -> None:
        self.waveforms = waveforms
        self.all_segments = segments
        self.recording_ranges = recording_ranges
        self.processor = processor
        self.cfg_data = cfg_data
        self.cfg_sed = cfg_sed
        self.cfg_aug = cfg_aug
        self.augment = augment

        self._hop = hop if hop is not None else cfg_sed.chunk_hop_train
        self._chunk_samples = int(cfg_sed.chunk_duration * cfg_data.sample_rate)
        self._frame_stride = frame_stride
        _n_frames_fn = compute_n_frames_fn if compute_n_frames_fn is not None else compute_n_frames
        self._n_frames_per_chunk = _n_frames_fn(self._chunk_samples)

        # Pre-build index: list of (filename, chunk_start_seconds).
        self._chunks: list[tuple[str, float]] = []
        self._build_chunks()

        logger.info(
            "BowelSoundSEDDataset: %d chunks  (augment=%s, hop=%.1f s)",
            len(self._chunks), augment, self._hop,
        )

    # ------------------------------------------------------------------
    # Chunk index construction
    # ------------------------------------------------------------------

    def _build_chunks(self) -> None:
        sr = self.cfg_data.sample_rate
        chunk_dur = self.cfg_sed.chunk_duration
        hop = self._hop

        for fname, (range_start, range_end) in self.recording_ranges.items():
            if fname not in self.waveforms:
                continue
            t = range_start
            while t < range_end:
                # Include chunk if its start is within the range.
                self._chunks.append((fname, t))
                t += hop

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fname, chunk_start = self._chunks[idx]
        sr = self.cfg_data.sample_rate
        chunk_dur = self.cfg_sed.chunk_duration

        waveform = self.waveforms[fname]
        start_sample = int(chunk_start * sr)
        end_sample = start_sample + self._chunk_samples

        # Extract chunk (zero-pad if we hit the end of the recording).
        chunk = waveform[start_sample : min(end_sample, len(waveform))].copy()
        if len(chunk) < self._chunk_samples:
            pad = np.zeros(self._chunk_samples - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])

        if self.augment:
            rng = np.random.default_rng()
            # Sample a non-overlapping slice from the same recording for noise mixing.
            # Excluded zone: any noise start ns where [ns, ns+chunk) overlaps [start, end),
            # i.e., ns in (start_sample - chunk_samples, end_sample).
            noise_pool = None
            if self.cfg_aug.use_background_noise:
                full_wav = self.waveforms[fname]
                n_chunk = self._chunk_samples
                max_noise_start = len(full_wav) - n_chunk
                if max_noise_start > 0:
                    excl_lo = max(0, start_sample - n_chunk + 1)
                    excl_hi = min(max_noise_start, end_sample - 1)
                    valid_before = excl_lo                        # positions [0, excl_lo)
                    valid_after = max(0, max_noise_start - excl_hi)  # positions (excl_hi, max_noise_start]
                    total_valid = valid_before + valid_after
                    if total_valid > 0:
                        pick = int(rng.integers(0, total_valid))
                        if pick < valid_before:
                            noise_start = pick
                        else:
                            noise_start = excl_hi + 1 + (pick - valid_before)
                        noise_slice = full_wav[noise_start:noise_start + n_chunk]
                        if len(noise_slice) < n_chunk:
                            noise_slice = np.concatenate([
                                noise_slice,
                                np.zeros(n_chunk - len(noise_slice), dtype=np.float32),
                            ])
                        noise_pool = noise_slice
            chunk = augment_waveform(
                chunk,
                sr=sr,
                p_pitch_shift=self.cfg_aug.p_pitch_shift,
                pitch_shift_range=self.cfg_aug.pitch_shift_range,
                p_gaussian_noise=self.cfg_aug.p_gaussian_noise,
                gaussian_noise_std_range=self.cfg_aug.gaussian_noise_std_range,
                p_gain_jitter=self.cfg_aug.p_gain_jitter,
                gain_jitter_range=self.cfg_aug.gain_jitter_range,
                p_background_noise=self.cfg_aug.p_background_noise,
                background_noise_snr_db_range=self.cfg_aug.background_noise_snr_db_range,
                noise_pool=noise_pool,
                rng=rng,
                use_pitch_shift=self.cfg_aug.use_pitch_shift,
                use_gaussian_noise=self.cfg_aug.use_gaussian_noise,
                use_gain_jitter=self.cfg_aug.use_gain_jitter,
                use_background_noise=self.cfg_aug.use_background_noise,
            )

        if self.processor is not None:
            processed = self.processor(chunk, sampling_rate=sr, return_tensors="pt")
            input_values = processed.input_values.squeeze(0)  # (chunk_samples,)
        else:
            # CNN14 backbone: normalise and feature-extract inside the model.
            input_values = torch.from_numpy(chunk)  # (chunk_samples,)

        frame_labels = self._frame_labels_for_chunk(
            fname, chunk_start, chunk_start + chunk_dur
        )  # (T_frames, num_classes) float32

        return {
            "input_values": input_values,
            "labels": frame_labels,
        }

    # ------------------------------------------------------------------
    # Frame label construction
    # ------------------------------------------------------------------

    def _frame_labels_for_chunk(
        self,
        fname: str,
        chunk_start: float,
        chunk_end: float,
    ) -> torch.Tensor:
        """
        Build a (T_frames, num_classes) float32 multi-label target.

        All frames default to 0.0 (background = all-zero row).  Frames that
        fall within an annotated event region have column seg.label set to 1.0.
        Multiple classes can be active on the same frame (multi-label).

        Frame-to-time mapping (approximate):
            frame i  ↔  chunk_start + i * frame_stride / sr  seconds

        Args:
            fname: Source recording filename key.
            chunk_start: Chunk start time in seconds.
            chunk_end: Chunk end time in seconds (= chunk_start + chunk_dur).

        Returns:
            Float32 tensor of shape (T_frames, num_classes), values in {0.0, 1.0}.
        """
        sr = self.cfg_data.sample_rate
        n_frames = self._n_frames_per_chunk
        num_classes = len(self.cfg_data.class_names)  # 4
        labels = torch.zeros(n_frames, num_classes, dtype=torch.float32)

        for seg in self.all_segments:
            if seg.source != fname:
                continue
            if seg.label < 0 or seg.label >= num_classes:
                continue
            if seg.end <= chunk_start or seg.start >= chunk_end:
                continue

            rel_start = max(0.0, seg.start - chunk_start)
            rel_end = min(chunk_end - chunk_start, seg.end - chunk_start)

            frame_start = int(rel_start * sr / self._frame_stride)
            frame_end = min(n_frames, int(rel_end * sr / self._frame_stride) + 1)

            if frame_start < frame_end:
                labels[frame_start:frame_end, seg.label] = 1.0

        return labels

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def frames_per_chunk(self) -> int:
        """Number of wav2vec2 output frames per chunk."""
        return self._n_frames_per_chunk

    def make_weighted_sampler(
        self,
        event_factor: float = 5.0,
    ) -> "torch.utils.data.WeightedRandomSampler":
        """
        Build a WeightedRandomSampler that oversamples event-containing chunks.

        Chunks that overlap at least one annotation get weight `event_factor`;
        pure-background chunks get weight 1.0.  With replacement=True this
        effectively increases the proportion of event examples per epoch without
        modifying the dataset or duplicating waveform data.

        Args:
            event_factor: Weight given to chunks containing at least one event.
                          5.0 → event chunks are 5× more likely to be drawn than
                          background-only chunks per sampling step.

        Returns:
            WeightedRandomSampler drawing len(dataset) samples per epoch.
        """
        from torch.utils.data import WeightedRandomSampler

        chunk_dur = self.cfg_sed.chunk_duration
        weights: list[float] = []
        n_event = 0

        for fname, chunk_start in self._chunks:
            chunk_end = chunk_start + chunk_dur
            has_event = any(
                seg.source == fname
                and seg.end > chunk_start
                and seg.start < chunk_end
                for seg in self.all_segments
            )
            if has_event:
                weights.append(event_factor)
                n_event += 1
            else:
                weights.append(1.0)

        n_bg = len(weights) - n_event
        logger.info(
            "WeightedSampler: %d event chunks (×%.1f) | %d bg-only chunks (×1.0)",
            n_event, event_factor, n_bg,
        )
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

    def make_hard_neg_sampler(
        self,
        hard_neg_ratio: float = 1.0,
    ) -> "torch.utils.data.SubsetRandomSampler":
        """
        Hard negative mining sampler.

        Keeps ALL event-containing chunks and randomly samples
        ``hard_neg_ratio × n_event`` background-only chunks.  This prevents
        the model from being overwhelmed by the large volume of silence while
        ensuring it sees every labelled event at least once per epoch.

        Args:
            hard_neg_ratio: Background chunks drawn per event chunk.
                            1.0 → equal numbers; 2.0 → twice as many bg as event.

        Returns:
            SubsetRandomSampler covering the selected chunk indices.
        """
        import random
        from torch.utils.data import SubsetRandomSampler

        chunk_dur = self.cfg_sed.chunk_duration
        event_indices: list[int] = []
        bg_indices: list[int] = []

        for idx, (fname, chunk_start) in enumerate(self._chunks):
            chunk_end = chunk_start + chunk_dur
            has_event = any(
                seg.source == fname
                and seg.end > chunk_start
                and seg.start < chunk_end
                for seg in self.all_segments
            )
            if has_event:
                event_indices.append(idx)
            else:
                bg_indices.append(idx)

        n_bg_target = int(hard_neg_ratio * len(event_indices))
        sampled_bg = (
            random.sample(bg_indices, n_bg_target)
            if len(bg_indices) > n_bg_target
            else bg_indices
        )

        all_indices = event_indices + sampled_bg
        random.shuffle(all_indices)

        logger.info(
            "HardNegSampler: %d event chunks + %d bg chunks (ratio=%.1f) → %d total/epoch",
            len(event_indices), len(sampled_bg), hard_neg_ratio, len(all_indices),
        )
        return SubsetRandomSampler(all_indices)
