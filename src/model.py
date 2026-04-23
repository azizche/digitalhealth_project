"""
src/model.py — SED model backbones.

Two backbones are supported, both returning (B, T_frames, num_classes) logits:

  wav2vec2  BowelSoundWav2Vec2Model  — facebook/wav2vec2-base transformer encoder.
            Frame stride = 320 samples (~50 fps at 16 kHz).
            Input: zero-mean / unit-variance normalised raw waveform.

  cnn14     BowelSoundCNN14Model     — CNN14 from PANNS, modified for SED.
            Mel spectrogram (hop=320) extracted internally; only the
            frequency axis is pooled so temporal resolution is preserved.
            Frame stride = 320 samples (~50 fps at 16 kHz).
            Input: raw waveform (log-mel + BN normalisation inside the model).

Use build_model(cfg) to get the right class.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError):
    _TORCHAUDIO_AVAILABLE = False

from config import ModelConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PANNs checkpoint auto-download
# ---------------------------------------------------------------------------

_PANNS_DLMAX_URL = (
    "https://zenodo.org/record/3987831/files/"
    "Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
)
_PANNS_DLMAX_FILENAME = "Cnn14_DecisionLevelMax_mAP=0.385.pth"
_PANNS_CACHE_DIR = Path.home() / ".cache" / "panns"


def _get_panns_dlmax_checkpoint() -> str:
    """Return path to Cnn14_DecisionLevelMax weights, downloading if needed."""
    _PANNS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _PANNS_CACHE_DIR / _PANNS_DLMAX_FILENAME
    if not local_path.exists():
        logger.info(
            "Downloading PANNs Cnn14_DecisionLevelMax weights to %s …", local_path
        )
        torch.hub.download_url_to_file(_PANNS_DLMAX_URL, str(local_path))
        logger.info("Download complete.")
    else:
        logger.info("Using cached PANNs weights: %s", local_path)
    return str(local_path)


# ---------------------------------------------------------------------------
# Wav2Vec2 backbone
# ---------------------------------------------------------------------------

class BowelSoundWav2Vec2Model(nn.Module):
    """
    Wav2Vec2-based frame-level SED model.

    Input:  (B, n_samples) normalised waveform (Wav2Vec2Processor output).
    Output: (B, T_frames, num_classes) raw logits.
    """

    FRAME_STRIDE = 320

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(cfg.base_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(768, cfg.num_classes)

        if cfg.freeze_feature_extractor:
            self._freeze_feature_extractor()

    def _freeze_feature_extractor(self) -> None:
        for p in self.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_feature_extractor(self) -> None:
        for p in self.wav2vec2.feature_extractor.parameters():
            p.requires_grad = True

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden = self.wav2vec2(input_values=input_values).last_hidden_state
        hidden = self.dropout(hidden)
        return self.classifier(hidden)

    def count_parameters(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}


# ---------------------------------------------------------------------------
# CNN14 backbone (PANNS-style, modified for SED)
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """
    Standard CNN14 conv block with frequency-only pooling.

    Original CNN14 pools both time and frequency (kernel 2×2).  Here only
    the frequency axis is pooled (kernel 1×2) so the time dimension passes
    through unchanged, preserving the ~50 fps temporal resolution needed for
    frame-level event detection.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        # Pool frequency only (dim=3 in NCTF layout) — time dim unchanged.
        return F.avg_pool2d(x, kernel_size=(1, 2))


class BowelSoundCNN14Model(nn.Module):
    """
    CNN14-based frame-level SED model.

    Mel spectrogram extraction is performed inside forward() so the model
    accepts raw waveforms exactly like the Wav2Vec2 backbone.

    Mel parameters:
      n_fft=1024, hop_length=320, n_mels=64, sr=16000
      → frame stride = 320 samples = ~50 fps, matching wav2vec2.

    Architecture (input shape annotations in NCTF = batch/channel/time/freq):
      Raw waveform (B, n_samples)
        → MelSpectrogram → log → BN0  :  (B, 1, T, 64)
        → ConvBlock1  pool freq ×2     :  (B,   64, T, 32)
        → ConvBlock2  pool freq ×2     :  (B,  128, T, 16)
        → ConvBlock3  pool freq ×2     :  (B,  256, T,  8)
        → ConvBlock4  pool freq ×2     :  (B,  512, T,  4)
        → ConvBlock5  pool freq ×2     :  (B, 1024, T,  2)
        → ConvBlock6  pool freq ×2     :  (B, 2048, T,  1)
        → squeeze freq                 :  (B, 2048, T)
        → permute                      :  (B, T, 2048)
        → Dropout → Linear(2048, C)    :  (B, T, num_classes)

    Pretrained weights:
      Pass cfg.panns_checkpoint = "/path/to/Cnn14_*.pth" to load weights
      downloaded from https://zenodo.org/record/3987831 (CNN14 checkpoint).
      Layers that match by name and shape are loaded; the final classifier
      layer (different output size) is always re-initialised.
    """

    FRAME_STRIDE = 320
    # Mel parameters — hop=320 keeps frame rate identical to wav2vec2.
    _N_FFT = 1024
    _HOP = 320
    _N_MELS = 64
    _SR = 16_000

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if not _TORCHAUDIO_AVAILABLE:
            raise ImportError(
                "torchaudio is required for the cnn14 backbone. "
                "Install a version matching your PyTorch/CUDA — see fix below."
            )

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._SR,
            n_fft=self._N_FFT,
            hop_length=self._HOP,
            n_mels=self._N_MELS,
            power=2.0,
        )
        self.bn0 = nn.BatchNorm2d(1)

        self.block1 = _ConvBlock(1, 64)
        self.block2 = _ConvBlock(64, 128)
        self.block3 = _ConvBlock(128, 256)
        self.block4 = _ConvBlock(256, 512)
        self.block5 = _ConvBlock(512, 1024)
        self.block6 = _ConvBlock(1024, 2048)

        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(2048, cfg.num_classes)

        if cfg.panns_checkpoint:
            self._load_panns_weights(cfg.panns_checkpoint)

        if cfg.freeze_feature_extractor:
            self._freeze_feature_extractor()

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _freeze_feature_extractor(self) -> None:
        for block in (self.mel, self.bn0,
                      self.block1, self.block2, self.block3,
                      self.block4, self.block5, self.block6):
            for p in block.parameters():
                p.requires_grad = False

    def unfreeze_feature_extractor(self) -> None:
        for block in (self.mel, self.bn0,
                      self.block1, self.block2, self.block3,
                      self.block4, self.block5, self.block6):
            for p in block.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def _load_panns_weights(self, checkpoint_path: str) -> None:
        """
        Load matching weights from an official CNN14 PANNS checkpoint.

        The official checkpoint uses the naming convention
        `conv_block{N}.conv1.weight` while this model uses `block{N}.conv1.weight`.
        The remapping below handles that difference.  Layers whose shapes differ
        (e.g. the AudioSet 527-class head) are silently skipped.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Official checkpoints wrap weights under a "model" key.
        state = ckpt.get("model", ckpt)

        # Remap conv_block{N} → block{N}
        remapped: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            new_k = k
            for n in range(1, 7):
                new_k = new_k.replace(f"conv_block{n}.", f"block{n}.")
            remapped[new_k] = v

        own = self.state_dict()
        matched = {k: v for k, v in remapped.items()
                   if k in own and v.shape == own[k].shape}
        own.update(matched)
        self.load_state_dict(own)
        logger.info(
            "CNN14: loaded %d / %d parameter tensors from %s",
            len(matched), len(own), checkpoint_path,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_values: (B, n_samples) raw float32 waveform.
        Returns:
            logits: (B, T_frames, num_classes).
        """
        # Mel spectrogram: torchaudio returns (B, n_mels, T).
        x = self.mel(input_values)                    # (B, 64, T)
        x = torch.log(x.clamp(min=1e-7))
        x = x.permute(0, 2, 1).unsqueeze(1)           # (B, 1, T, 64) — NCTF
        x = self.bn0(x)

        x = self.block1(x)   # (B,   64, T, 32)
        x = self.block2(x)   # (B,  128, T, 16)
        x = self.block3(x)   # (B,  256, T,  8)
        x = self.block4(x)   # (B,  512, T,  4)
        x = self.block5(x)   # (B, 1024, T,  2)
        x = self.block6(x)   # (B, 2048, T,  1)

        x = x.squeeze(-1)    # (B, 2048, T)
        x = x.permute(0, 2, 1)  # (B, T, 2048)

        x = self.dropout(x)
        return self.classifier(x)   # (B, T, num_classes)

    def count_parameters(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}


# Keep the old name as an alias so existing imports don't break.
BowelSoundSEDModel = BowelSoundWav2Vec2Model


# ---------------------------------------------------------------------------
# CNN14 DecisionLevelMax backbone
# ---------------------------------------------------------------------------

class _ConvBlock2D(nn.Module):
    """
    Standard CNN14 conv block with configurable 2D avg-pooling.

    Unlike _ConvBlock (frequency-only pooling), this pools both time and
    frequency — matching the original Cnn14_DecisionLevelMax architecture.
    Temporal resolution is recovered later via F.interpolate.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 pool_size: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool_size = pool_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if self.pool_size != (1, 1):
            x = F.avg_pool2d(x, kernel_size=self.pool_size)
        return x


class BowelSoundCNN14DecisionLevelMaxModel(nn.Module):
    """
    Cnn14_DecisionLevelMax-based frame-level SED model.

    Mirrors the PANNs Cnn14_DecisionLevelMax architecture:
      - bn0 normalises each mel bin (BatchNorm2d(n_mels=64)) — same as PANNs.
      - Blocks 1-5: 2×2 avg-pool (time and frequency halved each block).
      - Block 6: (1,1) pool (no spatial reduction).
      - Mean over residual frequency dimension → shape (B, 2048, T/32).
      - 1D temporal smoothing: max_pool1d + avg_pool1d.
      - fc1 (2048 → 2048, ReLU) — pretrained from PANNs.
      - Interpolate back to T_mel (nearest-neighbour, same as PANNs).
      - Classifier head → (B, T_mel, num_classes) logits.

    Checkpoint compatibility:
      Designed for Cnn14_DecisionLevelMax_mAP=*.pth from
      https://zenodo.org/record/3987831.  Key remapping: conv_block{N} →
      block{N}.  bn0 (64 features) and fc1 (2048×2048) load directly.
      The AudioSet 527-class fc_audioset head is always skipped.

    Frame stride = 320 samples at 16 kHz, identical to wav2vec2 and cnn14.
    """

    FRAME_STRIDE = 320
    _N_FFT = 1024
    _HOP = 320
    _N_MELS = 64
    _SR = 16_000

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._SR,
            n_fft=self._N_FFT,
            hop_length=self._HOP,
            n_mels=self._N_MELS,
            power=2.0,
        )
        # PANNs-style BN: one running stat per mel bin.
        # Input shape when applied: (B, n_mels, T, 1).
        self.bn0 = nn.BatchNorm2d(self._N_MELS)

        self.block1 = _ConvBlock2D(1, 64)                       # pool (2,2)
        self.block2 = _ConvBlock2D(64, 128)                     # pool (2,2)
        self.block3 = _ConvBlock2D(128, 256)                    # pool (2,2)
        self.block4 = _ConvBlock2D(256, 512)                    # pool (2,2)
        self.block5 = _ConvBlock2D(512, 1024)                   # pool (2,2)
        self.block6 = _ConvBlock2D(1024, 2048, pool_size=(1, 1))  # no pool

        self.fc1 = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(2048, cfg.num_classes)

        ckpt_path = cfg.panns_checkpoint or _get_panns_dlmax_checkpoint()
        self._load_panns_weights(ckpt_path)

        if cfg.freeze_feature_extractor:
            self._freeze_feature_extractor()

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _freeze_feature_extractor(self) -> None:
        for m in (self.mel, self.bn0,
                  self.block1, self.block2, self.block3,
                  self.block4, self.block5, self.block6,
                  self.fc1):
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_feature_extractor(self) -> None:
        for m in (self.mel, self.bn0,
                  self.block1, self.block2, self.block3,
                  self.block4, self.block5, self.block6,
                  self.fc1):
            for p in m.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def _load_panns_weights(self, checkpoint_path: str) -> None:
        """
        Load compatible weights from a Cnn14_DecisionLevelMax checkpoint.

        Key remapping conv_block{N} → block{N} matches the PANNs naming
        convention.  Layers whose shapes differ (fc_audioset, att_block)
        are silently skipped.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt)

        remapped: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            new_k = k
            for n in range(1, 7):
                new_k = new_k.replace(f"conv_block{n}.", f"block{n}.")
            remapped[new_k] = v

        own = self.state_dict()
        matched = {k: v for k, v in remapped.items()
                   if k in own and v.shape == own[k].shape}
        own.update(matched)
        self.load_state_dict(own)
        logger.info(
            "CNN14 (DecisionLevelMax): loaded %d / %d tensors from %s",
            len(matched), len(own), checkpoint_path,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_values: (B, n_samples) raw float32 waveform.
        Returns:
            logits: (B, T_mel, num_classes).
        """
        # Mel spectrogram: torchaudio returns (B, n_mels, T_mel)
        x = self.mel(input_values)
        x = torch.log(x.clamp(min=1e-7))
        T_mel = x.shape[2]  # save for interpolation target

        # PANNs bn0: normalise per mel bin.
        # BN2d expects channel as dim 1 → reshape to (B, n_mels, T_mel, 1).
        x = x.unsqueeze(-1)                        # (B, n_mels, T_mel, 1)
        x = self.bn0(x)
        x = x.squeeze(-1)                           # (B, n_mels, T_mel)
        x = x.permute(0, 2, 1).unsqueeze(1)         # (B, 1, T_mel, n_mels)

        # ConvBlocks with 2D pooling — halves T and freq each block (blocks 1-5).
        x = F.dropout(self.block1(x), p=0.2, training=self.training)  # (B,   64, T/2,  32)
        x = F.dropout(self.block2(x), p=0.2, training=self.training)  # (B,  128, T/4,  16)
        x = F.dropout(self.block3(x), p=0.2, training=self.training)  # (B,  256, T/8,   8)
        x = F.dropout(self.block4(x), p=0.2, training=self.training)  # (B,  512, T/16,  4)
        x = F.dropout(self.block5(x), p=0.2, training=self.training)  # (B, 1024, T/32,  2)
        x = F.dropout(self.block6(x), p=0.2, training=self.training)  # (B, 2048, T/32,  2)

        # Mean over residual frequency dimension.
        x = torch.mean(x, dim=3)                   # (B, 2048, T/32)

        # 1D temporal smoothing (replicates PANNs decision-level aggregation).
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)

        x = x.transpose(1, 2)                      # (B, T/32, 2048)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # Interpolate back to original mel-frame resolution.
        x = x.transpose(1, 2)                      # (B, 2048, T/32)
        x = F.interpolate(x, size=T_mel, mode="nearest")
        x = x.transpose(1, 2)                      # (B, T_mel, 2048)

        return self.classifier(x)                  # (B, T_mel, num_classes)

    def count_parameters(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> nn.Module:
    """Return the correct model for cfg.backbone."""
    if cfg.backbone == "wav2vec2":
        return BowelSoundWav2Vec2Model(cfg)
    elif cfg.backbone == "cnn14":
        return BowelSoundCNN14Model(cfg)
    elif cfg.backbone == "cnn14_dlmax":
        return BowelSoundCNN14DecisionLevelMaxModel(cfg)
    else:
        raise ValueError(
            f"Unknown backbone '{cfg.backbone}'. "
            "Choose 'wav2vec2', 'cnn14', or 'cnn14_dlmax'."
        )
