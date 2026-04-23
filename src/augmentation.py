"""
src/augmentation.py — Data augmentation for bowel sound waveforms.

All functions operate on raw float32 numpy waveforms at 16 kHz and return
a waveform of the same length as the input.

Augmentation is applied only during training (never on val/test sets).
The same segment receives different random augmentations on every epoch
because parameters are re-sampled on each __getitem__ call.
"""

from __future__ import annotations

import numpy as np
import librosa


def pitch_shift(
    waveform: np.ndarray,
    n_steps: float,
    sr: int = 16_000,
) -> np.ndarray:
    """
    Shift the pitch by `n_steps` semitones without changing duration.

    Args:
        waveform: float32 array shape (n_samples,).
        n_steps: semitones to shift (negative = lower pitch).
        sr: sample rate in Hz.

    Returns:
        float32 array shape (n_samples,).
    """
    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)


def additive_gaussian_noise(
    waveform: np.ndarray,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to simulate sensor/recording floor noise.

    Args:
        waveform: float32 array shape (n_samples,).
        std: Standard deviation of the noise (in waveform amplitude units).
             Typical range: 0.001–0.02 for 16-bit-like normalised audio.
        rng: numpy random Generator.

    Returns:
        float32 array shape (n_samples,).
    """
    noise = rng.normal(0.0, std, size=len(waveform)).astype(np.float32)
    return (waveform + noise).astype(np.float32)


def gain_jitter(
    waveform: np.ndarray,
    gain: float,
) -> np.ndarray:
    """
    Scale the waveform amplitude by a random gain factor.

    Simulates the large volume variation seen across patients and recording
    positions (e.g. loose vs. tight stethoscope contact).

    Args:
        waveform: float32 array shape (n_samples,).
        gain: Multiplicative gain factor, e.g. 0.5–2.0.

    Returns:
        float32 array shape (n_samples,).
    """
    return (waveform * gain).astype(np.float32)


def background_noise_mix(
    waveform: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Mix a noise sample into the waveform at a specified signal-to-noise ratio.

    The noise is trimmed or zero-padded to match the waveform length, then
    scaled so that the resulting SNR equals `snr_db` dB relative to the
    RMS of the clean signal.

    Args:
        waveform: float32 array shape (n_samples,) — the clean signal.
        noise: float32 array of any length — the noise to mix in.
        snr_db: Target SNR in dB (e.g. 10 dB = moderately noisy, 30 dB = faint noise).

    Returns:
        float32 array shape (n_samples,).
    """
    n = len(waveform)
    # Trim or zero-pad noise to match waveform length.
    if len(noise) >= n:
        noise = noise[:n]
    else:
        noise = np.concatenate([noise, np.zeros(n - len(noise), dtype=np.float32)])

    signal_rms = np.sqrt(np.mean(waveform ** 2))
    noise_rms = np.sqrt(np.mean(noise ** 2))

    if noise_rms < 1e-8 or signal_rms < 1e-8:
        return waveform

    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    return (waveform + scaled_noise).astype(np.float32)


def augment_waveform(
    waveform: np.ndarray,
    sr: int,
    p_pitch_shift: float,
    pitch_shift_range: tuple[float, float],
    p_gaussian_noise: float,
    gaussian_noise_std_range: tuple[float, float],
    p_gain_jitter: float,
    gain_jitter_range: tuple[float, float],
    p_background_noise: float,
    background_noise_snr_db_range: tuple[float, float],
    noise_pool: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    use_pitch_shift: bool = True,
    use_gaussian_noise: bool = True,
    use_gain_jitter: bool = True,
    use_background_noise: bool = True,
) -> np.ndarray:
    """
    Apply a stochastic combination of pitch-shift, Gaussian noise, gain jitter,
    and background noise mixing.

    Each augmentation is independently gated by its probability and its
    use_* flag.  Background noise mixing is skipped if noise_pool is None.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = waveform.copy()

    if use_pitch_shift and rng.random() < p_pitch_shift:
        n_steps = float(rng.uniform(*pitch_shift_range))
        result = pitch_shift(result, n_steps=n_steps, sr=sr)

    if use_gaussian_noise and rng.random() < p_gaussian_noise:
        std = float(rng.uniform(*gaussian_noise_std_range))
        result = additive_gaussian_noise(result, std=std, rng=rng)

    if use_gain_jitter and rng.random() < p_gain_jitter:
        g = float(rng.uniform(*gain_jitter_range))
        result = gain_jitter(result, gain=g)

    if use_background_noise and noise_pool is not None and rng.random() < p_background_noise:
        snr_db = float(rng.uniform(*background_noise_snr_db_range))
        result = background_noise_mix(result, noise=noise_pool, snr_db=snr_db)

    return result
