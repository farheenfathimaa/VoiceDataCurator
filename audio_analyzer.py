"""
audio_analyzer.py
-----------------
Analyzes audio files for quality metrics:
  - Signal-to-Noise Ratio (SNR)
  - Silence ratio
  - Clipping detection
  - Duration validation
  - Sample rate consistency
"""

import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AudioMetrics:
    """Container for all computed quality metrics of a single audio file."""
    filepath: str
    duration: float = 0.0
    sample_rate: int = 0
    snr_db: float = 0.0
    silence_ratio: float = 0.0
    clipping_ratio: float = 0.0
    sample_rate_ok: bool = True
    load_error: Optional[str] = None
    quality_score: float = 0.0
    rejection_reasons: list = field(default_factory=list)

    @property
    def is_loadable(self) -> bool:
        return self.load_error is None


def load_audio(filepath: str, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
    """
    Load an audio file and resample to target_sr.

    Returns:
        (audio_array, sample_rate, error_message_or_None)
    """
    try:
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
        return audio, sr, None
    except Exception as e:
        error_msg = f"Failed to load audio: {e}"
        logger.warning(f"[load_audio] {filepath}: {error_msg}")
        return None, 0, error_msg


def compute_duration(audio: np.ndarray, sr: int) -> float:
    """Return duration of the audio clip in seconds."""
    return len(audio) / sr


def validate_duration(duration: float, min_sec: float, max_sec: float) -> Tuple[bool, str]:
    """
    Check if duration is within acceptable bounds.

    Returns:
        (is_valid, reason_string)
    """
    if duration < min_sec:
        return False, f"too_short ({duration:.2f}s < {min_sec}s)"
    if duration > max_sec:
        return False, f"too_long ({duration:.2f}s > {max_sec}s)"
    return True, ""


def check_sample_rate(sr: int, target_sr: int) -> Tuple[bool, str]:
    """
    Verify the file's original sample rate matches the target.
    Note: librosa resamples on load, so we read native SR separately.
    """
    if sr != target_sr:
        return False, f"wrong_sample_rate ({sr}Hz != {target_sr}Hz)"
    return True, ""


def compute_silence_ratio(audio: np.ndarray, sr: int, top_db: int = 40) -> float:
    """
    Compute the fraction of the audio that is silent.

    Uses librosa.effects.split to find non-silent intervals.
    silence_ratio = 1 - (total non-silent frames / total frames)
    """
    if len(audio) == 0:
        return 1.0
    try:
        non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
        non_silent_frames = sum(end - start for start, end in non_silent_intervals)
        silence_ratio = 1.0 - (non_silent_frames / len(audio))
        return float(np.clip(silence_ratio, 0.0, 1.0))
    except Exception as e:
        logger.warning(f"[compute_silence_ratio] Error: {e}")
        return 0.0


def compute_snr(audio: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> float:
    """
    Estimate Signal-to-Noise Ratio (SNR) in decibels.

    Method:
      - Use RMS energy of the overall signal as 'signal'
      - Estimate noise from the quietest 10% of frames
      - SNR(dB) = 20 * log10(signal_rms / noise_rms)
    """
    if len(audio) == 0:
        return 0.0
    try:
        rms_frames = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        if len(rms_frames) == 0:
            return 0.0

        # Signal power = mean of all RMS frames
        signal_rms = float(np.mean(rms_frames))

        # Noise estimate = mean of the bottom 10% quietest frames
        sorted_rms = np.sort(rms_frames)
        noise_sample_count = max(1, int(len(sorted_rms) * 0.10))
        noise_rms = float(np.mean(sorted_rms[:noise_sample_count]))

        if noise_rms < 1e-10:
            return 60.0  # Very clean audio — cap at 60 dB

        snr = 20.0 * np.log10(signal_rms / noise_rms + 1e-10)
        return float(np.clip(snr, -20.0, 80.0))

    except Exception as e:
        logger.warning(f"[compute_snr] Error: {e}")
        return 0.0


def detect_clipping(audio: np.ndarray, threshold: float = 0.98) -> float:
    """
    Detect the fraction of samples that are clipped (amplitude >= threshold).

    Returns:
        clipping_ratio in [0.0, 1.0]
    """
    if len(audio) == 0:
        return 0.0
    clipped = np.sum(np.abs(audio) >= threshold)
    return float(clipped / len(audio))


def compute_quality_score(
    snr_db: float,
    silence_ratio: float,
    clipping_ratio: float,
    duration: float,
    min_snr: float = 10.0,
    max_silence: float = 0.4,
    max_clipping: float = 0.01,
    min_dur: float = 1.0,
    max_dur: float = 30.0,
) -> float:
    """
    Compute a composite quality score in [0.0, 1.0].

    Higher is better. Components:
      - SNR score:       weight 0.40
      - Silence score:   weight 0.30
      - Clipping score:  weight 0.20
      - Duration score:  weight 0.10
    """
    # SNR: normalize between 0 and ~40 dB
    snr_score = float(np.clip(snr_db / 40.0, 0.0, 1.0))

    # Silence: invert so less silence = higher score
    silence_score = float(np.clip(1.0 - (silence_ratio / max(max_silence, 1e-6)), 0.0, 1.0))

    # Clipping: invert, scaled
    clipping_score = float(np.clip(1.0 - (clipping_ratio / max(max_clipping * 10, 1e-6)), 0.0, 1.0))

    # Duration: peaks at midpoint between min and max
    mid = (min_dur + max_dur) / 2.0
    dur_score = float(np.clip(1.0 - abs(duration - mid) / (max_dur - min_dur), 0.0, 1.0))

    score = (
        0.40 * snr_score
        + 0.30 * silence_score
        + 0.20 * clipping_score
        + 0.10 * dur_score
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def analyze_audio_file(filepath: str, config: dict) -> AudioMetrics:
    """
    Run the full quality analysis on a single audio file.

    Args:
        filepath: Path to the audio file.
        config:   The 'audio' section of config.yaml as a dict.

    Returns:
        AudioMetrics populated with all computed values.
    """
    audio_cfg = config.get("audio", {})
    min_dur = audio_cfg.get("min_duration_sec", 1.0)
    max_dur = audio_cfg.get("max_duration_sec", 30.0)
    target_sr = audio_cfg.get("target_sample_rate", 16000)
    min_snr = audio_cfg.get("min_snr_db", 10.0)
    max_silence = audio_cfg.get("max_silence_ratio", 0.4)
    max_clipping = audio_cfg.get("max_clipping_ratio", 0.01)

    metrics = AudioMetrics(filepath=filepath)

    # --- Load audio ---
    audio, sr, error = load_audio(filepath, target_sr=target_sr)
    if error:
        metrics.load_error = error
        metrics.rejection_reasons.append("load_error")
        return metrics

    metrics.sample_rate = sr

    # --- Check native sample rate (before resampling) ---
    try:
        info = sf.info(filepath)
        native_sr = info.samplerate
        sr_ok, sr_reason = check_sample_rate(native_sr, target_sr)
        metrics.sample_rate_ok = sr_ok
        if not sr_ok:
            logger.debug(f"[analyze] {Path(filepath).name}: {sr_reason}")
            # Sample rate mismatch is informational, not a rejection by itself
    except Exception:
        pass

    # --- Duration ---
    metrics.duration = round(compute_duration(audio, sr), 3)
    dur_ok, dur_reason = validate_duration(metrics.duration, min_dur, max_dur)
    if not dur_ok:
        metrics.rejection_reasons.append(dur_reason)

    # --- Silence ratio ---
    metrics.silence_ratio = round(compute_silence_ratio(audio, sr), 4)
    if metrics.silence_ratio > max_silence:
        metrics.rejection_reasons.append(
            f"high_silence ({metrics.silence_ratio:.2%} > {max_silence:.0%})"
        )

    # --- SNR ---
    metrics.snr_db = round(compute_snr(audio, sr), 2)
    if metrics.snr_db < min_snr:
        metrics.rejection_reasons.append(
            f"low_snr ({metrics.snr_db:.1f}dB < {min_snr}dB)"
        )

    # --- Clipping ---
    metrics.clipping_ratio = round(detect_clipping(audio), 6)
    if metrics.clipping_ratio > max_clipping:
        metrics.rejection_reasons.append(
            f"clipping ({metrics.clipping_ratio:.4%} > {max_clipping:.4%})"
        )

    # --- Composite quality score ---
    metrics.quality_score = compute_quality_score(
        snr_db=metrics.snr_db,
        silence_ratio=metrics.silence_ratio,
        clipping_ratio=metrics.clipping_ratio,
        duration=metrics.duration,
        min_snr=min_snr,
        max_silence=max_silence,
        max_clipping=max_clipping,
        min_dur=min_dur,
        max_dur=max_dur,
    )

    logger.debug(
        f"[analyze] {Path(filepath).name} | dur={metrics.duration:.2f}s "
        f"snr={metrics.snr_db:.1f}dB sil={metrics.silence_ratio:.2%} "
        f"clip={metrics.clipping_ratio:.4%} score={metrics.quality_score:.3f}"
    )
    return metrics
