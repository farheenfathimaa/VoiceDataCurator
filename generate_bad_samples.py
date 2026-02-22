"""
generate_bad_samples.py
-----------------------
Generates intentionally low-quality audio files to test that the
VoiceDataCurator pipeline correctly identifies and rejects bad samples.

Degradation types produced:
  1. HIGH NOISE    — very loud white noise drowns out the signal (low SNR)
  2. MOSTLY SILENT — 90%+ of the clip is silence (high silence ratio)
  3. CLIPPED       — audio amplitude saturated/distorted (clipping > threshold)
  4. TOO SHORT     — clip under 1 second (fails duration min)
  5. TOO LONG      — clip over 30 seconds (fails duration max)
  6. WRONG SR      — recorded at 8kHz instead of expected 16kHz

Saved as WAV files to data/raw/ alongside the clean gTTS MP3s.

Requirements: numpy, soundfile (both already in requirements.txt)
Usage:
    python generate_bad_samples.py
"""

import numpy as np
import soundfile as sf
from pathlib import Path

OUTPUT_DIR = Path("./data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000  # Standard sample rate

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_speech_like(duration_sec: float, sr: int = SR) -> np.ndarray:
    """
    Generate a synthetic 'speech-like' signal: a mix of sine waves
    at typical human voice frequencies (100–300 Hz fundamental + harmonics).
    Amplitude normalized to ~0.3.
    """
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental + harmonics mimicking voiced speech
    signal = (
        0.40 * np.sin(2 * np.pi * 150 * t)
        + 0.25 * np.sin(2 * np.pi * 300 * t)
        + 0.15 * np.sin(2 * np.pi * 450 * t)
        + 0.10 * np.sin(2 * np.pi * 600 * t)
        + 0.05 * np.sin(2 * np.pi * 900 * t)
    )
    # Normalize to 0.3 peak
    peak = np.max(np.abs(signal)) + 1e-9
    return (signal / peak * 0.3).astype(np.float32)


def save_wav(audio: np.ndarray, filename: str, sr: int = SR) -> None:
    path = OUTPUT_DIR / filename
    sf.write(str(path), audio, sr, subtype="PCM_16")
    size_kb = path.stat().st_size / 1024
    print(f"    [saved] {filename}  ({size_kb:.1f} KB, {len(audio)/sr:.2f}s, {sr}Hz)")


# ── 1. HIGH NOISE — low SNR (signal buried under noise) ──────────────────────
print("\n[1] HIGH NOISE samples (SNR too low — will be REJECTED)")
for i in range(1, 4):
    duration = np.random.uniform(3.0, 8.0)
    speech = make_speech_like(duration)
    # Noise amplitude 5-10× louder than speech → SNR will be ~0-5 dB
    noise = np.random.normal(0, 0.6, speech.shape).astype(np.float32)
    audio = np.clip(speech + noise, -1.0, 1.0)
    save_wav(audio, f"bad_noisy_{i:02d}.wav")

# ── 2. MOSTLY SILENT — high silence ratio ────────────────────────────────────
print("\n[2] MOSTLY SILENT samples (silence_ratio > 40% — will be REJECTED)")
for i in range(1, 4):
    total_dur = np.random.uniform(4.0, 10.0)
    n_total = int(SR * total_dur)
    audio = np.zeros(n_total, dtype=np.float32)
    # Add only a tiny burst of speech (5% of duration)
    speech_dur = total_dur * 0.05
    speech = make_speech_like(speech_dur)
    start = int(SR * 0.5)
    end = start + len(speech)
    if end <= n_total:
        audio[start:end] = speech * 0.2
    save_wav(audio, f"bad_silent_{i:02d}.wav")

# ── 3. CLIPPED — severe amplitude saturation ─────────────────────────────────
print("\n[3] CLIPPED samples (clipping_ratio > 1% — will be REJECTED)")
for i in range(1, 4):
    duration = np.random.uniform(3.0, 7.0)
    speech = make_speech_like(duration)
    # Amplify massively then hard-clip → large fraction at ±1.0
    audio = np.clip(speech * 15.0, -1.0, 1.0).astype(np.float32)
    save_wav(audio, f"bad_clipped_{i:02d}.wav")

# ── 4. TOO SHORT — under minimum duration ────────────────────────────────────
print("\n[4] TOO SHORT samples (< 1.0s — will be REJECTED)")
for i in range(1, 4):
    duration = np.random.uniform(0.1, 0.8)
    speech = make_speech_like(duration)
    noise = np.random.normal(0, 0.02, speech.shape).astype(np.float32)
    audio = np.clip(speech + noise, -1.0, 1.0)
    save_wav(audio, f"bad_tooshort_{i:02d}.wav")

# ── 5. TOO LONG — over maximum duration ──────────────────────────────────────
print("\n[5] TOO LONG samples (> 30.0s — will be REJECTED)")
for i in range(1, 3):
    duration = np.random.uniform(35.0, 50.0)
    speech = make_speech_like(duration)
    noise = np.random.normal(0, 0.02, speech.shape).astype(np.float32)
    audio = np.clip(speech + noise, -1.0, 1.0)
    save_wav(audio, f"bad_toolong_{i:02d}.wav")

# ── 6. COMBINED BAD — noisy AND mostly silent ────────────────────────────────
print("\n[6] COMBINED failures (noisy + silent — multiple rejection reasons)")
for i in range(1, 3):
    duration = np.random.uniform(4.0, 8.0)
    n_total = int(SR * duration)
    # Mostly silence + low-level noise
    noise = np.random.normal(0, 0.3, n_total).astype(np.float32)
    audio = np.zeros(n_total, dtype=np.float32)
    # Only 5% actual speech
    speech_dur = duration * 0.05
    speech = make_speech_like(speech_dur)
    start = int(SR * 0.2)
    end = start + len(speech)
    if end <= n_total:
        audio[start:end] = speech * 0.15
    # Add significant noise everywhere
    audio = np.clip(audio + noise, -1.0, 1.0)
    save_wav(audio, f"bad_combined_{i:02d}.wav")


# ── Summary ───────────────────────────────────────────────────────────────────
bad_files = list(OUTPUT_DIR.glob("bad_*.wav"))
good_files = list(OUTPUT_DIR.glob("*.mp3"))

print("\n" + "=" * 52)
print(f"  Low-quality samples : {len(bad_files)} WAV files")
print(f"  Clean samples       : {len(good_files)} MP3 files")
print(f"  Total in data/raw   : {len(bad_files) + len(good_files)} files")
print("=" * 52)
print("\nExpected pipeline results:")
print("  bad_noisy_*    -> REJECTED (low_snr)")
print("  bad_silent_*   -> REJECTED (high_silence)")
print("  bad_clipped_*  -> REJECTED (clipping)")
print("  bad_tooshort_* -> REJECTED (too_short)")
print("  bad_toolong_*  -> REJECTED (too_long)")
print("  bad_combined_* -> REJECTED (low_snr + high_silence)")
print("  en/hi/mr/fr... -> ACCEPTED (clean speech)")
print("\nNow run:")
print("  python main.py --input ./data/raw --verbose\n")
