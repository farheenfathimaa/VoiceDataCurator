# 🎙️ VoiceDataCurator

> **Automated multilingual speech dataset quality analyzer and curator — built in Python.**

A CLI + Streamlit pipeline that ingests raw audio files, scores each clip on signal quality, detects the spoken language using OpenAI Whisper, filters out low-quality samples, and exports a clean dataset manifest — all without needing ffmpeg installed.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔍 **Audio Quality Analysis** | SNR, silence ratio, clipping ratio, duration, sample-rate checks |
| 🌐 **Language Detection** | OpenAI Whisper via librosa loader — works on MP3/WAV/FLAC without ffmpeg |
| 📊 **Quality Scoring** | Composite 0–1 score per file for easy ranking and filtering |
| 🚦 **Smart Filtering** | Configurable thresholds; rejected files moved to a quarantine folder |
| 📄 **Dataset Manifest** | `dataset_manifest.csv` with all metrics per file, ready for ML pipelines |
| 📈 **Streamlit Dashboard** | Interactive dark-mode UI — charts, filters, per-file report, CSV export |
| ⚡ **Dry Run Mode** | Preview what would be accepted/rejected without touching any files |
| 🐳 **Docker Support** | One-command reproducible environment via `docker-compose` |

---

## 🗂️ Project Structure

```
VoiceDataCurator/
├── main.py                    # CLI entrypoint (argparse)
├── pipeline.py                # Core orchestrator
├── audio_analyzer.py          # SNR, silence, clipping, duration checks
├── language_detector.py       # Whisper language detection (librosa loader)
├── dashboard.py               # Streamlit visualization app
├── config.yaml                # All tunable thresholds
├── generate_samples.py        # Generate clean multilingual MP3 test clips (gTTS)
├── generate_bad_samples.py    # Generate degraded WAV samples for rejection testing
│
├── generate_transcripts.py    # [Research] Auto-generate .txt transcripts via Whisper
├── prepare_training_data.py   # [Research] Build HuggingFace datasets from manifest
├── finetune_whisper.py        # [Research] Fine-tune whisper-tiny on curated vs raw data
├── evaluate_wer.py            # [Research] Compute WER for all 3 models, output table
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── data/
    └── raw/               # Drop your audio files here (.mp3 .wav .flac .ogg)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/farheenfathimaa/VoiceDataCurator.git
cd VoiceDataCurator
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate Test Audio (optional)

```bash
# 30 clean multilingual clips (English, Hindi, Marathi, French, Spanish, German, Japanese, Arabic)
python generate_samples.py

# 16 intentionally degraded clips to test rejection logic
python generate_bad_samples.py
```

### 3. Run the Pipeline

```bash
# Full run with Whisper language detection
python main.py --input ./data/raw --verbose

# Fast run without language detection (audio quality checks only)
python main.py --input ./data/raw --no-language-detect

# Dry run — preview results without moving any files
python main.py --input ./data/raw --dry-run

# Custom config
python main.py --input ./data/raw --config my_config.yaml --whisper-model small
```

### 4. Launch Dashboard

```bash
python -m streamlit run dashboard.py
# Open http://localhost:8501
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
audio:
  min_duration_sec: 1.0        # Reject clips shorter than this
  max_duration_sec: 30.0       # Reject clips longer than this
  min_snr_db: 10.0             # Reject clips with SNR below this (dB)
  max_silence_ratio: 0.4       # Reject clips where >40% is silence
  max_clipping_ratio: 0.01     # Reject clips where >1% of samples are clipped
  target_sample_rate: 16000    # Flag clips recorded at wrong sample rate

language:
  whisper_model: "base"        # tiny | base | small | medium | large
  accepted_languages:          # Empty list = accept all
    - "en"
    - "hi"
    - "mr"
    - "fr"
    - "de"
    - "ar"
    - "es"
    - "ja"
  detect_confidence_threshold: 0.05   # Skip filter if Whisper confidence is too low
```

---

## 🔬 How It Works

```
data/raw/          Audio files
    │
    ▼
audio_analyzer.py  SNR · Silence · Clipping · Duration · Sample rate
    │
    ▼
language_detector.py  Whisper (loaded via librosa — no ffmpeg required)
    │
    ▼
pipeline.py        Filter · Move rejected · Write manifest
    │
    ├── output/dataset_manifest.csv
    ├── rejected/   (low-quality files)
    └── logs/       (run reports)
    │
    ▼
dashboard.py       Streamlit interactive visualization
```

---

## 📊 Dashboard

The Streamlit dashboard (`http://localhost:8501`) shows:

- **Dataset Overview** — total / accepted / rejected counts, accept rate, avg duration, language count
- **Language Distribution** — donut chart per detected language
- **Quality Score Distribution** — histogram, accepted vs rejected coloured
- **Avg Duration per Language** — bar chart
- **SNR Distribution by Language** — box plot
- **Per-File Quality Report** — sortable table with all metrics + CSV export

### Screenshots

![Dataset Overview](docs/screenshots/Overview.png)
*Overview cards and Language Distribution + Quality Score charts*

![Charts](docs/screenshots/Charts.png)
*Average Duration per Language and SNR Distribution by Language*

![Per-File Quality Report](docs/screenshots/ReportTable.png)
*Sortable per-file quality report with rejection reasons and CSV export*

---

## 🔬 Research Extension: Fine-tuning & WER Evaluation

### Research Question

> **Does data quality affect downstream Whisper fine-tuning performance?**
>
> VoiceDataCurator filters out high-noise, mostly-silent, clipped, and too-short/long audio.
> This extension proves that training on clean, curated data produces a lower Word Error Rate (WER)
> than training on the raw, unfiltered dataset — using identical model architecture and hyperparameters.

### Prerequisites

```bash
# Install fine-tuning dependencies
pip install transformers>=4.37.0 datasets>=2.18.0 evaluate>=0.4.1 jiwer>=3.0.3 accelerate>=0.27.0

# Run the main pipeline first so output/dataset_manifest.csv exists
python main.py --input ./data/raw
```

> **Note on Transcripts:** Each audio file needs a matching `.txt` transcript file in `data/raw/`.
> Step 1 auto-generates these using Whisper-tiny itself (demo/proof-of-concept setup).
> For production use, replace these with **human-verified transcripts** for more meaningful WER results.

### How to Run (in order)

```bash
# Step 1 — Auto-generate .txt transcripts for all audio files (~1 min)
python generate_transcripts.py

# Step 2 — Build HuggingFace datasets: curated (accepted-only) and raw (all files)
python prepare_training_data.py

# Step 3 — Fine-tune whisper-tiny on both sets (~10–60 min on CPU for 30–40 clips)
python finetune_whisper.py

# Step 4 — Evaluate WER for all 3 models and print comparison table
python evaluate_wer.py
```

### WER Results

> **Fill in your results after running `python evaluate_wer.py`:**

| Model | WER | vs Baseline |
|---|---|---|
| `openai/whisper-tiny` (zero-shot) | __%  | baseline |
| Fine-tuned on **raw** data | __%  | — |
| Fine-tuned on **curated** data | __%  | — |

Results are also saved automatically to `wer_results.json`.

### Interpreting the Results

- **Lower WER = better** — 0% is perfect, 100%+ means more errors than words
- **Zero-shot baseline**: built-in Whisper capability without any fine-tuning
- **Raw fine-tuned**: effect of more data, including noisy/poor-quality clips
- **Curated fine-tuned**: effect of VoiceDataCurator's quality filtering on model performance
- If curated WER < raw WER, the pipeline has **proven its value** quantitatively

---

## 🐳 Docker

```bash
# Run pipeline
docker-compose run pipeline --input /data/raw

# Run dashboard (accessible on http://localhost:8501)
docker-compose up dashboard
```

---

## 📦 Requirements

| Package | Purpose |
|---|---|
| `openai-whisper` | Language detection |
| `librosa` | Audio loading (MP3/WAV/FLAC without ffmpeg) |
| `soundfile` | WAV read/write |
| `numpy` | Numerical processing |
| `pandas` | Manifest CSV handling |
| `streamlit` | Dashboard UI |
| `plotly` | Interactive charts |
| `tqdm` | Progress bars |
| `pyyaml` | Config parsing |
| `gTTS` | Test audio generation |

---

## 📝 Output Files

| File | Description |
|---|---|
| `output/dataset_manifest.csv` | Per-file metrics: language, SNR, silence ratio, quality score, status |
| `logs/run_report_<timestamp>.txt` | Human-readable summary with rejection breakdown |
| `rejected/<filename>` | Audio files that failed quality or language checks |

---

## 🛠️ CLI Reference

```
python main.py [OPTIONS]

Options:
  --input,  -i   Path to folder with raw audio files
  --output, -o   Output directory (default: ./output)
  --rejected     Directory for rejected files (default: ./rejected)
  --config, -c   Path to config YAML (default: config.yaml)
  --dry-run      Preview without moving any files
  --verbose, -v  Debug-level logging
  --whisper-model  Whisper model size: tiny|base|small|medium|large
  --no-language-detect  Skip Whisper (audio quality checks only, much faster)
```

---

## 💡 Rejection Reasons

| Code | Meaning |
|---|---|
| `too_short` | Clip under `min_duration_sec` |
| `too_long` | Clip over `max_duration_sec` |
| `low_snr` | SNR below `min_snr_db` |
| `high_silence` | Silence above `max_silence_ratio` |
| `clipping` | Clipping above `max_clipping_ratio` |
| `wrong_sample_rate` | Sample rate doesn't match `target_sample_rate` |
| `rejected_language` | Detected language not in `accepted_languages` |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with Python · OpenAI Whisper · librosa · Streamlit · Plotly</p>
