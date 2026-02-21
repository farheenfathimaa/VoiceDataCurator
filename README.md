# 🎙️ VoiceDataCurator

<div align="center">

**Multilingual Speech Quality Analyzer & Dataset Curator**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A production-grade CLI + Streamlit pipeline for cleaning, scoring, and curating raw speech datasets for ML model training*

</div>

---

## 📌 Overview

**VoiceDataCurator** is an end-to-end audio dataset quality control and curation tool designed for speech AI teams and researchers. It ingests raw audio files (WAV/MP3), runs automated quality checks, detects the spoken language, scores each sample, filters out low-quality clips, and produces a clean, curated dataset — all with a rich visual dashboard for monitoring.

Whether you're building an ASR model, a TTS system, or a multilingual voice assistant, VoiceDataCurator ensures your training data meets the bar before it ever reaches your model.

---

## ✨ Features

### 🔬 Audio Quality Analyzer
- **Silence Ratio Detection** — flags clips with excessive silence
- **Signal-to-Noise Ratio (SNR)** — quantifies background noise levels
- **Clipping Detection** — identifies audio distortion from recording peaks
- **Duration Validation** — rejects clips that are too short or too long
- **Sample Rate Consistency Check** — ensures all files match target sample rate

### 🌍 Language Detection & Tagging
- Automatic language identification per audio clip using **OpenAI Whisper**
- Tags each clip with its detected language code (e.g., `en`, `hi`, `fr`, `ar`)
- Builds a **multilingual-aware dataset** out of the box

### ⚙️ Automated Filtering Pipeline
- Configurable quality thresholds via `config.yaml`
  - Min SNR, Max silence %, accepted languages list, duration bounds
- Low-quality files are moved to a `rejected/` folder automatically
- Each rejection includes a **reason log** for full auditability
- `--dry-run` flag shows what *would* be filtered without moving any files

### 📊 Streamlit Dashboard
- **Language distribution** — interactive pie chart
- **Quality score histogram** — see the spread of your dataset
- **Accepted vs Rejected** — live counts and ratios
- **Average duration per language** — bar chart
- **Per-file quality report** — sortable, filterable table

### 📦 Export & Reporting
- Outputs `dataset_manifest.csv` with:
  - `filename`, `language`, `duration`, `snr`, `silence_ratio`, `quality_score`, `status`, `rejection_reason`
- Generates a **run report log** with timestamps and pipeline summary
- Fully reproducible runs with config versioning

---

## 🗂️ Project Structure

```
VoiceDataCurator/
├── main.py                  # CLI entrypoint
├── pipeline.py              # Orchestration logic
├── audio_analyzer.py        # SNR, silence, clipping, duration checks
├── language_detector.py     # Whisper-based language detection
├── dashboard.py             # Streamlit web dashboard
├── config.yaml              # Quality threshold configuration
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker containerization
├── docker-compose.yml       # Multi-service compose setup
├── logs/                    # Pipeline run logs
├── rejected/                # Files that failed quality checks
└── output/
    └── dataset_manifest.csv # Final curated dataset manifest
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- `ffmpeg` installed and available in PATH (required by Whisper & librosa)
- Docker (optional, for containerized runs)

### Installation

```bash
# Clone the repository
git clone https://github.com/farheenfathimaa/VoiceDataCurator.git
cd VoiceDataCurator

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml` to set your quality thresholds:

```yaml
audio:
  min_duration_sec: 1.0
  max_duration_sec: 30.0
  target_sample_rate: 16000
  min_snr_db: 10.0
  max_silence_ratio: 0.4
  max_clipping_ratio: 0.01

language:
  accepted_languages: ["en", "hi", "fr", "de", "ar", "es"]
  detect_confidence_threshold: 0.6

pipeline:
  input_dir: "./data/raw"
  output_dir: "./output"
  rejected_dir: "./rejected"
  log_dir: "./logs"
```

---

## 🖥️ Usage

### CLI — Run the Pipeline

```bash
# Standard run
python main.py --input ./data/raw --output ./output

# Dry run (preview filters without moving files)
python main.py --input ./data/raw --dry-run

# Specify a custom config file
python main.py --input ./data/raw --config my_config.yaml

# Verbose logging
python main.py --input ./data/raw --verbose
```

### Streamlit Dashboard

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser to view the interactive dashboard.

---

## 🐳 Docker

```bash
# Build the image
docker build -t voicedatacurator .

# Run the pipeline
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output voicedatacurator

# Run with Docker Compose (pipeline + dashboard)
docker-compose up
```

---

## 📋 Output: `dataset_manifest.csv`

| filename | language | duration | snr | silence_ratio | quality_score | status | rejection_reason |
|---|---|---|---|---|---|---|---|
| clip_001.wav | en | 4.32 | 22.1 | 0.08 | 0.91 | accepted | |
| clip_002.wav | hi | 1.5 | 6.3 | 0.61 | 0.34 | rejected | low_snr, high_silence |
| clip_003.wav | fr | 12.8 | 18.7 | 0.12 | 0.85 | accepted | |

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Audio Processing | `librosa`, `soundfile`, `numpy` |
| Language Detection | `openai-whisper` |
| CLI Interface | `argparse`, `tqdm` |
| Dashboard | `Streamlit`, `Plotly`, `Pandas` |
| Configuration | `PyYAML` |
| Logging | Python `logging` module |
| Containerization | `Docker`, `docker-compose` |

---

## 🛠️ Core Modules

### `audio_analyzer.py`
Provides functions for:
- `compute_snr(audio, sr)` — estimates signal-to-noise ratio
- `compute_silence_ratio(audio, sr)` — measures proportion of silent frames
- `detect_clipping(audio)` — checks for amplitude clipping
- `validate_duration(audio, sr, min_dur, max_dur)` — enforces duration bounds
- `check_sample_rate(sr, target_sr)` — validates sample rate consistency

### `language_detector.py`
- Loads Whisper model (configurable size: `tiny`, `base`, `small`, `medium`)
- `detect_language(audio_path)` → returns `(language_code, confidence)`
- Caches model in memory across batch processing for speed

### `pipeline.py`
- Orchestrates the full ingestion → analysis → filtering → export flow
- Handles file I/O, manifest building, and rejection logging
- Supports `dry_run` mode without side effects

### `dashboard.py`
- Reads `dataset_manifest.csv` and renders real-time Streamlit charts
- Interactive filters for language, status, and quality score range

---

## 📊 Example Dashboard

The Streamlit dashboard provides at-a-glance visibility into your dataset:

- 🥧 **Language distribution** pie chart
- 📈 **Quality score** histogram
- ✅ **Accepted vs ❌ Rejected** count cards
- ⏱️ **Average duration** per language bar chart
- 📋 **Per-file report** table with sort & filter

---

## 🔄 Pipeline Flow

```
Raw Audio Files (WAV/MP3)
        │
        ▼
┌───────────────────┐
│  Audio Analyzer   │  ← SNR, silence, clipping, duration, sample rate
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Language Detector │  ← Whisper transcription → language tag
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Filtering Pipeline│  ← Apply thresholds from config.yaml
└───┬───────────────┘
    │
    ├──── ✅ Accepted → output/
    └──── ❌ Rejected → rejected/ (with reason log)
        │
        ▼
┌───────────────────┐
│ dataset_manifest  │  ← CSV export with all metadata
│    .csv           │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Streamlit Dashboard│ ← Visual stats & per-file report
└───────────────────┘
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Farheen Fathima**
- GitHub: [@farheenfathimaa](https://github.com/farheenfathimaa)

---

<div align="center">
  <i>Built for speech AI teams who care about data quality as much as model quality.</i>
</div>
