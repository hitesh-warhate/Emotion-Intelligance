<div align="center">

# RAMEI

### Reliability-Aware Adaptive Multimodal Emotion Intelligence

Recognizes human emotion from **voice, words and face** together, knows **how much to trust each
signal** at every moment, and **keeps working when signals are noisy, degraded or missing**.

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React%2018-Vite%20%2B%20TypeScript-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)

<img src="docs/presentation/img/05_live.jpg" alt="RAMEI live analysis dashboard" width="860">

</div>

---

## Overview

Most emotion-recognition systems trust a single signal, or average several blindly. RAMEI scores
the **reliability** of each modality for every 3-second segment (background noise in the voice, an
uncertain transcript, a face that is blurred, dark or out of frame) and lets the reliable signals
decide. Every prediction comes with a **calibrated confidence**, an **"uncertain"** answer instead of
a guess when the evidence is weak, and a **plain-language explanation** of why.

> Built as an AD-INFOCOMM internship project, evolving an audio-only speech-emotion baseline into a
> multimodal system with a full web platform.

## Key features

| | |
|---|---|
| 🎙️ **Three modalities** | Voice (WavLM), words (Whisper + DeBERTa-v3), face (MediaPipe detection + ViT) |
| ⚖️ **Reliability-gated fusion** | Per-segment quality scores decide how much each signal counts |
| 🛡️ **Robust to degradation** | ≤ 10% accuracy drop under every tested corruption; missing modalities are handled |
| 🎯 **Calibrated + abstains** | ECE 0.044; returns "uncertain" rather than guessing |
| 💬 **Explainable** | e.g. *"Voice down-weighted: background noise (SNR ≈ 7 dB)"* |
| 📡 **Live and upload modes** | Real-time mic + camera streaming over WebSocket, or analysis of recorded files |
| 📊 **Dashboard** | Session history, emotion timelines, valence/arousal, alerts, full reports |
| 🔐 **Privacy by design** | Consent required before capture; raw audio/video streams are not stored |

## Results

| Model | Data | Macro-F1 | Notes |
|---|---|---|---|
| Thesis baseline (MFCC + MLP) | RAVDESS, 6-fold speaker CV | 0.512 ± 0.076 | [M0 report](docs/reports/M0-baseline-report.md) |
| Audio branch (WavLM + LoRA) | RAVDESS, 6-fold speaker CV | 0.686 ± 0.028 | [M1 report](docs/reports/M1-unimodal-report.md) |
| Text branch (DeBERTa-v3 + LoRA) | MELD test | 0.448 (weighted-F1 0.591) | [M1 report](docs/reports/M1-unimodal-report.md) |
| Audio branch alone (frozen WavLM) | CREMA-D, 5-fold speaker CV | 0.592 ± 0.033 | 6 classes |
| **RAMEI fusion (voice + words + face)** | CREMA-D, 5-fold speaker CV | **0.769 ± 0.023** | ECE 0.044 · [M2 report](docs/reports/M2-fusion-report.md) |

## Architecture

```
 Browser (React + Vite)  ──HTTPS / WebSocket──►  FastAPI
                                                   ├─ auth, sessions, uploads, live stream
                                                   ├─ Inference engine (versioned model bundle)
                                                   │    3 s segments → [ WavLM | Whisper + DeBERTa | Face + ViT ]
                                                   │    → reliability per modality → reliability-gated fusion
                                                   │    → calibration / abstention → explanation
                                                   └─ SQLite (dev) / PostgreSQL (prod)
```

Details: [System Architecture](docs/03-System-Architecture.md) ·
[ADR-0001 Reliability-gated fusion](docs/adr/0001-reliability-gated-fusion.md)

## Tech stack

| Layer | Technologies |
|---|---|
| **ML** | PyTorch, Hugging Face Transformers, PEFT (LoRA), faster-whisper, MediaPipe, OpenCV |
| **Backend** | FastAPI, Uvicorn, SQLAlchemy, WebSockets, JWT auth |
| **Frontend** | React 18, TypeScript, Vite, TanStack Query, Recharts, Tailwind CSS |
| **Infra** | Docker Compose, nginx, GitHub Actions CI |

## Repository structure

```
├── backend/     FastAPI service, inference engine, analysis pipeline, tests
├── frontend/    React + Vite + TypeScript dashboard
├── ml/          Data pipelines, models, training, fusion, evaluation, bundle export
├── database/    Reference schema
├── infra/       Dockerfiles and nginx config
├── scripts/     Load test, demo-clip generator, report builder, bundle packager
└── docs/        PRD, TRD, architecture, ADRs, milestone + final reports, demo and presentation guides
```

## Getting started

### Prerequisites

- **Python 3.11 or 3.12** (64-bit) and **Node.js 20+**
- 8 GB RAM minimum (16 GB recommended), ~15 GB free disk
- An NVIDIA GPU is optional; it makes analysis about 3× faster
- The **trained model bundle** (`ramei-model-bundle.zip`, ~9 MB). Model weights are not stored in
  Git; get the zip from the project team or train it yourself (see [Training](#training-the-model)).

Extract the bundle into `ml/artifacts/bundles/` so that this file exists:

```
ml/artifacts/bundles/ramei-rgcmt-1.0.0/manifest.json
```

### Option A: run locally

```bash
# 1. Python environment (shared by ML and backend)
cd ml
python -m venv .venv
.venv\Scripts\activate                      # macOS/Linux: source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu    # or /whl/cu124 for NVIDIA
pip install -e ".[audio,multimodal]" -e "../backend[inference]"

# 2. Create a login (the password is written to backend/.dev-credentials, git-ignored)
cd ../backend
python -m app.cli create-user --email admin@ramei.local --role admin --name "Admin"

# 3. Start the API (http://127.0.0.1:8000/docs)
uvicorn app.main:app --port 8000 --ws-ping-interval 30 --ws-ping-timeout 300
```

In a second terminal:

```bash
# 4. Start the dashboard (http://localhost:5173)
cd frontend
npm install
npm run dev
```

### Option B: Docker

```bash
cp .env.example .env               # set JWT_SECRET to 32+ random characters
docker compose up -d --build       # dashboard + API on http://localhost:8080
docker compose exec api python -m app.cli create-user --email admin@ramei.local --role admin --password "ChooseAPassword1"
```

### Check that it works

Open <http://127.0.0.1:8000/api/v1/health> (or `http://localhost:8080/api/v1/health` with Docker).
You should see `"model_bundle_found": true`.

## Usage notes

- **First analysis:** the public encoder models (~2 GB) download automatically on first use, then
  load into memory (about a minute on CPU). The first live session after each backend start waits on
  this "loading models" step; later sessions start instantly.
- **Model cache location:** set `HF_HOME` to choose where the encoder models are stored, and
  `MODEL_BUNDLE_DIR` if the bundle lives outside `ml/artifacts/bundles`.
- **Live mode on CPU:** each 3-second window runs four models, so results arrive a few seconds behind
  real time. When analysis falls behind, it skips ahead to the latest window (keeping that window's
  camera frames). A GPU keeps up with real time.
- **Best accuracy:** face the camera with even lighting, use a quiet room or headset mic, and speak
  in full sentences.
- `--ws-ping-timeout 300` keeps the live connection open while the models load on slower machines.

## Training the model

Training on CREMA-D takes about 6 hours on an RTX 3050 (resumable):

```bash
cd ml
python scripts/download_crema_d.py && python scripts/download_crema_d_video.py
python -m ramei_ml.training.text_branch
python scripts/run_m2.py             # features → cross-validation → bundle
python ../scripts/package_bundle.py  # optional: writes ramei-model-bundle.zip to share
```

## Documentation

| Document | What it answers |
|---|---|
| [Product Requirements (PRD)](docs/01-PRD.md) | What is built, for whom, and how success is measured |
| [Technical Requirements (TRD)](docs/02-TRD.md) | Models, data, APIs, database, performance, security, testing |
| [System Architecture](docs/03-System-Architecture.md) | Components, runtime flows, deployment |
| [Architecture Decision Records](docs/adr/) | Why key decisions were made |
| [Milestone reports](docs/reports/) | Results of each phase (M0 baseline, M1 branches, M2 fusion) |
| [Final report](docs/reports/Final-Report.md) | The whole project in one document |
| [Run on another computer](docs/RUN-ON-ANOTHER-COMPUTER.md) | Running with the shared model zip, no retraining |
| [Demo runbook](docs/demo/Demo-Script.md) | Demo preparation, video script, likely questions |
| [Presentation guide (PDF)](docs/presentation/RAMEI-Presentation-Guide.pdf) | Diagrams, results, screenshots, script and Q&A |

## Development

```bash
cd backend && pytest -q          # backend tests
cd ml && pytest -q               # ML tests
cd frontend && npm test          # frontend tests
```

- Branches: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, merged into `main` after review
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
- Never commit secrets, datasets, model weights or recordings (enforced by `.gitignore`)

## Responsible use

Emotion recognition is probabilistic and can be wrong, especially across accents, cultures, lighting
and recording conditions that differ from the training data (acted English speech in CREMA-D). Use
RAMEI only with the informed consent of everyone recorded, treat its output as one signal among
many, and never as the sole basis for decisions about people.
