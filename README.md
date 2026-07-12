<p align="center">
  <img src="frontend/assets/cq-genesis-logo.png" width="555">
</p>

<h1 align="center">CQ-Genesis</h1>

<h3 align="center">
<i>LLM-Assisted and Human-Guided Competency Question Generation<br>
from Structured Data and User Stories</i>
</h3>

<p align="center">

<a href="https://cq-genesis.streamlit.app">
<img src="https://img.shields.io/badge/Demo-Online-success?style=for-the-badge">
</a>

<a href="https://cq-genesis-api.onrender.com/docs">
<img src="https://img.shields.io/badge/REST_API-Online-blue?style=for-the-badge">
</a>

<a href="LICENSE">
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</a>

</p>

---

## Overview

CQ-Genesis assists knowledge engineers in generating and refining Competency Questions (CQs) from structured datasets, user stories, or their combination.

Rather than replacing knowledge engineers, CQ-Genesis provides configurable AI-assisted support while preserving human control over input formulation, model selection, generation, and review.

---

## Key Features

- 🌱 Generate Competency Questions from:
  - structured datasets;
  - user stories;
  - their combination.

- 🤖 Support for proprietary and open-weight Large Language Models.

- ⚙️ Configurable generation workflow:
  - model selection;
  - generation parameters;
  - output language;
  - representative dataset sample;
  - number of Competency Questions.

- 🧩 Structured prompting strategy combining:
  - explicit task instructions;
  - conceptual guidance;
  - quality constraints;
  - CQ formulation examples;
  - optional CQ patterns;
  - structured JSON output.

- 👨‍💻 Human-guided review:
  - interactive editing;
  - inclusion/exclusion of generated CQs;
  - export in multiple formats.

- 🔄 Reproducible generation records.

---

## Repository Organization

| Path | Description |
|------|-------------|
| `backend/` | FastAPI backend exposing the CQ-Genesis REST API and coordinating the generation workflow. |
| `backend/cq_genesis/` | Core modules implementing dataset profiling, prompting, generation, validation, models, and shared utilities. |
| `frontend/` | Streamlit-based graphical user interface. |
| `frontend/assets/` | Images and graphical resources used by the interface and documentation. |
| `requirements.txt` | Python dependencies required to run CQ-Genesis. |
| `README.md` | Project documentation and installation guide. |
| `LICENSE` | Software license. |

---

## Online Services

| Service | URL |
|---------|-----|
| 🌐 Web Application | https://cq-genesis.streamlit.app |
| ⚙️ REST API | https://cq-genesis-api.onrender.com |
| 📘 API Documentation | https://cq-genesis-api.onrender.com/docs |

---

## Local Installation

Clone the repository:

```bash
git clone https://github.com/minhdavideragagni/CQ-Genesis.git
cd CQ-Genesis
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python3 -m uvicorn backend.main:app --reload
```

Start the frontend:

```bash
python3 -m streamlit run frontend/app.py
```

---

## Supported LLM Ecosystems

CQ-Genesis currently supports:

- **OpenAI** (closed-weight models)
- **Hugging Face Inference Providers** (open-weight models)

The knowledge engineer provides their own API credentials. No credentials are stored by the system.
