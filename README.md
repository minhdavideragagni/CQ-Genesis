<p align="center">
  <img src="frontend/assets/cq-genesis-logo.png" width="555">
</p>

<h1 align="center">CQ-Genesis</h1>

<h3 align="center">
<i>LLM-Assisted and Human-Guided Competency Question Generation from Structured Data and User Stories</i>
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

CQ-Genesis is a research prototype that assists knowledge engineers in generating and refining Competency Questions (CQs) from structured datasets, user stories, or their combination.

Rather than replacing ontology engineers, CQ-Genesis provides configurable AI-assisted support while preserving human control over input formulation, model selection, generation, and review.

---

## Design Philosophy

CQ-Genesis follows a **human-guided** design philosophy.

The system is designed to support—not replace—the expertise of knowledge engineers throughout the Competency Question elicitation process. Every stage of the workflow remains configurable and reviewable, allowing users to retain full control over requirement specification, model selection, generation parameters, and the final set of Competency Questions.

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

## Project Structure

```text
CQ-Genesis
│
├── backend/
│   ├── main.py                 # FastAPI application
│   └── cq_genesis/             # Core generation engine
│       ├── generation.py
│       ├── profiling.py
│       ├── prompting.py
│       ├── validation.py
│       ├── models.py
│       └── ...
│
├── frontend/
│   ├── app.py                  # Streamlit interface
│   └── assets/                 # Images and visual resources
│
├── requirements.txt
├── README.md
└── LICENSE
```

CQ-Genesis adopts a modular architecture separating the user interface, the REST API, and the core generation engine. This design facilitates maintainability, reproducibility, and future integration with external ontology engineering workflows.

---

## System Architecture

```text
                User
                  │
                  ▼
        Streamlit Frontend
                  │
                  ▼
        CQ-Genesis REST API
             (FastAPI)
                  │
                  ▼
      Large Language Models
   (OpenAI / Hugging Face)
```

---

## Online Services

| Service | URL |
|----------|-----|
| Web Application | https://cq-genesis.streamlit.app |
| REST API | https://cq-genesis-api.onrender.com |
| API Documentation | https://cq-genesis-api.onrender.com/docs |

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
