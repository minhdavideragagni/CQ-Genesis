# 🌱 CQ-Genesis

> **LLM-Assisted and Human-Guided Competency Question Generation from Structured Data and User Stories**

CQ-Genesis is a research prototype that assists knowledge engineers in generating and refining Competency Questions (CQs) from structured datasets, user stories, or their combination.

Rather than replacing the expertise of ontology engineers, CQ-Genesis provides configurable AI-assisted support while preserving human control over input formulation, model selection, generation, and review.

---

## ✨ Key Features

- 🗂️ Generate Competency Questions from:
  - structured datasets;
  - user stories;
  - structured datasets + user stories.

- 🤖 Support for both proprietary and open-weight LLM ecosystems.

- ⚙️ Configurable generation process:
  - model;
  - temperature;
  - output language;
  - number of CQs;
  - representative dataset sample.

- 🧩 Structured prompting strategy combining:
  - task instructions;
  - conceptual guidance;
  - CQ formulation examples;
  - optional CQ patterns;
  - structured JSON generation.

- 👨‍💻 Human-guided workflow:
  - editable CQs;
  - interactive review;
  - export in multiple formats.

---

## 🏗️ System Architecture

```
Frontend (Streamlit)
        │
        ▼
CQ-Genesis REST API (FastAPI)
        │
        ▼
Large Language Models
(OpenAI / Hugging Face)
```

---

## 🚀 Online Demo

https://cq-genesis.streamlit.app

---

## 🔗 REST API

Base URL

```
https://cq-genesis-api.onrender.com
```

Main endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/profile` | Structured dataset profiling |
| POST | `/generate` | Competency Question generation |

---

## 💻 Local Installation

Clone the repository

```bash
git clone https://github.com/minhdavideragagni/CQ-Genesis.git
cd CQ-Genesis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the backend

```bash
python3 -m uvicorn backend.main:app --reload
```

Run the frontend

```bash
python3 -m streamlit run frontend/app.py
```

---

## 🔑 Supported LLM Providers

CQ-Genesis currently supports:

- OpenAI (proprietary models)
- Hugging Face Inference Providers (open-weight models)

The user provides their own API credentials. No credentials are stored by the system.

---

## 📄 Research

CQ-Genesis is currently under active research and development.

If you use CQ-Genesis in academic work, please cite the accompanying publication (coming soon).

---

## 📜 License

MIT License
