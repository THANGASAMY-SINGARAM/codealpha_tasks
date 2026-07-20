<div align="center">

# 🚀 CodeAlpha — AI & ML Internship Tasks

**A collection of Artificial Intelligence / Machine Learning projects built during the CodeAlpha internship.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Overview](#-overview) • [Tasks](#-tasks) • [Repository Structure](#-repository-structure) • [Getting Started](#️-getting-started) • [Tech Stack](#️-tech-stack) • [Author](#-author)

</div>

---

## 📖 Overview

This repository consolidates the tasks completed as part of the **CodeAlpha Artificial Intelligence Internship**. Each task lives in its own self-contained folder with its own source code, dependencies, and detailed `README.md`, so every project can be set up and run independently.

| # | Task | Domain | Folder |
|---|---|---|---|
| 1 | FAQ Chatbot with Hybrid NLP + LLM | Natural Language Processing | [`FAQ_Chatbot/`](./FAQ_Chatbot) |
| 2 | Real-Time Object Detection & Tracking | Computer Vision | [`Object_Detection/`](./Object_Detection) |

---

## 🧩 Tasks

### Task 1 · 🤖 FAQ Chatbot

[`/FAQ_Chatbot`](./FAQ_Chatbot)

An interactive Streamlit chatbot that answers FAQs using a **hybrid matching engine** — local NLP for speed and precision, with an LLM fallback for open-ended queries.

**Highlights**
- Text preprocessing pipeline (tokenization, stopword removal, WordNet lemmatization) via **NLTK**
- **TF-IDF + cosine similarity** matching (scikit-learn), with **Groq LLM** (`llama-3.1-8b-instant`) for grounded, context-aware fallback answers
- Live diagnostics view: token breakdown and per-query similarity score charts
- Built-in FAQ database manager (search, add, edit, delete)

**Run it:**
```bash
cd FAQ_Chatbot
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
📄 Full documentation: [`FAQ_Chatbot/README.md`](./FAQ_Chatbot/README.md)

---

### Task 2 · 🎯 Real-Time Object Detection & Tracking

[`/Object_Detection`](./Object_Detection)

A computer vision web app for detecting and tracking objects in images, videos, and live webcam streams, built on **YOLOv8** and **OpenCV**.

**Highlights**
- Real-time inference across all 80 COCO object classes
- Multi-object tracking with per-object trajectory visualization
- Adjustable confidence and NMS thresholds, with class-level filtering
- Live analytics: object counts, detection metrics, frame-by-frame breakdown

**Run it:**
```bash
cd Object_Detection
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```
📄 Full documentation: [`Object_Detection/README.md`](./Object_Detection/README.md)

---

## 📁 Repository Structure

```
codealpha_tasks/
├── FAQ_Chatbot/                 # Task 1 — FAQ Chatbot (NLP + Groq LLM)
│   ├── app.py                   # Streamlit web interface
│   ├── nlp_processor.py         # NLTK preprocessing + TF-IDF matching engine
│   ├── faqs.json                # FAQ knowledge base
│   ├── test_nlp.py              # Automated tests
│   ├── requirements.txt
│   └── README.md
│
├── Object_Detection/             # Task 2 — Object Detection & Tracking
│   ├── streamlit_app.py         # Streamlit web application
│   ├── main.py                  # CLI entry point
│   ├── yolov8n.pt                # Pre-trained YOLOv8 model weights
│   ├── src/                      # Detection & tracking modules
│   ├── assets/                   # Sample video assets
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
├── LICENSE
└── README.md                     # You are here
```

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- Git

### Clone the repository
```bash
git clone https://github.com/THANGASAMY-SINGARAM/codealpha_tasks.git
cd codealpha_tasks
```

Each task has its own virtual environment and dependencies — see the **Run it** steps above, or the task's own README, for full setup details.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Web Framework** | Streamlit |
| **NLP / ML** | NLTK, scikit-learn, Groq API, NumPy, Pandas |
| **Computer Vision** | OpenCV, Ultralytics YOLOv8, PyTorch |

---

## 📜 License

This repository is licensed under the [MIT License](./LICENSE).

---

## 👤 Author

**Thangasamy Singaram**
AI/ML Intern @ CodeAlpha
[GitHub](https://github.com/THANGASAMY-SINGARAM)

<div align="center">

*Built as part of the CodeAlpha Internship Program.*

</div>
