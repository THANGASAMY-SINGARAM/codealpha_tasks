# CodeAlpha AI Tasks Repository

Welcome to the **CodeAlpha Tasks** repository! This repository consolidates key Artificial Intelligence & Machine Learning projects built as part of the CodeAlpha internship tasks.

---

## 📁 Repository Structure

```text
codealpha_tasks/
├── FAQ_Chatbot/             # Task 1: FAQ Chatbot with NLP & Groq LLM
│   ├── app.py               # Main Streamlit Web Interface
│   ├── nlp_processor.py     # NLTK preprocessing & TF-IDF Cosine Similarity Engine
│   ├── faqs.json            # FAQ Database
│   ├── test_nlp.py          # Automated verification script
│   ├── requirements.txt     # Dependencies
│   └── README.md            # Detailed documentation
│
├── Object_Detection/        # Task 2: Real-time Object Detection & Tracking
│   ├── streamlit_app.py     # Main Streamlit Web Application
│   ├── main.py              # CLI / Entry point
│   ├── yolov8n.pt           # Pre-trained YOLOv8 model weights
│   ├── src/                 # Object detection & tracking modules
│   ├── assets/              # Sample video assets
│   ├── requirements.txt     # Dependencies
│   └── README.md            # Detailed documentation
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Projects Overview

### 1. 🤖 FAQ Chatbot (`/FAQ_Chatbot`)
An interactive, data-driven web chatbot built with **Streamlit**, featuring a **Hybrid NLP & Groq LLM Processing Engine**.

#### Key Features:
- **Local NLP Engine**: Text tokenization, stopword removal, and WordNet lemmatization using **NLTK**, combined with **TF-IDF Vectorization** and **Cosine Similarity** matching (scikit-learn).
- **Groq LLM Engine**: Semantic intent parsing and context-grounded response generation using the **Groq API** (`llama-3.1-8b-instant`).
- **Interactive Diagnostics**: Real-time breakdown of preprocessed tokens and interactive bar charts displaying cosine similarity scores across the FAQ database.
- **FAQ Database Manager**: Add, search, and delete FAQs dynamically via a built-in administration tab.

#### Quick Start:
```bash
cd FAQ_Chatbot
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

### 2. 🎯 Real-Time Object Detection & Tracking (`/Object_Detection`)
A computer vision web application leveraging **YOLOv8** (Ultralytics) and **OpenCV** to perform object detection and object tracking on images, videos, and live webcam feeds.

#### Key Features:
- **YOLOv8 Inference**: Real-time object identification across 80 COCO classes.
- **Multi-Object Tracking**: Track movement trajectories of objects across video frames.
- **Customizable Filters**: Adjust confidence thresholds, NMS thresholds, and select target object classes on the fly.
- **Analytics & Visualizations**: Displays object counts, detection metrics, and frame-by-frame diagnostic breakdowns.

#### Quick Start:
```bash
cd Object_Detection
python -m venv venv
# On Windows: venv\Scripts\activate
# On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🛠️ Requirements & Tech Stack
- **Languages**: Python 3.10+
- **Web Framework**: Streamlit
- **NLP & ML**: NLTK, Scikit-Learn, Groq API, NumPy, Pandas
- **Computer Vision**: OpenCV, Ultralytics YOLOv8, PyTorch