# 💠 MindCompass AI: Multimodal Mental Health Prediction & RAG System

**MindCompass AI** is a state-of-the-art emotional support system designed to analyze mental health states using **Multimodal Emotion Recognition** (Text, Audio, Video) and provide personalized, empathetic insights through a **Retrieval-Augmented Generation (RAG)** chatbot ("Guardian AI").

---

## 🏗️ System Architecture

### 1. Multimodal Emotion Recognition (The "Senses")
The system uses three specialized deep learning models to perceive emotion:

*   **📝 Text Model (Mental-RoBERTa)**:
    *   **Architecture**: `roberta-base` fine-tuned on mental health datasets.
    *   **Function**: Analyzes user text input to detect semantic emotional cues.
*   **🎤 Audio Model (Wav2Vec2 + XGBoost)**:
    *   **Architecture**: `facebook/wav2vec2-base` for feature extraction + XGBoost classifier.
    *   **Function**: Extracts acoustic features (tone, pitch, prosody) from voice recordings.
*   **🎥 Video Model (ConvNeXt + Spatial Attention)**:
    *   **Architecture**: `convnext_tiny` backbone + Custom **Spatial Attention** module + XGBoost classifier.
    *   **Function**: Analyzes facial expressions and visual cues from video frames.

### 2. Attention Fusion Network (The "Brain")
*   **Mechanism**: A custom `ResidualAttentionBlock` fuses the outputs (logits) from the Text, Audio, and Video models.
*   **Goal**: Dynamically weighs the importance of each modality to produce a final, high-confidence "Wellbeing Cluster" prediction.

### 3. RAG System (The "Guardian")
*   **Knowledge Base**: A curated clinical repository stored in `knowledge_base.json`.
*   **Vector Database**: FAISS (Facebook AI Similarity Search) index storing semantic embeddings of the knowledge base.
*   **Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` embeds user queries to find relevant clinical advice.
*   **Generation**: Groq API (using **`llama-3.3-70b-versatile`** or **`mixtral-8x7b-32768`**) synthesizes the retrieved context into an empathetic response.

---

## 🚀 Setup & Installation

### 1. Prerequisites
*   **OS**: Windows / Linux / MacOS
*   **Python**: 3.8+
*   **Hardware**: GPU (NVIDIA CUDA) recommended for real-time inference.
*   **API Key**: Groq API Key (Sign up at [console.groq.com](https://console.groq.com/)).

### 2. Environment Setup
1.  **Clone/Download** the repository.
2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Key libraries: `torch`, `transformers`, `fastapi`, `streamlit`, `langchain-groq`, `faiss-cpu`, `xgboost`.*

4.  **Configure API Keys**:
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

---

## 🖥️ Usage Guide

### 1. Running the System
The system consists of two parts: the Backend API and the Frontend Dashboard.

**Step 1: Start the Backend (API)**
This handles all model inference and RAG logic.
```bash
python app.py
```
*   **URL**: `http://localhost:8000`
*   *Wait for "Application startup complete" message.*

**Step 2: Start the Frontend (UI)**
This launches the interactive dashboard.
```bash
streamlit run dashboard.py
```
*   **URL**: `http://localhost:8501`

### 2. Using the Dashboard
1.  **Multimodal Input**:
    *   **Text**: Type how you are feeling.
    *   **Audio**: Upload a `.wav` file.
    *   **Video**: Upload a `.mp4` file.
2.  **Generate Insights**: Click "Generate Emotional Insights" to see the Fusion Prediction and analysis.
3.  **Chat with Guardian**: Use the sidebar chat to ask follow-up questions. The AI uses the RAG system to answer based on your detected emotion.

---

## 🧠 Training the Models (Optional)

If you wish to retrain the models, use the scripts in the `training/` directory.

### 1. Dataset Downloads
You need to download these datasets and place them in the `datasets/` folder:

| Modality | Dataset | Description | Link |
| :--- | :--- | :--- | :--- |
| **Video** | **RAVDESS** | Ryerson Audio-Visual Database of Emotional Speech and Song. (Download `Video_Speech_Actor_01` to `24`) | [Zenodo Link](https://zenodo.org/record/1188976) |
| **Audio** | **CREMA-D** | Crowd-sourced Emotional Multimodal Actors Dataset. | [Kaggle Link](https://www.kaggle.com/datasets/ejlok1/cremad) |
| **Image** | **FER-2013** | Facial Expression Recognition 2013 (Base training). | [Kaggle Link](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) |

### 2. Training Scripts
Run these scripts in order to retrain:

1.  **Text Model**: `python training/1_train_text_model.py`
2.  **Audio Model**: `python training/5_train_high_perf_audio.py` (Uses Embeddings + XGBoost)
3.  **Video Model**: `python training/6_train_high_perf_video.py` (Uses ConvNeXt + Spatial Attention)
4.  **Fusion Model**: `python training/4_train_fusion_model.py` (Trains the Attention Fusion network)

---

## 🛡️ RAG System & Knowledge Base

The **Guardian AI** relies on `knowledge_base.json` to provide safe advice.

### Updating the Knowledge Base
To add new mental health topics or coping strategies:
1.  Open `knowledge_base.json`.
2.  Append a new entry:
    ```json
    {
        "emotion": "stress",
        "text": "The 4-7-8 breathing technique involves inhaling for 4 seconds, holding for 7, and exhaling for 8..."
    }
    ```
3.  **Restart the Backend** (`python app.py`). The system will automatically rebuild the FAISS vector index with your new data.

---

## ⚠️ Troubleshooting / Common Issues

*   **Port Conflicts (Address already in use)**:
    *   If port 8000 is stuck:
        ```powershell
        Get-NetTCPConnection -LocalPort 8000 | % { Stop-Process -Id $_.OwningProcess -Force }
        ```
*   **Groq Model Errors**:
    *   If `llama-3.1-70b-versatile` is decommissioned, edit `rag_service.py` to use `mixtral-8x7b-32768`.
*   **CUDA Out of Memory**:
    *   Reduce `BATCH_SIZE` in `config.py` appropriately.

---
**© 2026 MindCompass AI Project**
