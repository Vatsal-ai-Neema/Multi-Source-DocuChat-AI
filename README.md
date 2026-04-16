# 📄 Multi-Source DocuChat AI (Advanced RAG)

## 🚀 Overview

Multi-Source DocuChat AI is an intelligent document-based chatbot system that leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware answers based on user-uploaded documents.

Unlike traditional chatbots, this system ensures that responses are strictly grounded in the provided documents, reducing hallucination and improving reliability.

---

## 🎯 Key Features

* 📄 Upload multiple documents (PDF, TXT)
* ✂️ Automatic text chunking
* 🧠 Embedding generation
* 📚 Vector database (FAISS)
* 💬 Context-aware question answering (RAG)
* 📌 Document summarization 
* 🔍 Keyword extraction 
* ⚡ HyDE Retrieval (Hypothetical Document Embeddings)
* 🧠 Socratic Answering with follow-ups
* 📊 Table extraction from PDFs
* 📈 Table & chart analysis using LLM
* ⚖️ Contradiction detection (NLI + LLM)
* 🐳 Docker support for deployment
* ⚙️ CI/CD integration (GitHub Actions)

---

## ⚙️ How It Works (RAG Pipeline)

```
User uploads PDF/TXT
        ↓
Text extracted per page
        ↓
Split into chunks (800 chars, 150 overlap)
        ↓
Each chunk → vector embedding (MiniLM-L6)
        ↓
Vectors stored in FAISS index
        ↓
User asks question
        ↓
Question → vector → FAISS similarity search
        ↓
Top-4 relevant chunks retrieved
        ↓
Chunks + Question → Gemini API
        ↓
Accurate, cited answer shown to user
```

---


## 🛠 Tech Stack

| Layer        | Technology                          | Why                              |
|-------------|-------------------------------------|----------------------------------|
| UI           | Streamlit                           | Fast, Python-native web UI       |
| Embeddings   | sentence-transformers/all-MiniLM-L6 | Free, local, fast, good quality  |
| Vector DB    | FAISS (CPU)                         | Fast similarity search, local    |
| LLM          | Google Gemini 1.5 Flash             | Free tier, fast, capable         |
| RAG Pipeline | LangChain                           | Industry standard RAG framework  |
| PDF Reading  | pdfplumber + PyPDF2                 | Reliable text extraction         |
| Container    | Docker                              | Reproducible deployments         |
| CI/CD        | GitHub Actions                      | Automated testing + build        |

---

## 📂 Project Structure

```
docuchat-ai-combined/
└── docuchat-ai/
    │
    ├── app.py
    ├── requirements.txt
    ├── Dockerfile
    ├── README.md
    ├── .env
    ├── .env.example
    ├── .gitignore
    │
    ├── data/
    │   ├── ai_lab_manual.txt
    │   ├── AI_Table_Strong_Grid.pdf
    │   ├── AI_Visualization_Table_Template.pdf
    │   ├── product_brief_docuchat.txt
    │   ├── student_handbook_v1.txt
    │   └── student_handbook_v2.txt
    │
    ├── faiss_index/
    │   ├── index.faiss
    │   └── index.pkl
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── loader.py
    │   ├── chunker.py
    │   ├── embedder.py
    │   ├── retriever.py
    │   └── llm.py
    │
    ├── tests/
    │   └── test_pipeline.py
    │
    ├── .github/
    │   └── workflows/
    │       └── ci-cd.yml
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone <repo-url>
cd docuchat-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Key

Create `.env` file:

```
API_KEY=your_key_here
```

### 5. Run Application

```bash
streamlit run app.py
```

---

## 🐳 Docker Setup

```bash
docker build -t docuchat .
docker run -p 8501:8501 docuchat
```

---

## 🔄 CI/CD Pipeline

* GitHub Actions automatically:

  * Runs tests
  * Builds Docker image
  * Deploys application

---

## 🎓 Course Mapping

### IBM Generative AI Engineering

* Prompt engineering
* LLM integration
* AI application development

### IBM RAG & Agentic AI

* Document processing
* Embeddings
* Vector search
* RAG pipeline

### DevOps Mastery

* Docker containerization
* CI/CD automation
* Deployment

---
## 💡 Features Upgradation

### 🟢 Core
- Multi-document upload (PDF + TXT)
- RAG-based Q&A (grounded answers only)
- Source citations with page numbers
- Chat history

### 🟡 Smart
- Document summarization
- Keyword & concept extraction

### 🔵 Advanced
- Multi-document synthesis
- Streaming responses
- Docker deployment
- CI/CD pipeline
---

## 📌 Future Enhancements

* Multi-document comparison
* Chat history
* Source citation (page-level)
* Resume analysis integration
