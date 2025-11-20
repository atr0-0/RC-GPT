# CaseLawGPT - Supreme Court Tort Law Research Tool

CaseLawGPT is a practical research tool for Indian Supreme Court tort law cases. It provides a modern web interface and backend for searching, filtering, and reading case law, built for lawyers and legal researchers.

## 📁 Project Structure

```
CaseLawGPT/
│
├── src/                # Backend (FastAPI) & Processing Scripts
├── frontend/           # React web interface
├── storage/            # Data, Vector Store, and Processed files
├── misc/               # Research, Docs, Archive, Tests
├── requirements.txt    # Python dependencies
├── Dockerfile          # Backend Docker config
├── docker-compose.yml  # Container orchestration
├── setup.ps1           # One-click setup script
└── run.ps1             # One-click run script
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Node.js** (for local frontend development)
- **Google Cloud Credentials** (`google_credentials.json` in root)
- **Pinecone API Key** (Optional, for Cloud Vector Store)
- **Docker Desktop** (Optional, for containerized run)

### Option 1: Local Setup (Recommended for Dev)

We have simplified the setup process with PowerShell scripts.

1.  **Setup Environment** (Installs Python venv, dependencies, and Node modules):
    ```powershell
    .\setup.ps1
    ```

2.  **Run Application** (Starts Backend & Frontend):
    ```powershell
    .\run.ps1
    ```
    - Frontend: http://localhost:3000
    - Backend API: http://localhost:8000/docs

### Option 2: Docker (Production / Easy Run)

Run the entire stack (Frontend + Backend + Nginx) in containers.

```powershell
docker-compose up --build
```
- Access the app at http://localhost:3000

---

## ☁️ Cloud Vector Database

This project uses **Pinecone** as the vector database.

1.  **Get API Key**: Sign up at [Pinecone.io](https://www.pinecone.io/) (Free Tier).
2.  **Configure**: Add your key to `.env`:
    ```env
    PINECONE_API_KEY=your-api-key-here
    ```
3.  **Build/Update Index**:
    ```powershell
    python src/build_vector_store.py
    ```
    This will process your documents and upload them to the cloud index.

---

## 🧠 Data Processing

If you need to re-process the raw text files:

```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Run processing scripts
python src/batch_process.py      # Process raw text files -> all_documents.pkl
python src/build_vector_store.py # Upload to Pinecone
```

## 🎯 Key Features

- **Modern React UI**: Dark theme, responsive design.
- **RAG Pipeline**: Retrieval-Augmented Generation using Google Gemini & Pinecone.
- **Live Stats**: Real-time database statistics.
- **Advanced Filters**: Year range, tort types.
- **Rich Citations**: Expandable source excerpts with confidence scores.

## 🔧 Technology Stack

- **Backend**: FastAPI, Python 3.10, LangChain
- **Frontend**: React, Vite, Tailwind/CSS
- **AI/ML**: Google Gemini Pro, Pinecone (Serverless Vector DB)
- **Infrastructure**: Docker, Nginx

## 📝 License

Internal research project for legal professionals.

---
**Last Updated**: November 20, 2025
