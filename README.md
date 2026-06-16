# RC-GPT — Supreme Court Tort Law Research Tool

RC-GPT is a RAG-powered legal research tool for Indian Supreme Court tort law cases. It provides a modern web interface for searching, filtering, and reading case law, built for lawyers and legal researchers.

## Project Structure

```
RC-GPT/
│
├── src/                # Backend (FastAPI) & Processing Scripts
├── frontend/           # React web interface (Vite + Tailwind)
├── storage/            # Data, Vector Store, and Processed files
├── misc/               # Research, Docs, Archive, Tests
├── requirements.txt    # Python dependencies
├── .env.example        # Required environment variables (copy to .env)
├── Dockerfile          # Backend Docker config
├── docker-compose.yml  # Container orchestration
├── setup.ps1           # One-click setup script
└── run.ps1             # One-click run script
```

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for local frontend development)
- **Google Gemini API Key** — get one free at https://aistudio.google.com/apikey
- **Pinecone API Key** — free Serverless tier at https://pinecone.io

## Quick Start (Local Dev)

1. **Copy and fill in environment variables:**
   ```powershell
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY and PINECONE_API_KEY
   ```

2. **Setup environment** (Python venv + Node modules):
   ```powershell
   .\setup.ps1
   ```

3. **Run the application** (backend + frontend):
   ```powershell
   .\run.ps1
   ```
   - Frontend: http://localhost:3000
   - Backend API docs: http://localhost:8000/docs

## Docker (Production / Easy Run)

```powershell
# Set keys in your shell first (or in a .env file):
$env:GOOGLE_API_KEY = "your-key"
$env:PINECONE_API_KEY = "your-key"

docker-compose up --build
```

Access the app at http://localhost:3000.

---

## Deploying to Production (Vercel + backend host)

The frontend is a static SPA and deploys to Vercel. The FastAPI backend needs a persistent host (Railway, Render, or Fly.io — all have free tiers).

### 1. Deploy the backend

Example with [Railway](https://railway.app):
```bash
railway login
railway init
railway up
```
Set `GOOGLE_API_KEY`, `PINECONE_API_KEY`, and `ALLOWED_ORIGINS` in Railway's environment settings.

Note the public URL Railway gives you (e.g. `https://rcgpt-backend-production.up.railway.app`).

### 2. Deploy the frontend to Vercel

In Vercel project settings → Environment Variables, add:
```
VITE_API_BASE_URL = https://rcgpt-backend-production.up.railway.app
```

Push to main — Vercel rebuilds automatically. The frontend will point all `/query` and `/health` calls at your hosted backend.

### 3. Update CORS

Make sure your backend's `ALLOWED_ORIGINS` env var includes the Vercel URL:
```
ALLOWED_ORIGINS=https://rcgpt.vercel.app,http://localhost:3000
```

---

## Cloud Vector Database

This project uses **Pinecone** as the vector database.

1. Sign up at [Pinecone.io](https://www.pinecone.io/) (free Serverless tier).
2. Add your key to `.env`: `PINECONE_API_KEY=your-key`
3. Build / upload the index (first time only):
   ```powershell
   .\venv\Scripts\Activate.ps1
   python src/batch_process.py       # Process raw text files → all_documents.pkl
   python src/build_vector_store.py  # Upload to Pinecone index
   ```

---

## Data Processing

Re-process raw text files if you add new case documents:

```powershell
.\venv\Scripts\Activate.ps1
python src/batch_process.py      # Process raw .txt files → storage/processed_data/all_documents.pkl
python src/build_vector_store.py # Upload chunks to Pinecone
```

---

## Key Features

- **Modern React UI**: Dark theme, responsive design, animated backgrounds.
- **RAG Pipeline**: Retrieval-Augmented Generation using Google Gemini & Pinecone.
- **Hybrid Search**: Combines Pinecone semantic search with BM25 keyword search.
- **Advanced Filters**: Year range, tort types, max sources.
- **Rich Citations**: Expandable source excerpts with confidence scores.
- **Intent Classification**: Distinguishes general chat from legal queries automatically.

## Technology Stack

- **Backend**: FastAPI, Python 3.10, LangChain 0.3+
- **Frontend**: React 18, Vite, Tailwind CSS, shadcn/ui
- **AI/ML**: Google Gemini (`gemini-2.0-flash`), Pinecone Serverless Vector DB
- **Infrastructure**: Docker, Nginx

## License

Internal research project for legal professionals.
