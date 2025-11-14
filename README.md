# CaseLawGPT - Supreme Court Tort Law Research Tool

CaseLawGPT is a practical research tool for Indian Supreme Court tort law cases. It provides a modern web interface and backend for searching, filtering, and reading case law, built for lawyers and legal researchers.


## 📁 Project Structure

```
CaseLawGPT/
│
├── src/                # Backend (FastAPI, document processing)
├── frontend/           # React web interface
├── Processing/         # Document processing scripts
├── docs/               # Documentation and requirements
├── package.json        # Project manifest
├── README.md           # This file
├── vite.config.js      # Frontend config
```


## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Cloud credentials (for Gemini API)
- Case law text files (not included)

### Setup
```powershell
cd E:\CaseLawGPT
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r docs\requirements.txt
```

### Configure Credentials
Place your `google_credentials.json` in `.config/`.

### Process Documents & Build Vector Store
```powershell
cd src
python batch_process.py
python build_vector_store.py
```

### Run the Application
```powershell
cd frontend
npm install
npm run dev
```
Backend (FastAPI):
```powershell
cd src
python api.py
```
Open:
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs


## 🎯 Key Features

- Modern React UI: Dark theme, responsive design
- Live stats: Database statistics
- Interactive chat: Messaging experience
- Query suggestions: Example queries
- Advanced filters: Year range, tort types, search weights
- Rich citations: Expandable source excerpts


## 📊 Usage Examples

Web Interface:
1. Filter by year range (1950-2025)
2. Select tort types
3. Ask questions about tort law cases
4. View citations and source excerpts


## 🛠️ Development

### Rebuilding After Changes
```powershell
cd src
python build_vector_store.py
```


## 📚 Documentation

- **[QUICK_START.md](docs/QUICK_START.md)** - User guide


## 🔧 Technology Stack

- Backend: FastAPI, Python 3.10
- Frontend: React, Vite
- Vector Store: FAISS
- Embeddings: Google embedding-001


## 📈 Future Improvements

- Citation graph (precedent relationships)
- Export to PDF
- Improved filtering and UI


## 📝 License

Internal research project for legal professionals.


## 👥 Contact

For questions or improvements, contact the development team.

---

**Last Updated**: November 14, 2025  
**Version**: 2.1 (Cleaned for GitHub release)
