# Directory Structure Reference

## 📂 Folder Purposes

### `/src/` - Core Application
**Purpose**: All runnable Python application code  
**Files**:
- `app.py` - Streamlit web interface (main entry point for UI)
- `process_doc.py` - Single document processing logic
- `batch_process.py` - Process all 690 case files
- `build_vector_store.py` - Create FAISS embeddings
- `chat_with_agent.py` - CLI interface for testing

**Usage**:
```powershell
cd src
python batch_process.py        # Step 1: Process documents
python build_vector_store.py   # Step 2: Build embeddings
python chat_with_agent.py      # Step 3: Test CLI
streamlit run app.py           # Step 4: Launch web UI
```

---

### `/storage/` - Data & Embeddings
**Purpose**: All data files (raw, processed, indexed)  
**Subfolders**:
- `data/` - 690 raw .txt case files (OCR from PDFs)
- `processed_data/` - Pickle file with 4,863 LangChain documents
- `vector_store/` - FAISS index files (index.faiss, index.pkl)

**Size**: ~500 MB total
**Note**: Excluded from git (see .gitignore)

---

### `/docs/` - Documentation
**Purpose**: All project documentation and guides  
**Files**:
- `PROJECT_PIPELINE.md` - Complete technical overview
- `TECHNICAL_IMPROVEMENTS.md` - Implementation details
- `BEFORE_AFTER_COMPARISON.md` - Performance metrics
- `IMPLEMENTATION_SUMMARY.md` - Development timeline
- `QUICK_START.md` - User guide
- `requirements.txt` - Python dependencies

**Usage**: Read these for understanding the system

---

### `/tests/` - Testing & Debugging
**Purpose**: Test cases, sample data, debug scripts  
**Files**:
- `jaywant_sankpal_v_suman_gholap.txt` - Real case for testing
- `test_case_story.md` - Fictional test scenario
- `test_regex.py` - Test judgment extraction patterns
- `test_retrieval.py` - Test retrieval accuracy
- `analyze_data.py` - Data analysis utilities
- Various debug scripts from development

**Usage**:
```powershell
cd tests
python test_retrieval.py       # Test if queries work
python test_regex.py            # Test document parsing
```

---

### `/scripts/` - Utility Scripts
**Purpose**: Automation and helper scripts  
**Files**:
- `rebuild_vectorstore.ps1` - Full rebuild pipeline
- `start_webapp.ps1` - Launch Streamlit with proper env

**Usage**:
```powershell
cd scripts
.\rebuild_vectorstore.ps1      # Rebuild everything
.\start_webapp.ps1             # Start web interface
```

---

### `/.config/` - Configuration
**Purpose**: Configuration files and credentials  
**Files**:
- `google_credentials.json` - Google Cloud API key

**Note**: **DO NOT commit to git!** (sensitive data)

---

### `/archive/` - Old Files
**Purpose**: Deprecated/backup files for reference  
**Contents**:
- `Processing/` - Old processing folder structure
- `data_old/` - Backup of old data
- `misc/` - Old debug scripts

**Note**: Can be deleted after confirming new structure works

---

## 🔄 Typical Workflows

### First Time Setup
```powershell
# 1. Install dependencies
pip install -r docs\requirements.txt

# 2. Process documents
cd src
python batch_process.py

# 3. Build vector store
python build_vector_store.py

# 4. Launch web UI
cd ..\scripts
.\start_webapp.ps1
```

### After Code Changes
```powershell
# If you modified processing logic
cd src
python batch_process.py
python build_vector_store.py

# If you only modified UI
cd scripts
.\start_webapp.ps1
```

### After Adding New Cases
```powershell
# 1. Add .txt files to storage/data/
# 2. Rebuild everything
cd scripts
.\rebuild_vectorstore.ps1
```

### Testing Retrieval
```powershell
cd tests
python test_retrieval.py
```

---

## 📊 File Size Reference

| Folder | Approximate Size | Excluded from Git? |
|--------|-----------------|-------------------|
| `/storage/data/` | ~200 MB | ✅ Yes |
| `/storage/processed_data/` | ~6 MB | ✅ Yes |
| `/storage/vector_store/` | ~300 MB | ✅ Yes |
| `/src/` | <1 MB | ❌ No (code) |
| `/docs/` | <1 MB | ❌ No (docs) |
| `/tests/` | ~5 MB | ⚠️ Partial |
| `/venv/` | ~500 MB | ✅ Yes |
| **Total** | ~1 GB | - |

---

## 🚨 Important Notes

1. **Never commit** `.config/google_credentials.json` to git
2. **Storage folder** is regenerable - can be excluded from backups
3. **Archive folder** can be deleted once new structure is validated
4. **All imports** now use relative paths from `src/` directory
5. **Scripts** must be run from their respective folders

---

## 🔧 Path Reference for Code

When writing code in `/src/`:
```python
# Correct paths (relative to src/)
VECTOR_STORE_PATH = "../storage/vector_store"
PROCESSED_DATA_FILE = "../storage/processed_data/all_documents.pkl"
DATA_DIR = "../storage/data"
CREDENTIALS = "../.config/google_credentials.json"
```

When running from project root:
```python
# Correct paths (relative to root)
VECTOR_STORE_PATH = "storage/vector_store"
PROCESSED_DATA_FILE = "storage/processed_data/all_documents.pkl"
DATA_DIR = "storage/data"
CREDENTIALS = ".config/google_credentials.json"
```

---

**Last Updated**: November 13, 2025
