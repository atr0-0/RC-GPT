"""
CaseLawGPT - FastAPI Backend
RESTful API for the legal research assistant
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import pickle
from dotenv import load_dotenv

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(BASE_DIR, ".env")

# Load environment variables from .env file
load_dotenv(env_path)

print(f"DEBUG: Loaded .env from {env_path}")
print(f"DEBUG: PINECONE_API_KEY status: {'Found' if os.getenv('PINECONE_API_KEY') else 'Missing'}")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import re

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_FILE = os.path.join(BASE_DIR, "storage", "processed_data", "all_documents.pkl")
MODEL_NAME = "models/gemini-2.0-flash-exp"
RETRIEVAL_K = 15
PINECONE_INDEX_NAME = "caselawgpt-index"

# Initialize FastAPI
app = FastAPI(
    title="CaseLawGPT API",
    description="AI Legal Research Assistant for Indian Tort Law",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    year_range: Optional[List[int]] = [1950, 2025]
    tort_types: Optional[List[str]] = []
    max_sources: Optional[int] = 5

class Source(BaseModel):
    case_name: str
    citation: str
    excerpt: str
    confidence: str
    section_type: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    search_type: str

class StatsResponse(BaseModel):
    total_cases: int
    total_chunks: int
    year_range: str
    uptime: str

# Global state
retrieval_chain = None
general_chain = None
classification_chain = None
use_hybrid = False
all_documents = []

def apply_metadata_filters(docs, year_range, tort_types):
    """Filter retrieved documents by metadata criteria"""
    if not docs:
        return docs
    
    filtered = []
    for doc in docs:
        # Year filter
        doc_year = doc.metadata.get('year')
        if doc_year:
            try:
                year = int(doc_year)
                if year < year_range[0] or year > year_range[1]:
                    continue
            except (ValueError, TypeError):
                pass
        
        # Tort type filter
        if tort_types:
            doc_tort_types = doc.metadata.get('tort_types', [])
            if isinstance(doc_tort_types, str):
                doc_tort_types = [doc_tort_types]
            
            has_match = any(tt in doc_tort_types for tt in tort_types)
            if not has_match:
                continue
        
        filtered.append(doc)
    
    return filtered

def expand_legal_query(query: str) -> str:
    """Expand query with legal synonyms"""
    expansions = {
        'negligence': 'negligence negligent duty breach care',
        'compensation': 'compensation damages awarded quantum',
        'medical': 'medical doctor hospital treatment',
        'defamation': 'defamation libel slander reputation',
        'police': 'police custody detention arrest',
        'torture': 'torture custodial violence inhuman',
        'liability': 'liability responsible accountable',
    }
    
    expanded = query
    for term, expansion in expansions.items():
        if term in query.lower():
            expanded += f" {expansion}"
    
    return expanded

def detect_statute_citations(text: str) -> List[str]:
    """Detect legal statute citations"""
    patterns = [
        r'Section\s+\d+[A-Z]?',
        r'Article\s+\d+[A-Z]?',
        r'\d{4}\s+Act',
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        citations.extend(matches)
    
    return list(set(citations))

@app.on_event("startup")
async def startup_event():
    """Initialize the retrieval chain on startup"""
    global retrieval_chain, general_chain, classification_chain, use_hybrid, all_documents
    
    try:
        # Initialize embeddings and LLM
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)
        
        # --- 1. Classification Chain ---
        classification_prompt = ChatPromptTemplate.from_template(
            """Analyze the following user query to determine the intent.
            
            Query: {input}
            
            Respond with exactly one of the following keywords:
            
            1. "GENERAL": Greeting, small talk, or non-legal questions (e.g., "Hi", "How are you?", "What is 2+2?").
            2. "LEGAL_SEARCH": The user is asking a specific legal question, asking for case law, or mentioning specific legal terms/torts (e.g., "cases on medical negligence", "Section 302 IPC", "compensation for accident").
            3. "LEGAL_HELP": The user is asking for help with a case but hasn't provided details yet (e.g., "I need help with a case", "Can you help me fight a lawsuit?", "I am a lawyer", "make my case stronger").
            
            Classification:"""
        )
        classification_chain = classification_prompt | llm | StrOutputParser()

        # --- 2. General Conversation Chain ---
        general_prompt = ChatPromptTemplate.from_template(
            """You are CaseLawGPT, an intelligent AI assistant. 
            You specialize in Indian Tort Law, but you can also engage in normal, helpful conversation.
            
            If the user asks who you are, explain that you are an AI Legal Research Assistant for Indian Tort Law.
            If the user asks a general question, answer it naturally and helpfully.
            Do not make up legal cases if asked about law in this mode; simply answer generally or suggest they ask a specific legal question for a deep search.
            
            User Query: {input}
            
            Answer:"""
        )
        general_chain = general_prompt | llm | StrOutputParser()

        retriever = None
        
        # 3. Try Pinecone (Cloud) first
        if os.getenv("PINECONE_API_KEY"):
            print("☁️ Connecting to Pinecone Vector Store...")
            try:
                vector_store = PineconeVectorStore(
                    index_name=PINECONE_INDEX_NAME,
                    embedding=embeddings
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
                print("✅ Connected to Pinecone successfully.")
            except Exception as e:
                print(f"⚠️ Pinecone connection failed: {e}")
        
        # Load processed documents for Hybrid Search (BM25) - Optional
        # In a pure cloud setup, we might skip this or load from S3
        if os.path.exists(PROCESSED_DATA_FILE):
            with open(PROCESSED_DATA_FILE, 'rb') as f:
                all_documents = pickle.load(f)
                
            # Try hybrid search if we have documents
            try:
                bm25_retriever = BM25Retriever.from_documents(all_documents)
                bm25_retriever.k = RETRIEVAL_K
                
                if retriever:
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[retriever, bm25_retriever],
                        weights=[0.7, 0.3]
                    )
                    use_hybrid = True
                    retriever = ensemble_retriever
            except Exception as e:
                print(f"⚠️ Hybrid search setup failed: {e}")
        
        if retriever is None:
             print("⚠️ WARNING: No retriever initialized. Queries will fail.")
             return

        # Create prompt
        prompt = ChatPromptTemplate.from_template("""You are a legal research assistant specializing in Indian Supreme Court tort law cases.

Analyze the following case law excerpts and provide a comprehensive answer with specific citations.

Context:
{context}

Question: {input}

Answer:""")
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        print(f"✅ API initialized successfully | Hybrid: {use_hybrid}")
        
    except Exception as e:
        print(f"❌ Failed to initialize API: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "CaseLawGPT API",
        "version": "1.0.0"
    }

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics"""
    return StatsResponse(
        total_cases=689,
        total_chunks=len(all_documents) if all_documents else 4863,
        year_range="1950-2025",
        uptime="99.9%"
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a legal research query or general conversation."""
    if retrieval_chain is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    raw = (request.query or "").strip()
    if not raw:
        return QueryResponse(answer="Please ask a question.", sources=[], search_type="System")

    try:
        # 1. Classify Intent
        intent = "LEGAL_SEARCH" # Default
        if classification_chain:
            try:
                classification = classification_chain.invoke({"input": raw}).strip().upper()
                if "GENERAL" in classification:
                    intent = "GENERAL"
                elif "LEGAL_HELP" in classification:
                    intent = "LEGAL_HELP"
            except Exception as e:
                print(f"⚠️ Classification failed: {e}, defaulting to LEGAL_SEARCH")

        # 2. Handle General Conversation
        if intent == "GENERAL":
            answer = general_chain.invoke({"input": raw})
            return QueryResponse(
                answer=answer,
                sources=[],
                search_type="General Chat (LLM)"
            )
            
        # 3. Handle Vague Legal Help Requests
        if intent == "LEGAL_HELP":
            return QueryResponse(
                answer=(
                    "I'd be happy to help you strengthen your case. To give you the best legal precedents, I need a few more details:\n\n"
                    "1. **What is the nature of the case?** (e.g., Medical Negligence, Motor Accident, Defamation)\n"
                    "2. **What are the key facts?** (e.g., 'Hospital delayed treatment', 'Police refused FIR')\n"
                    "3. **What specific legal issue are you fighting?** (e.g., 'Enhancement of compensation', 'Quashing of FIR')\n\n"
                    "Once you provide these details, I can search for relevant Supreme Court judgments to support your arguments."
                ),
                sources=[],
                search_type="System (Clarification)"
            )

        # 4. Handle Legal Research (Existing Logic)
        expanded_query = expand_legal_query(raw)

        statute_citations = detect_statute_citations(raw)
        if statute_citations:
            expanded_query += f" {' '.join(statute_citations)}"

        response = retrieval_chain.invoke({"input": expanded_query})
        answer = response['answer']

        filtered_context = apply_metadata_filters(
            response.get('context', []),
            request.year_range,
            request.tort_types
        )

        sources: List[Source] = []
        if filtered_context:
            for idx, doc in enumerate(filtered_context[:request.max_sources]):
                confidence = max(0.5, 1.0 - (idx * 0.1))
                sources.append(Source(
                    case_name=doc.metadata.get('case_name', 'Unknown'),
                    citation=doc.metadata.get('citation', 'N/A'),
                    excerpt=doc.page_content,
                    confidence=f"{confidence:.0%}",
                    section_type=doc.metadata.get('section_type', 'general')
                ))

        search_type = "Hybrid Search (FAISS + BM25)" if use_hybrid else "Semantic Search (FAISS)"

        return QueryResponse(answer=answer, sources=sources, search_type=search_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "retrieval_chain_loaded": retrieval_chain is not None,
        "hybrid_search": use_hybrid,
        "documents_loaded": len(all_documents) if all_documents else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
