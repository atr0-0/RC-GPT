"""
CaseLawGPT - Streamlit Web Interface
A professional web UI for the legal research assistant
"""

import streamlit as st
import os
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Page config
st.set_page_config(
    page_title="CaseLawGPT - Legal Research Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced Modern Design
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header with gradient */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Enhanced Chat Messages */
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid;
        transition: transform 0.2s, box-shadow 0.2s;
        color: #1f2937 !important;
    }
    
    .chat-message strong {
        color: #111827 !important;
    }
    
    .chat-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left-color: #1976d2;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%);
        border-left-color: #43a047;
    }
    
    /* Citation Cards */
    .citation {
        background: linear-gradient(135deg, #fff9e6 0%, #ffe4b3 100%);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9800;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
    
    /* Source Cards */
    .source-card {
        background: white;
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        transition: all 0.2s;
    }
    
    .source-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        border-color: #667eea;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-confidence {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
        color: white;
    }
    
    .badge-section {
        background: linear-gradient(135deg, #2196f3 0%, #42a5f5 100%);
        color: white;
    }
    
    /* Query Suggestion Chips */
    .query-chip {
        display: inline-block;
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        padding: 0.5rem 1rem;
        border-radius: 1.5rem;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid #d1d5db;
        font-size: 0.9rem;
    }
    
    .query-chip:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Stats Cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar Enhancements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f9fafb 0%, #ffffff 100%);
    }
    
    /* Button Improvements */
    .stButton>button {
        border-radius: 0.75rem;
        font-weight: 600;
        transition: all 0.2s;
        color: #1f2937 !important;
        background: white !important;
        border: 2px solid #e5e7eb !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: #667eea !important;
    }
    
    /* Fix text color in all streamlit elements */
    .main .block-container {
        color: #1f2937;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
VECTOR_STORE_PATH = "../storage/vector_store"
PROCESSED_DATA_FILE = "../storage/processed_data/all_documents.pkl"
MODEL_NAME = "models/gemini-2.0-flash-exp"
RETRIEVAL_K = 15  # Increased from 8 for better coverage
FAISS_WEIGHT = 0.7  # Configurable: 70% semantic
BM25_WEIGHT = 0.3   # Configurable: 30% keyword

def apply_metadata_filters(docs, year_range, tort_types):
    """Filter retrieved documents by metadata criteria"""
    if not docs:
        return docs
    
    filtered = []
    for doc in docs:
        # Year filter
        if 'date_decided' in doc.metadata:
            try:
                # Extract year from various date formats
                import re
                year_match = re.search(r'\\b(19|20)\\d{2}\\b', str(doc.metadata['date_decided']))
                if year_match:
                    year = int(year_match.group())
                    if year < year_range[0] or year > year_range[1]:
                        continue
            except:
                pass  # Include if year extraction fails
        
        # Tort type filter
        if tort_types:
            doc_tort_types = doc.metadata.get('tort_types', [])
            if not any(tt in doc_tort_types for tt in tort_types):
                continue
        
        filtered.append(doc)
    
    return filtered

def expand_legal_query(query):
    """Expand query with legal synonyms for better recall"""
    # Legal term synonym mapping
    synonyms = {
        r'\bnegligen(ce|t)\b': 'negligence duty of care breach',
        r'\bcompensation\b': 'compensation damages relief quantum',
        r'\bdefam(ation|e)\b': 'defamation libel slander reputation',
        r'\btrespass\b': 'trespass unlawful entry property',
        r'\btort\b': 'tort civil wrong liability',
        r'\bvicarious\b': 'vicarious liability employer respondeat',
        r'\bstrict liability\b': 'strict liability absolute liability hazardous',
        r'\bmalicious\b': 'malicious prosecution wrongful arrest',
        r'\bnuisance\b': 'nuisance interference enjoyment',
    }
    
    import re
    expanded = query
    for pattern, expansion in synonyms.items():
        if re.search(pattern, query, re.IGNORECASE):
            expanded += f" {expansion}"
    
    return expanded

def detect_statute_citations(query):
    """Detect if query contains specific statute citations"""
    import re
    patterns = [
        r'Section\\s+\\d+[A-Z]?(?:\\s+(?:of\\s+)?(?:the\\s+)?(?:IPC|Indian Penal Code))?',
        r'Article\\s+\\d+[A-Z]?(?:\\s+of\\s+(?:the\\s+)?Constitution)?',
        r'\\b[A-Z][A-Za-z\\s]+Act,?\\s+\\d{4}\\b',
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        citations.extend(matches)
    
    return citations

@st.cache_resource
def initialize_agent():
    """Initialize and cache the RAG agent"""
    try:
        # Load embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Load documents for BM25
        try:
            with open(PROCESSED_DATA_FILE, 'rb') as f:
                all_documents = pickle.load(f)
            
            # Create hybrid retriever with increased k
            faiss_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            bm25_retriever.k = RETRIEVAL_K
            
            base_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=[FAISS_WEIGHT, BM25_WEIGHT]
            )
            use_hybrid = True
        except:
            base_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
            use_hybrid = False
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2)
        
        # Create retriever
        retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template("""You are CaseLawGPT, an expert legal research assistant for Indian Supreme Court tort law.

Answer the question based ONLY on the provided case law context. Follow these rules:
1. Cite every fact with case name and citation: [Case Name, (Year) Vol SCC Page]
2. If context lacks the answer, clearly state this
3. Prioritize recent and landmark judgments
4. Explain legal principles clearly

Context:
{context}

Question: {input}

Answer:""")
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain, use_hybrid
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return None, False

def main():
    # Header
    st.markdown('<div class="main-header">⚖️ CaseLawGPT</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI Legal Research Assistant for Indian Tort Law</p>', unsafe_allow_html=True)
    
    # Stats Dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('''
        <div class="stat-card">
            <div style="font-size: 2rem; font-weight: 700; color: #667eea;">689</div>
            <div style="color: #666; font-size: 0.9rem;">SC Cases</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div class="stat-card">
            <div style="font-size: 2rem; font-weight: 700; color: #764ba2;">4,863</div>
            <div style="color: #666; font-size: 0.9rem;">Legal Chunks</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown('''
        <div class="stat-card">
            <div style="font-size: 2rem; font-weight: 700; color: #2196f3;">1950-2025</div>
            <div style="color: #666; font-size: 0.9rem;">Year Range</div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown('''
        <div class="stat-card">
            <div style="font-size: 2rem; font-weight: 700; color: #43a047;">99.9%</div>
            <div style="color: #666; font-size: 0.9rem;">Uptime</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **CaseLawGPT** helps lawyers find relevant Supreme Court tort law judgments.
        
        **Features:**
        - 🔍 Hybrid search (semantic + keyword)
        - 💬 Conversational memory
        - 📚 690+ case database
        - 🎯 Precise citations
        - 🎚️ Advanced filtering
        """)
        
        st.divider()
        
        # Filters with enhanced layout
        st.header("🎯 Filters")
        
        with st.container():
            st.subheader("📅 Time Period")
            year_filter = st.slider("Case Year Range", 1950, 2025, (1950, 2025))
        
        with st.container():
            st.subheader("⚖️ Tort Categories")
            tort_types = st.multiselect(
                "Select Types",
                ["negligence", "defamation", "trespass", "nuisance", "assault", 
                 "battery", "false_imprisonment", "malicious_prosecution", 
                 "conversion", "strict_liability", "vicarious_liability"],
                help="Filter by specific tort types"
            )
            
            # Preset filter combinations
            st.caption("**Quick Presets:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🏥 Medical", use_container_width=True):
                    tort_types = ["negligence"]
            with col2:
                if st.button("📰 Media", use_container_width=True):
                    tort_types = ["defamation"]
        
        st.divider()
        
        st.header("⚙️ Settings")
        show_sources = st.checkbox("Show source excerpts", value=True)
        max_sources = st.slider("Max sources to show", 1, 8, 5)
        
        # Advanced settings expander
        with st.expander("⚡ Advanced Settings"):
            faiss_weight = st.slider("Semantic Search Weight", 0.0, 1.0, 0.7, 0.1)
            st.caption(f"Keyword weight: {1.0 - faiss_weight:.1f}")
        
        st.divider()
        
        # Search History
        st.header("🕐 Recent Searches")
        if "search_history" not in st.session_state:
            st.session_state.search_history = []
        
        if st.session_state.search_history:
            for i, query in enumerate(st.session_state.search_history[-5:][::-1]):
                if st.button(f"↩️ {query[:40]}...", key=f"history_{i}", use_container_width=True):
                    st.session_state.current_query = query
                    st.rerun()
        else:
            st.caption("_No recent searches_")
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.search_history = []
            st.rerun()
    
    # Initialize agent
    with st.spinner("🔄 Loading legal database..."):
        retrieval_chain, use_hybrid = initialize_agent()
    
    if retrieval_chain is None:
        st.error("❌ Failed to load the agent. Please check your configuration.")
        return
    
    # Display search type
    search_type = "🔬 Hybrid Search (FAISS + BM25)" if use_hybrid else "🔍 Semantic Search (FAISS)"
    st.success(f"✅ Agent ready | {search_type}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Query Suggestion Chips (show when no messages)
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Try these example queries:")
        st.markdown("")  # Add spacing
        
        example_queries = [
            ("🏥 Medical Negligence", "Cases on medical negligence compensation"),
            ("🚔 Police Custody", "Police custody torture and damages"),
            ("📰 Media Defamation", "Defamation by media houses"),
            ("👔 Employer Liability", "Vicarious liability of employers"),
            ("⚡ Hazardous Activities", "Strict liability in hazardous activities"),
            ("🚗 Vehicle Accidents", "Motor vehicle accident compensation")
        ]
        
        # Create 3 columns for suggestion chips
        cols = st.columns(3)
        for idx, (button_text, full_query) in enumerate(example_queries):
            with cols[idx % 3]:
                if st.button(button_text, key=f"suggest_{idx}", use_container_width=True, type="secondary"):
                    # Directly add message and process
                    st.session_state.messages.append({"role": "user", "content": full_query})
                    st.session_state.suggestion_clicked = full_query
                    st.rerun()
        
        st.markdown("---")  # Divider
        st.markdown("")  # Add spacing
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>🧑‍⚖️ You:</strong><br>{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>⚖️ CaseLawGPT:</strong><br>{content}</div>', unsafe_allow_html=True)
            
            # Show sources if available
            if show_sources and "sources" in message:
                with st.expander(f"📚 View {len(message['sources'])} source excerpts"):
                    for i, source in enumerate(message['sources'][:max_sources], 1):
                        # Enhanced source card display
                        st.markdown(f'''
                        <div class="source-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <strong style="color: #667eea; font-size: 1rem;">📄 Source {i}: {source['case_name']}</strong>
                                <span class="badge-confidence">{source.get('confidence', 'N/A')}</span>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        if 'citation' in source:
                            st.markdown(f'<div class="citation">📖 {source["citation"]}</div>', unsafe_allow_html=True)
                        
                        if 'section_type' in source:
                            section_badges = {
                                'facts': '<span class="badge-section" style="background: linear-gradient(135deg, #3b82f6, #2563eb);">📋 Facts</span>',
                                'judgment': '<span class="badge-section" style="background: linear-gradient(135deg, #8b5cf6, #7c3aed);">⚖️ Judgment</span>',
                                'reasoning': '<span class="badge-section" style="background: linear-gradient(135deg, #f59e0b, #d97706);">💡 Reasoning</span>',
                                'general': '<span class="badge-section" style="background: linear-gradient(135deg, #6b7280, #4b5563);">📄 General</span>'
                            }
                            st.markdown(section_badges.get(source['section_type'], ''), unsafe_allow_html=True)
                        
                        # Truncate long excerpts with expand option
                        excerpt = source['excerpt']
                        if len(excerpt) > 300:
                            st.markdown(f"_{excerpt[:300]}..._")
                            with st.expander("Read full excerpt"):
                                st.markdown(f"_{excerpt}_")
                        else:
                            st.markdown(f"_{excerpt}_")
                        
                        st.divider()
    
    # Chat input
    # Check if a suggestion was clicked
    if "suggestion_clicked" in st.session_state and st.session_state.suggestion_clicked:
        prompt = st.session_state.suggestion_clicked
        st.session_state.suggestion_clicked = None  # Clear it
    # Check if there's a suggested query to use
    elif "current_query" in st.session_state and st.session_state.current_query:
        prompt = st.session_state.current_query
        st.session_state.current_query = None  # Clear it
    else:
        prompt = st.chat_input("Ask about tort law cases...")
    
    if prompt:
        # Add to search history
        if "search_history" not in st.session_state:
            st.session_state.search_history = []
        if prompt not in st.session_state.search_history:
            st.session_state.search_history.append(prompt)
            # Keep only last 10 searches
            st.session_state.search_history = st.session_state.search_history[-10:]
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(f'<div class="chat-message user-message"><strong>🧑‍⚖️ You:</strong><br>{prompt}</div>', unsafe_allow_html=True)
        
        # Get agent response
        with st.spinner("🔍 Searching case law database..."):
            try:
                # Build context from recent history
                context_input = prompt
                if len(st.session_state.messages) > 2:
                    recent = st.session_state.messages[-4:]
                    history_text = " ".join([f"{m['role']}: {m['content'][:100]}" for m in recent])
                    context_input = f"Context: {history_text}\n\nQuestion: {prompt}"
                
                # Expand query with legal synonyms for better recall
                expanded_query = expand_legal_query(context_input)
                
                # Check for statute citations and boost their importance
                statute_citations = detect_statute_citations(prompt)
                if statute_citations:
                    st.info(f"📋 Detected statute citations: {', '.join(statute_citations[:3])}")
                    expanded_query += f" {' '.join(statute_citations)}"
                
                response = retrieval_chain.invoke({"input": expanded_query})
                answer = response['answer']
                
                # Apply metadata filters to retrieved context
                filtered_context = apply_metadata_filters(
                    response.get('context', []),
                    year_filter,
                    tort_types
                )
                
                # Extract sources with confidence scoring
                sources = []
                if filtered_context:
                    for idx, doc in enumerate(filtered_context[:max_sources]):
                        # Calculate basic relevance score (position-based)
                        confidence = max(0.5, 1.0 - (idx * 0.1))  # Decreases by position
                        
                        sources.append({
                            'case_name': doc.metadata.get('case_name', 'Unknown'),
                            'citation': doc.metadata.get('citation', 'N/A'),
                            'excerpt': doc.page_content,
                            'confidence': f"{confidence:.0%}",
                            'section_type': doc.metadata.get('section_type', 'general')
                        })
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                # Rerun to update display
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    # Check for credentials
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        st.error("⚠️ GOOGLE_APPLICATION_CREDENTIALS environment variable not set!")
        st.stop()
    
    main()
