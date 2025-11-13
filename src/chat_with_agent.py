# chat_with_agent.py
import os
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- Configuration ---
VECTOR_STORE_PATH = "../storage/vector_store"
PROCESSED_DATA_FILE = "../storage/processed_data/all_documents.pkl"
MODEL_NAME = "models/gemini-2.0-flash-exp" # Using faster model for better experience
TEMPERATURE = 0.2  # Lower temperature for more factual responses
RETRIEVAL_K = 15  # Increased from 8 for better coverage

def main():
    """
    Loads the pre-built vector store and starts an interactive chat session with memory.
    """
    print("="*70)
    print("   CaseLawGPT - AI Legal Research Assistant for Indian Tort Law")
    print("="*70)

    # 1. Load the vector store and embeddings
    print("\n[1/4] Loading pre-built FAISS vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("    ✓ Vector store loaded successfully.")

    # 2. Load documents for BM25 (hybrid search)
    print("[2/4] Loading documents for hybrid retrieval...")
    try:
        with open(PROCESSED_DATA_FILE, 'rb') as f:
            all_documents = pickle.load(f)
        print(f"    ✓ Loaded {len(all_documents)} document chunks for BM25.")
    except Exception as e:
        print(f"    ! Could not load documents for BM25: {e}")
        print("    ! Falling back to FAISS-only retrieval.")
        all_documents = None

    # 3. Initialize the LLM and HYBRID Retriever
    print("[3/4] Initializing LLM and Hybrid Retriever (FAISS + BM25)...")
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE)
    
    # Create hybrid retriever (semantic + keyword search)
    if all_documents:
        # FAISS retriever (semantic)
        faiss_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
        # BM25 retriever (keyword matching)
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = RETRIEVAL_K
        
        # Combine: 70% semantic + 30% keyword for balanced results
        base_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        print("    ✓ Hybrid retriever (FAISS + BM25) initialized.")
    else:
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        print("    ✓ FAISS retriever initialized.")
    
    # Wrap with MultiQuery for better recall
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )
    print("    ✓ MultiQuery wrapper applied.")

    # 4. Create conversational RAG chain with memory
    print("[4/4] Creating conversational RAG chain with memory...")
    
    # Conversation memory
    chat_history = []
    
    # Enhanced prompt with conversation context
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are CaseLawGPT, an expert legal research assistant specializing in Indian Supreme Court tort law.

Your capabilities:
- Analyze case details and understand legal issues
- Search through 690+ Supreme Court tort law judgments
- Provide relevant case citations with precise legal reasoning
- Remember conversation context for follow-up questions

Rules you MUST follow:
1. Base answers ONLY on the provided case law context - never fabricate citations
2. Always cite sources with case name and citation: [Case Name, (Year) Volume SCC Page]
3. If the context doesn't contain the answer, clearly state this
4. For follow-up questions, reference previous parts of our conversation
5. Highlight the most relevant cases first
6. Explain legal principles in clear, professional language

Context from Supreme Court Cases:
{context}"""),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("    ✓ Conversational RAG chain ready.")
    
    print("\n" + "="*70)
    print("   Agent is ready! Ask about Indian tort law cases.")
    print("   Type 'quit' to exit | 'clear' to clear conversation history")
    print("="*70)

    # 5. Interactive chat loop with memory
    while True:
        question = input("\n🔍 Your Question: ")
        
        if question.lower() == 'quit':
            print("\n👋 Thank you for using CaseLawGPT. Goodbye!")
            break
        
        if question.lower() == 'clear':
            chat_history = []
            print("\n✓ Conversation history cleared.")
            continue
            
        if not question.strip():
            continue
        
        print("\n⚖️  Searching case law database...")
        
        # Build context from chat history
        context_input = question
        if chat_history:
            context_input = f"Previous conversation: {' '.join(chat_history[-4:])}\n\nCurrent question: {question}"
        
        response = retrieval_chain.invoke({"input": context_input})
        
        print("\n" + "-"*70)
        print("📋 Answer:")
        print("-"*70)
        print(response['answer'])
        
        # Show source count
        if 'context' in response:
            print(f"\n📚 Retrieved from {len(response['context'])} relevant case excerpts")
        
        # Update chat history
        chat_history.append(f"Q: {question}")
        chat_history.append(f"A: {response['answer'][:200]}...")  # Store summary
        
if __name__ == "__main__":
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
    else:
        main()