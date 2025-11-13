# build_vector_store.py
import os
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
# Handle execution from either Processing/ or CaseLawGPT/ directory
if os.path.exists("processed_data"):
    PROCESSED_DATA_FILE = "processed_data/all_documents.pkl"
    VECTOR_STORE_PATH = "vector_store"
else:
    PROCESSED_DATA_FILE = "../processed_data/all_documents.pkl"
    VECTOR_STORE_PATH = "../vector_store"

def main():
    """
    Loads processed documents, generates embeddings, and saves the
    FAISS vector store to disk.
    """
    print("--- Building Vector Store ---")

    # 1. Load the processed documents
    print(f"[1/3] Loading processed documents from {PROCESSED_DATA_FILE}...")
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        all_documents = pickle.load(f)
    print(f"    Loaded {len(all_documents)} document chunks.")

    # 2. Initialize the embedding model
    print("[2/3] Initializing Google Generative AI Embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("    Model initialized.")

    # 3. Create and save the FAISS vector store in batches to avoid rate limiting
    print("[3/3] Creating FAISS vector store from documents...")
    print("    Using batched approach to avoid rate limits...")
    
    batch_size = 500
    total_batches = (len(all_documents) + batch_size - 1) // batch_size
    
    vector_store = None
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        print(f"    Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        try:
            if vector_store is None:
                # Create initial vector store
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                # Add to existing vector store
                batch_store = FAISS.from_documents(batch, embeddings)
                vector_store.merge_from(batch_store)
            
            # Brief pause to avoid rate limiting
            import time
            if i + batch_size < len(all_documents):
                time.sleep(2)
        except Exception as e:
            print(f"    ERROR in batch {batch_num}: {e}")
            print(f"    Continuing with remaining batches...")
            continue
    
    if vector_store is None:
        print("ERROR: Failed to create vector store!")
        return
    
    # Save the vector store locally
    print(f"    Saving vector store to '{VECTOR_STORE_PATH}'...")
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"--- Vector store created and saved successfully! ---")

if __name__ == "__main__":
    main()