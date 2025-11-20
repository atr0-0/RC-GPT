# build_vector_store.py
import os
import pickle
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_FILE = os.path.join(BASE_DIR, "storage", "processed_data", "all_documents.pkl")
INDEX_NAME = "caselawgpt-index"

def main():
    """
    Loads processed documents and uploads them to Pinecone.
    """
    print("--- Building Cloud Vector Store (Pinecone) ---")

    # 0. Check Environment Variables
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY environment variable not set.")
        return

    # 1. Initialize Pinecone
    print("[1/4] Initializing Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists, if not create it
    existing_indexes = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"    Creating new index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768, # Dimension for Google embedding-001
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
    else:
        print(f"    Index '{INDEX_NAME}' already exists.")

    # 2. Load the processed documents
    print(f"[2/4] Loading processed documents from {PROCESSED_DATA_FILE}...")
    try:
        with open(PROCESSED_DATA_FILE, 'rb') as f:
            all_documents = pickle.load(f)
        print(f"    Loaded {len(all_documents)} document chunks.")
    except FileNotFoundError:
        print(f"ERROR: Could not find {PROCESSED_DATA_FILE}")
        return

    # 3. Initialize the embedding model
    print("[3/4] Initializing Google Generative AI Embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Upload to Pinecone
    print("[4/4] Uploading vectors to Pinecone...")
    print("    Using batched approach...")
    
    batch_size = 100
    total_batches = (len(all_documents) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        print(f"    Uploading batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        try:
            PineconeVectorStore.from_documents(
                batch, 
                embeddings, 
                index_name=INDEX_NAME
            )
            time.sleep(1) # Rate limit safety
        except Exception as e:
            print(f"    ERROR in batch {batch_num}: {e}")
            continue
            
    print(f"--- Build complete! Vectors uploaded to index '{INDEX_NAME}' ---")

if __name__ == "__main__":
    main()

