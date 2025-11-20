# batch_process.py
import os
import glob
import pickle
from typing import List
from dotenv import load_dotenv
import process_doc as phase1

# Load environment variables
load_dotenv()

from langchain_core.documents import Document

# --- Configuration ---
# Paths for new organized structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "storage", "data")
OUTPUT_FILE = os.path.join(BASE_DIR, "storage", "processed_data", "all_documents.pkl")

def main():
    """
    Processes all .txt files in the data directory and saves the combined
    list of Document objects to a pickle file.
    """
    # Use glob to find all .txt files in the data directory
    file_paths = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    print(f"--- Found {len(file_paths)} documents to process. ---")

    all_documents = []
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, file_path in enumerate(file_paths):
        print(f"\nProcessing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            start_index = phase1.find_judgment_start_index(raw_text)
            # Now start_index returns 0 as fallback, not -1
            
            header_text = raw_text[:start_index] if start_index > 0 else raw_text[:2000]
            raw_body_text = raw_text[start_index:]
            
            # Pass full text for enhanced metadata extraction
            metadata = phase1.extract_metadata(header_text, raw_text)
            # If case name extraction fails, use the filename as a fallback
            if 'case_name' not in metadata or not metadata['case_name']:
                 metadata['case_name'] = os.path.basename(file_path)

            cleaned_body = phase1.clean_judgment_body(raw_body_text)
            
            # Skip if cleaned body is too short (likely extraction failed)
            if len(cleaned_body) < 100:
                print(f"    [!] Skipping file: Extracted content too short ({len(cleaned_body)} chars)")
                skip_count += 1
                continue
            
            documents = phase1.create_document_chunks(cleaned_body, metadata, file_path)
            
            all_documents.extend(documents)
            success_count += 1
            print(f"    [OK] Successfully processed and added {len(documents)} chunks.")

        except Exception as e:
            print(f"    [X] ERROR processing file {file_path}: {e}")
            error_count += 1

    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total files found: {len(file_paths)}")
    print(f"Successfully processed: {success_count} ({success_count/len(file_paths)*100:.1f}%)")
    print(f"Skipped (too short): {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Total chunks created: {len(all_documents)}")
    print(f"Average chunks per file: {len(all_documents)/success_count:.1f}" if success_count > 0 else "N/A")

    # Save the combined list of documents to a file using pickle
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_documents, f)

    print(f"\n--- Batch processing complete! All documents saved to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    main()