import re
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Configuration ---
FILE_PATH = 'jaywant_sankpal_v_suman_gholap.txt'
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250  # Increased from 150 to prevent fact splitting

def find_judgment_start_index(raw_text):
    """
    Finds the starting character index of the judgment body using multiple robust patterns.
    Tries patterns in order of reliability, with fallbacks.
    """
    # Multiple patterns in order of preference
    patterns = [
        # Pattern 1: Judge name with dash (most reliable)
        (r'\n([A-Z\s\.,]+J\.)[—─-]', "Judge name with dash"),
        # Pattern 2: More flexible judge pattern
        (r'\n([A-Z\s\.,]+,?\s*J\.)\s*[-—─:]', "Flexible judge pattern"),
        # Pattern 3: "JUDGMENT" or "ORDER" heading
        (r'\n(JUDGMENT|ORDER)\s*\n', "Judgment/Order heading"),
        # Pattern 4: "The Judgment of the Court"
        (r'(The Judgment of the Court was delivered|JUDGMENT\s*:)', "Court judgment phrase"),
        # Pattern 5: Numbered paragraph start after case details
        (r'\n1\.\s+[A-Z]', "Numbered paragraph"),
    ]
    
    for pattern, description in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE if 'JUDGMENT' in pattern else 0)
        if match:
            start_index = match.start()
            print(f"    Found judgment anchor using '{description}' at index {start_index}")
            return start_index
    
    # Ultimate fallback: Find case citation and start after it
    citation_match = re.search(r'\(\d{4}\)\s+\d+\s+(Supreme Court Cases|SCC)', raw_text)
    if citation_match:
        # Start 800 chars after citation (skip header details)
        start_index = citation_match.end() + 800
        print(f"    Using citation fallback, starting at index {start_index}")
        return min(start_index, len(raw_text) - 100)  # Safety check
    
    print("    WARNING: Could not find judgment anchor with any pattern. Using full text.")
    return 0  # Return 0 instead of -1 to process full text

def extract_metadata(header_text, full_text=""):
    """
    Extracts and cleans metadata from the header text and analyzes full text for legal metadata.
    """
    print("--> Extracting and cleaning metadata...")
    metadata = {}
    
    # 1. Case Name Extraction (The "Ntion" Fix)
    # Anchor between JJ.) and Versus
    pattern = re.compile(r'JJ\.\)\s*(.*?)\s*Versus', re.DOTALL)
    match = pattern.search(header_text)
    
    if match:
        raw_petitioner = match.group(1)
        
        # --- THE FIX: Explicitly remove noise words ---
        # Remove 'Petitioner', 'Appellant', and the OCR artifact 'Ntion' (case-insensitive)
        cleaned_petitioner = re.sub(r'(Petitioner|Appellant|Ntion)', '', raw_petitioner, flags=re.IGNORECASE)
        
        # Collapse multiple spaces/newlines into a single space and strip
        petitioner_name = ' '.join(cleaned_petitioner.split()).strip()

        # Get Respondent (simpler, usually cleaner)
        resp_pattern = re.compile(r'Versus\s+(.*?)\n')
        resp_match = resp_pattern.search(header_text)
        respondent_name = "UNKNOWN"
        if resp_match:
             # Remove 'AND OTHERS' for a cleaner short name, or keep it. Let's keep it.
             respondent_name = ' '.join(resp_match.group(1).split()).strip()

        metadata['case_name'] = f"{petitioner_name} v. {respondent_name}"

    # 2. Citation
    citation_pattern = re.compile(r'\((\d{4})\)\s+\d+\s+Supreme Court Cases\s+\d+')
    match = citation_pattern.search(header_text)
    if match: metadata['citation'] = match.group(0)

    # 3. Date
    date_pattern = re.compile(r'decided on (.*?)\n')
    match = date_pattern.search(header_text)
    if match: metadata['date_decided'] = match.group(1).strip()

    # 4. Judges
    judges_pattern = re.compile(r'\(BEFORE\s*(.*?),\s*JJ\.\)')
    match = judges_pattern.search(header_text)
    if match:
        judges_raw = match.group(1)
        metadata['judges'] = [name.strip() for name in judges_raw.split(' AND ')]
    
    # 5. ENHANCED: Extract Tort Types from full text
    if full_text:
        tort_keywords = {
            'negligence': r'\bnegligen(ce|t)\b',
            'defamation': r'\bdefam(ation|atory|e)\b',
            'trespass': r'\btrespass\b',
            'nuisance': r'\bnuisance\b',
            'assault': r'\bassault\b',
            'battery': r'\bbattery\b',
            'false_imprisonment': r'\bfalse imprisonment\b',
            'malicious_prosecution': r'\bmalicious prosecution\b',
            'conversion': r'\bconversion\b',
            'strict_liability': r'\bstrict liability\b',
            'vicarious_liability': r'\bvicarious liability\b',
        }
        
        tort_types = []
        for tort, pattern in tort_keywords.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                tort_types.append(tort)
        
        if tort_types:
            metadata['tort_types'] = tort_types
    
    # 6. ENHANCED: Extract cited statutes (IPC sections, Acts)
    if full_text:
        # Find IPC sections
        ipc_sections = re.findall(r'Section\s+\d+[A-Z]?\s+(?:of\s+)?(?:the\s+)?(?:IPC|Indian Penal Code)', full_text, re.IGNORECASE)
        # Find other acts
        acts = re.findall(r'(?:the\s+)?([A-Z][A-Za-z\s]+Act,?\s+\d{4})', full_text)
        
        all_statutes = list(set(ipc_sections[:5] + acts[:5]))  # Top 5 unique
        if all_statutes:
            metadata['statutes_cited'] = all_statutes
        
    print(f"    Extracted Case Name: {metadata.get('case_name', 'Extraction Failed')}")
    if 'tort_types' in metadata:
        print(f"    Identified Tort Types: {', '.join(metadata['tort_types'][:3])}")
    return metadata

# In process_doc.py

def clean_judgment_body(body_text):
    """
    Performs final cleanup on the judgment body to remove footers and OCR artifacts.
    """
    print("--> Cleaning judgment body...")
    
    # 1. Remove the main SCC ONLINE footer
    footer_pattern = re.compile(r'SC\s*\u24c7.*?True Print\u2122', re.DOTALL)
    cleaned_text = footer_pattern.sub('', body_text)
    
    # This removes the multi-line SCC header/footer that sometimes remains in chunks
    scc_header_pattern = re.compile(r'SCC Online Web Edition,.*?paras \d+, \d+ & \d+\.', re.DOTALL)
    cleaned_text = scc_header_pattern.sub('', body_text)

    # 2. Remove the "From the Judgment and Order..." footer
    # This regex is NOT greedy and only matches a single line. This is the fix.
    judgment_footer_pattern = re.compile(r'From the Judgment and Order dated.*')
    cleaned_text = judgment_footer_pattern.sub('', cleaned_text)

    # 3. Remove floating single letters at the end of lines
    cleaned_text = re.sub(r'\s+[a-zA-Z]\n', '\n', cleaned_text)

    # 4. Remove page markers like "h", "g" etc. at the start of lines
    cleaned_text = re.sub(r'\n[fgh]\n', '\n', cleaned_text)

    # 5. Consolidate whitespace
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()


def create_document_chunks(body_text, metadata, file_path):
    """
    Splits the cleaned BODY text into chunks using semantic section awareness.
    """
    print("--> Splitting document body into chunks (semantic mode)...")
    
    # Legal document section markers for better semantic chunking
    legal_separators = [
        "\n\nFACTS:\n",
        "\n\nHELD:\n",
        "\n\nRATIO:\n",
        "\n\nREASONS:\n",
        "\n\nJUDGMENT:\n",
        "\n\nORDER:\n",
        "\n\n\n",  # Triple newline (major section break)
        "\n\n",    # Double newline (paragraph break)
        "\n",      # Single newline
        ". ",      # Sentence end
        " ",       # Word break
        ""         # Character break (last resort)
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=legal_separators
    )
    chunks = text_splitter.split_text(body_text)
    
    documents = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = metadata.copy()
        chunk_metadata['source'] = file_path
        chunk_metadata['chunk_number'] = i + 1
        
        # Tag chunk with section type for smarter retrieval
        chunk_lower = chunk.lower()
        if 'facts' in chunk_lower[:100] or 'appellant' in chunk_lower[:100]:
            chunk_metadata['section_type'] = 'facts'
        elif 'held' in chunk_lower[:100] or 'judgment' in chunk_lower[:100]:
            chunk_metadata['section_type'] = 'judgment'
        elif 'ratio' in chunk_lower[:100] or 'reason' in chunk_lower[:100]:
            chunk_metadata['section_type'] = 'reasoning'
        else:
            chunk_metadata['section_type'] = 'general'
        
        doc = Document(page_content=chunk, metadata=chunk_metadata)
        documents.append(doc)
        
    print(f"    Split document body into {len(documents)} chunks.")
    return documents

def main():
    print(f"--- Starting processing for: {FILE_PATH} ---")

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # 1. Slice
    start_index = find_judgment_start_index(raw_text)
    # Note: start_index now returns 0 as fallback instead of -1

    header_text = raw_text[:start_index] if start_index > 0 else raw_text[:2000]
    raw_body_text = raw_text[start_index:]
    print("    Successfully sliced document.")

    # 2. Extract & Clean Metadata (pass full text for enhanced extraction)
    metadata = extract_metadata(header_text, raw_text)
    
    # 3. Clean Body
    cleaned_body = clean_judgment_body(raw_body_text)

    # 4. Chunk
    documents = create_document_chunks(cleaned_body, metadata, FILE_PATH)
    
    print("\n--- VERIFICATION COMPLETE ---")
    print(f"Successfully processed '{metadata.get('case_name')}' into {len(documents)} documents.")
    
    print("\n--- Example of First Document Object ---")
    if documents:
        print(json.dumps(documents[0].model_dump(), indent=2))

if __name__ == "__main__":
    main()