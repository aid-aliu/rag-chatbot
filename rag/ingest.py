import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. SETUP & PATHS ---
# We use pathlib for robust path handling
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

# Verify API Key is loaded
if not os.getenv("OPENAI_API_KEY"):
    print(f"ERROR: OPENAI_API_KEY not found in {ENV_PATH}")
    sys.exit(1)

PDF_DIR = os.getenv("PDF_DIR", PROJECT_ROOT / "documents")
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_index"

# OpenAI's text-embedding-3-small is standard now (1536 dimensions)
OPENAI_EMBED_MODEL = "text-embedding-3-small"


def clean_text(text: str) -> str:
    """Helper to remove excessive whitespace from PDF extraction."""
    if not text:
        return ""
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdfs(pdf_dir: Path):
    docs = []
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory '{pdf_dir}' does not exist.")

    # Find all PDFs in the directory
    pdfs = sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])
    print(f"Found {len(pdfs)} PDFs in {pdf_dir}")

    for name in pdfs:
        file_path = pdf_dir / name
        try:
            loader = PyPDFLoader(str(file_path))
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = name
            docs.extend(loaded_docs)
            print(f"OK: {name} ({len(loaded_docs)} pages)")
        except Exception as e:
            print(f"FAIL: {name} -> {e}")

    return docs


if __name__ == "__main__":
    print(f"--- 1. Loading PDFs from {PDF_DIR} ---")
    raw_docs = load_pdfs(Path(PDF_DIR))

    if not raw_docs:
        print("No documents found. Exiting.")
        sys.exit(0)

    print(f"--- 2. Splitting {len(raw_docs)} pages ---")
    # OpenAI models have a larger context window, so we can slightly increase chunk size
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} chunks.")

    print(f"--- 3. Creating Vector Store with {OPENAI_EMBED_MODEL} ---")

    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBED_MODEL
    )

    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(str(FAISS_INDEX_PATH))
        print(f"SUCCESS: Index saved to '{FAISS_INDEX_PATH}'")
    except Exception as e:
        print(f"ERROR: Could not create/save index. {e}")