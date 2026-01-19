import os
import sys
from typing import List, Tuple
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- PATH CONFIG ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_index"

FETCH_K = int(os.getenv("FETCH_K", "4"))
# We match the chunk size from ingest (roughly) for display purposes
MAX_CHARS_PER_CHUNK = 2000

OPENAI_EMBED_MODEL = "text-embedding-3-small"


def _embeddings():
    """Must match the model used in ingest.py exactly."""
    return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)


def get_vector_store():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Could not find FAISS index at '{FAISS_INDEX_PATH}'. \n"
            "You likely need to run: python rag/ingest.py"
        )

    # DANGEROUS DESERIALIZATION WARNING:
    # FAISS uses pickle. Only set allow_dangerous_deserialization=True
    # if you created the index yourself (which you did in ingest.py).
    return FAISS.load_local(
        str(FAISS_INDEX_PATH),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )


def format_source_line(doc: Document) -> Tuple[str, str]:
    src = doc.metadata.get("source", "unknown")
    filename = os.path.basename(src)
    page = doc.metadata.get("page_label") or doc.metadata.get("page")
    return filename, str(page)


def build_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        filename, page = format_source_line(d)
        content = (d.page_content or "")[:MAX_CHARS_PER_CHUNK]
        blocks.append(f"Source [{i}] - {filename} (Page {page}):\n{content}")
    return "\n\n".join(blocks)


def search_docs(vector_store, query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=FETCH_K)