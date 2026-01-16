import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PDF_DIR = "documents"
INDEX_NAME = "ragchatbot"
NAMESPACE = "policies_uk_car_v1"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

pc = Pinecone()
index = pc.Index(INDEX_NAME)

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index=index,
    namespace=NAMESPACE
)


def clean_text(text: str) -> str:
    """Normalizes text by removing excessive whitespace, newlines, and non-printable characters."""
    if not text:
        return ""
    text = re.sub(r'[\n\t\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_pdfs(pdf_dir: str):
    """Iterates through the directory, loads PDF files, and applies the cleaning function to every page."""
    docs = []
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"Directory '{pdf_dir}' does not exist.")

    pdfs = sorted([n for n in os.listdir(pdf_dir) if n.lower().endswith(".pdf")])

    for name in pdfs:
        path = os.path.join(pdf_dir, name)
        try:
            loader = PyPDFLoader(path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.page_content = clean_text(doc.page_content)

            docs.extend(loaded_docs)
            print(f"OK & Cleaned: {name}")
        except Exception as e:
            print(f"FAIL: {name} -> {type(e).__name__}: {e}")
            continue

    return docs


print("--- Starting Ingestion Pipeline ---")
raw_docs = load_pdfs(PDF_DIR)

if not raw_docs:
    raise RuntimeError(f"No PDF pages loaded from '{PDF_DIR}'. Check the folder path and PDFs.")

# Assign metadata to all documents for filtering
for d in raw_docs:
    d.metadata["country"] = "uk"
    d.metadata["product"] = "private_car"
    d.metadata["dataset"] = "uk_private_car_policy_wordings_v1"

# Split documents into smaller chunks for embedding
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(raw_docs)

# Filter out chunks that are too short or truncate those that are too long
MAX_CHARS = 2000
MIN_CHARS = 50

safe_chunks = []
skipped_short = 0
truncated = 0

for c in chunks:
    text = c.page_content or ""
    if len(text) < MIN_CHARS:
        skipped_short += 1
        continue
    if len(text) > MAX_CHARS:
        c.page_content = text[:MAX_CHARS]
        truncated += 1
    safe_chunks.append(c)

print(f"Chunks total={len(chunks)} | safe={len(safe_chunks)} | truncated={truncated} | skipped_short={skipped_short}")

# Upload chunks to the vector store in batches, retrying individually if a batch fails
BATCH_SIZE = 32
skipped_embed = 0

print("--- Uploading to Vector Store ---")
for i in range(0, len(safe_chunks), BATCH_SIZE):
    batch = safe_chunks[i:i + BATCH_SIZE]
    try:
        vector_store.add_documents(batch)
        print(f"Upserted batch {i // BATCH_SIZE + 1} ({i}..{i + len(batch) - 1})")
    except Exception as e:
        print(f"Batch failed at {i}..{i + len(batch) - 1} -> {type(e).__name__}: {e}")
        for doc in batch:
            try:
                vector_store.add_documents([doc])
            except Exception as e2:
                skipped_embed += 1
                meta = doc.metadata
                print(
                    f"SKIP chunk -> {type(e2).__name__}: {e2} | "
                    f"source={meta.get('source')} page={meta.get('page')}"
                )

print(
    f"DONE. Pages={len(raw_docs)} | chunks_created={len(chunks)} | chunks_ingested~={len(safe_chunks) - skipped_embed} "
    f"| skipped_embed={skipped_embed}"
)