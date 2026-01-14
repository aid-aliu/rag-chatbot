from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "ragchatbot"
NAMESPACE = "policies_uk_car_v1"

llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    temperature=0,
)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

pc = Pinecone()
index = pc.Index(INDEX_NAME)

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index=index,
    namespace=NAMESPACE
)

def build_context(docs):
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page_label") or d.metadata.get("page")
        blocks.append(
            f"[{i}] {src} | page {page}\n{d.page_content}"
        )
    return "\n\n".join(blocks)

def ask(query: str, k: int = 6):
    docs = vector_store.similarity_search(query, k=k)
    context = build_context(docs)

    prompt = f"""
You answer questions about UK private car insurance policy wordings.

Use ONLY the provided context.

Disambiguation rule:
- If the question is ambiguous (e.g., could mean theft of the vehicle vs theft of the keys / key cover),
  assume the user means THEFT OF THE VEHICLE unless they explicitly mention
  "key cover", "key protection", or "theft of keys".

If the answer is not clearly stated in the context, reply exactly:
"I don't know. It's not in the provided documents."

Question:
{query}

Context:
{context}

Rules:
- Be concise (3–6 sentences).
- Answer ONLY what is explicitly stated in the provided context.
- Cite ONLY sources that directly state the rule you relied on.
- Do NOT cite unrelated sections, examples, or add-on covers unless the question explicitly asks about them.
- If the answer is not clearly stated in the context, reply exactly:
  "I don't know. It's not in the provided documents."
  and DO NOT include citations.
- Otherwise, end your answer with:
  "Citations: [n], [m]"
  using the source numbers from the context.

"""

    answer = llm.invoke(prompt).content
    return answer, docs

if __name__ == "__main__":
    query = "Is theft covered if keys were left inside?"
    answer, docs = ask(query)

    print("\nANSWER:\n")
    print(answer)

    print("\nSOURCES:\n")
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page_label") or d.metadata.get("page")
        print(f"[{i}] {src} | page {page}")
