import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# --- CONSTANTS & CONFIGURATION ---
INDEX_NAME = "ragchatbot"
NAMESPACE = "policies_uk_car_v1"
IDK = "I don't know. It's not in the provided documents."

# Dynamic Retrieval Settings
FETCH_K = 10  # Cast a wide net
SCORE_THRESHOLD = 0.60  # Only keep relevant chunks
MIN_VALID_CHUNKS = 1  # Minimum chunks needed to answer

st.set_page_config(page_title="UK Car Insurance Assistant", layout="wide")


@st.cache_resource
def get_stack():
    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    pc = Pinecone()
    index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(
        embedding=embeddings,
        index=index,
        namespace=NAMESPACE
    )
    return llm, vector_store


def format_source_line(d):
    full_path = d.metadata.get("source", "unknown")
    filename = os.path.basename(full_path)
    page = d.metadata.get("page_label") or d.metadata.get("page")
    return filename, page


def build_context(docs):
    blocks = []
    for i, d in enumerate(docs, start=1):
        filename, page = format_source_line(d)
        blocks.append(f"[{i}] {filename} | page {page}\n{d.page_content}")
    return "\n\n".join(blocks)


def looks_like_prompt_injection(text: str) -> bool:
    patterns = [
        r"ignore (all|any|previous) instructions",
        r"system prompt",
        r"act as",
        r"jailbreak",
    ]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def retrieve_and_filter(vector_store, query: str):
    results = vector_store.similarity_search_with_score(query, k=FETCH_K)

    valid_results = []
    for doc, score in results:
        if score >= SCORE_THRESHOLD and (doc.page_content or "").strip():
            valid_results.append((doc, score))

    return valid_results


def generate_answer(llm, question: str, context: str):
    prompt = f"""
You are an assistant for UK private car insurance. Answer ONLY using the provided context.

Rules:
1. If the answer is not in the context, output exactly: "{IDK}"
2. Do not make up information.
3. Be concise.

Question: {question}

Context:
{context}
"""
    return llm.invoke(prompt).content.strip()


# ---------------- UI LAYOUT ----------------

with st.sidebar:
    st.header("Car Insurance Assistant")
    st.markdown("Ask questions about your policy documents.")

    if st.button("Start New Conversation", type="primary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me anything about UK private car insurance policy wording."}
        ]
        st.rerun()

    st.divider()

    st.subheader("Preferences")
    show_sources = st.toggle("Show Source Citations", value=True)

    st.divider()
    st.caption(f"System: Qwen 2.5 | Threshold: {SCORE_THRESHOLD}")

# Main Chat Interface
st.title("UK Private Car Insurance Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about UK private car insurance policy wording."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_q = st.chat_input("Type your question (e.g., “Is theft covered if keys were left inside?”)")

if user_q and user_q.strip():
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    llm, vector_store = get_stack()

    with st.chat_message("assistant"):

        # 1. PROCESSING PHASE (Inside Spinner)
        # We calculate everything here but DO NOT write to screen yet.
        final_text = ""
        is_idk = False
        valid_results = []

        with st.spinner("Searching policy documents..."):
            valid_results = retrieve_and_filter(vector_store, user_q)
            docs = [d for (d, s) in valid_results]

            context_text = build_context(docs)

            # Safety & Content Checks
            if len(docs) < MIN_VALID_CHUNKS or looks_like_prompt_injection(context_text):
                final_text = IDK
                is_idk = True
            else:
                # Generate Answer
                answer = generate_answer(llm, user_q, context_text)
                if answer.strip() == IDK:
                    final_text = IDK
                    is_idk = True
                else:
                    final_text = answer

        # 2. DISPLAY PHASE (Spinner is gone)
        # Now we write the results to the UI

        if is_idk:
            st.write(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})
        else:
            # Format Output with Citations
            if show_sources:
                citations_ref = ", ".join(f"[{i}]" for i in range(1, len(valid_results) + 1))
                display_text = f"{final_text}\n\n**References:** {citations_ref}"
            else:
                display_text = final_text

            st.write(display_text)
            st.session_state.messages.append({"role": "assistant", "content": display_text})

            # Feedback Buttons
            col1, col2 = st.columns([1, 15])
            with col1:
                st.button("👍", key=f"thumbs_up_{len(st.session_state.messages)}")
            with col2:
                st.button("👎", key=f"thumbs_down_{len(st.session_state.messages)}")

            # Source Expander
            if show_sources:
                with st.expander("View Source Snippets"):
                    for i, (d, s) in enumerate(valid_results, start=1):
                        filename, page = format_source_line(d)
                        st.markdown(f"**[{i}] {filename} (Page {page})**")
                        st.text(d.page_content[:600] + "...")