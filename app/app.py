import os
import sys
import time
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. SECRETS SETUP (CRITICAL FOR CLOUD DEPLOYMENT) ---
# We use a try-except block because st.secrets crashes if no secrets.toml exists (like on your local PC).
# This allows the app to work locally (using your .env file) AND on the cloud (using st.secrets).
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    # This happens when running locally without a secrets.toml file.
    # We pass silently because the .env file will be loaded later by the rag module.
    pass
except Exception:
    # Catch any other specific Streamlit secrets errors to prevent crashing.
    pass

# --- PATH CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- IMPORTS ---
try:
    from app.dashboard import render_dashboard
except ImportError:
    from dashboard import render_dashboard

from rag.retrieval import get_vector_store, search_docs, build_context, format_source_line
from rag.prompting import (
    get_llm,
    generate_answer,
    contextualize_question,
    looks_like_prompt_injection,
    IDK_RESPONSE,
)

# --- PAGE SETUP ---
st.set_page_config(page_title="UK Car Insurance Assistant", layout="wide")


@st.cache_resource
def load_resources():
    llm_instance = get_llm()
    vector_store_instance = get_vector_store()
    return llm_instance, vector_store_instance


# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about UK private car insurance policy wording."}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "queries": [],
        "latencies": [],
        "tokens_generated": [],
        "feedback_score": 0,
    }

# Track which messages have already received feedback
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Car Insurance Assistant")

    if st.button("Start New Conversation", type="primary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me anything about UK private car insurance policy wording."}
        ]
        st.session_state.chat_history = []
        st.session_state.feedback_given = set()  # Reset feedback tracking
        st.rerun()

    st.divider()
    st.subheader("Preferences")
    show_sources = st.toggle("Show Source Citations", value=True)
    st.divider()
    st.caption("Powered by OpenAI + FAISS")

# --- MAIN TABS ---
tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Observability Dashboard"])

with tab_chat:
    st.title("UK Private Car Insurance Assistant")

    # 1. Render History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if "sources" in msg and show_sources and msg["sources"]:
                with st.expander(f"📚 References ({len(msg['sources'])})"):
                    for idx, doc in enumerate(msg["sources"], start=1):
                        filename = doc.get("source", "Unknown")
                        page = doc.get("page", "?")
                        content = doc.get("content", "")
                        st.markdown(f"**[{idx}] {filename}** (Page {page})")
                        st.caption(content)
                        st.divider()

    # 2. Handle Feedback Buttons (Only if there is history)
    if len(st.session_state.messages) > 1:
        # Use the length of messages as a unique ID for the current interaction
        msg_id = len(st.session_state.messages)

        # Check if feedback has already been given for this specific interaction
        if msg_id not in st.session_state.feedback_given:
            col1, col2, _ = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_{msg_id}"):
                    st.session_state.metrics["feedback_score"] += 1
                    st.session_state.feedback_given.add(msg_id)
                    st.toast("Feedback recorded!", icon="👍")
                    st.rerun()
            with col2:
                if st.button("👎", key=f"down_{msg_id}"):
                    st.session_state.metrics["feedback_score"] -= 1
                    st.session_state.feedback_given.add(msg_id)
                    st.toast("Thanks for the feedback.", icon="👎")
                    st.rerun()
        else:
            # Optional: Show a disabled state or a "Thank you" message
            st.caption("✅ Feedback submitted for this response.")

    # 3. Handle User Input
    user_q = st.chat_input("Type your question (e.g. 'Is theft covered?')")

    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})

        with st.chat_message("user"):
            st.write(user_q)

        with st.chat_message("assistant"):
            final_text = ""
            valid_docs = []
            source_data = []

            start_time = time.time()

            with st.spinner("Analyzing policy documents..."):
                try:
                    llm, vector_store = load_resources()

                    if looks_like_prompt_injection(user_q):
                        final_text = "I cannot fulfill that request. Please stick to questions about insurance policies."
                    else:
                        if st.session_state.chat_history:
                            search_query = contextualize_question(llm, st.session_state.chat_history, user_q)
                        else:
                            search_query = user_q

                        valid_docs = search_docs(vector_store, search_query)
                        context_text = build_context(valid_docs)

                        if len(valid_docs) < 1 or looks_like_prompt_injection(context_text):
                            final_text = IDK_RESPONSE
                            valid_docs = []
                        else:
                            final_text = generate_answer(llm, search_query, context_text)

                except Exception as e:
                    final_text = f"An error occurred: {str(e)}"
                    valid_docs = []
                    st.error(final_text)

            latency = time.time() - start_time
            tokens = int(len(final_text) / 4)

            st.session_state.metrics["queries"].append(user_q)
            st.session_state.metrics["latencies"].append(latency)
            st.session_state.metrics["tokens_generated"].append(tokens)

            st.write(final_text)

            if valid_docs and final_text != IDK_RESPONSE:
                for doc in valid_docs:
                    filename, page = format_source_line(doc)
                    source_data.append(
                        {"source": filename, "page": page, "content": doc.page_content}
                    )

                if show_sources:
                    with st.expander(f"📚 References ({len(source_data)})"):
                        for i, data in enumerate(source_data, start=1):
                            st.markdown(f"**[{i}] {data['source']}** (Page {data['page']})")
                            st.caption(data["content"])
                            st.divider()

            st.session_state.messages.append(
                {"role": "assistant", "content": final_text, "sources": source_data}
            )

            st.session_state.chat_history.extend(
                [HumanMessage(content=user_q), AIMessage(content=final_text)]
            )
            if len(st.session_state.chat_history) > 6:
                st.session_state.chat_history = st.session_state.chat_history[-6:]

            st.rerun()

with tab_dashboard:
    render_dashboard()