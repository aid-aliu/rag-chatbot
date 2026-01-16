import streamlit as st
import sys
import os
import time

# Ensure 'rag' module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.retrieval import get_vector_store, search_docs, build_context, format_source_line
from rag.prompting import get_llm, generate_answer, contextualize_question, looks_like_prompt_injection, IDK_RESPONSE
from langchain_core.messages import HumanMessage, AIMessage

# Import the dashboard component
from dashboard import render_dashboard

st.set_page_config(page_title="UK Car Insurance Assistant", layout="wide")


@st.cache_resource
def load_resources():
    """Cache LLM and Vector Store to prevent reloading on every interaction."""
    llm_instance = get_llm()
    vector_store_instance = get_vector_store()
    return llm_instance, vector_store_instance


# --- STATE MANAGEMENT ---
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
        "feedback_score": 0
    }

# --- SIDEBAR ---
with st.sidebar:
    st.header("Car Insurance Assistant")

    if st.button("Start New Conversation", type="primary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me anything about UK private car insurance policy wording."}
        ]
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.subheader("Preferences")
    show_sources = st.toggle("Show Source Citations", value=True)
    st.divider()
    st.caption("Features: Memory | Guardrails | Observability")

# --- TABS LAYOUT ---
tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Observability Dashboard"])

# --- TAB 1: MAIN CHAT INTERFACE ---
with tab_chat:
    st.title("UK Private Car Insurance Assistant")

    # 1. RENDER HISTORY
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            # Render citations (if available and NOT empty)
            if "sources" in msg and show_sources and msg["sources"]:
                with st.expander(f"📚 References ({len(msg['sources'])})"):
                    for idx, doc in enumerate(msg["sources"], start=1):
                        filename = doc.get("source", "Unknown")
                        page = doc.get("page", "?")
                        content = doc.get("content", "")
                        st.markdown(f"**[{idx}] {filename}** (Page {page})")
                        st.caption(content)
                        st.divider()

    # 2. STATE MACHINE LOGIC
    # We check the LAST message to decide if we are "Thinking" or "Ready"
    last_msg = st.session_state.messages[-1]

    if last_msg["role"] == "user":
        # --- PROCESSING STATE (Input Hidden) ---
        # The last message was from the user, so we MUST generate an answer now.

        with st.chat_message("assistant"):
            final_text = ""
            valid_docs = []
            source_data = []

            with st.spinner("Thinking..."):
                start_time = time.time()
                user_q = last_msg["content"]

                llm, vector_store = load_resources()

                # Guardrails (Input)
                if looks_like_prompt_injection(user_q):
                    final_text = "I cannot fulfill that request. Please stick to questions about insurance policies."
                else:
                    # Memory
                    if st.session_state.chat_history:
                        search_query = contextualize_question(llm, st.session_state.chat_history, user_q)
                    else:
                        search_query = user_q

                    # Retrieval
                    valid_docs = search_docs(vector_store, search_query)
                    context_text = build_context(valid_docs)

                    # Guardrails (Context)
                    MIN_VALID_CHUNKS = 1
                    if len(valid_docs) < MIN_VALID_CHUNKS or looks_like_prompt_injection(context_text):
                        final_text = IDK_RESPONSE
                    else:
                        # Generation
                        final_text = generate_answer(llm, search_query, context_text)

            # --- CRITICAL FIX: STRIP CITATIONS IF "I DON'T KNOW" ---
            # If the LLM output contains the IDK phrase, we force-clear the citations.
            # We use 'in' to be safe against slight variations.
            if IDK_RESPONSE in final_text:
                final_text = IDK_RESPONSE
                valid_docs = []  # Clear docs so they don't show up in sources
            # -------------------------------------------------------

            # Metrics Calculation
            end_time = time.time()
            latency = end_time - start_time
            tokens = len(final_text) / 4

            st.session_state.metrics["queries"].append(user_q)
            st.session_state.metrics["latencies"].append(latency)
            st.session_state.metrics["tokens_generated"].append(tokens)

            # Output Answer
            st.write(final_text)

            # Process Sources (Only if we found valid docs AND it's not an IDK response)
            if valid_docs and final_text != IDK_RESPONSE:
                for doc in valid_docs:
                    filename, page = format_source_line(doc)
                    source_data.append({
                        "source": filename,
                        "page": page,
                        "content": doc.page_content
                    })

                # Show sources immediately
                if show_sources:
                    with st.expander(f"📚 References ({len(source_data)})"):
                        for i, data in enumerate(source_data, start=1):
                            st.markdown(f"**[{i}] {data['source']}** (Page {data['page']})")
                            st.caption(data['content'])
                            st.divider()

            # Save Assistant Message to State
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_text,
                "sources": source_data  # This will be empty if IDK
            })

            # Update Memory Chain
            st.session_state.chat_history.extend([
                HumanMessage(content=user_q),
                AIMessage(content=final_text)
            ])
            if len(st.session_state.chat_history) > 6:
                st.session_state.chat_history = st.session_state.chat_history[-6:]

            # FORCE RERUN to switch state back to "Ready"
            st.rerun()

    else:
        # --- READY STATE (Input Visible) ---
        # The bot is done. Show feedback buttons and the input box.

        # 1. Show Feedback Buttons for the answer above (Optional)
        # We only show this if there is at least 1 Q&A pair
        if len(st.session_state.messages) > 1:
            col1, col2, _ = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_{len(st.session_state.messages)}"):
                    st.session_state.metrics["feedback_score"] += 1
                    st.toast("Feedback recorded!", icon="👍")
            with col2:
                if st.button("👎", key=f"down_{len(st.session_state.messages)}"):
                    st.session_state.metrics["feedback_score"] -= 1
                    st.toast("Thanks for the feedback.", icon="👎")

        # 2. Show Chat Input
        user_q = st.chat_input("Type your question (e.g. 'Is theft covered?')")

        if user_q and user_q.strip():
            # Append to state and RERUN immediately.
            # This triggers the "Processing State" block above.
            st.session_state.messages.append({"role": "user", "content": user_q})
            st.rerun()

# --- TAB 2: OBSERVABILITY DASHBOARD ---
with tab_dashboard:
    render_dashboard()