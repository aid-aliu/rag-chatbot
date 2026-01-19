import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# --- PATH CONFIG ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

IDK_RESPONSE = "I don't know. It's not in the provided documents."


def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in .env file.")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    return ChatOpenAI(
        model=model_name,
        temperature=0.0,
        api_key=api_key,
    )


def looks_like_prompt_injection(text: str) -> bool:
    import re
    patterns = [
        r"ignore (all|any|previous) instructions",
        r"system prompt",
        r"act as",
        r"jailbreak",
        r"forget your rules",
        r"you are now",
    ]
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)


def contextualize_question(llm, history, latest_question: str) -> str:
    if not history:
        return latest_question

    system_prompt = """Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question
    which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"chat_history": history, "input": latest_question}).strip()


def generate_answer(llm, question: str, context: str) -> str:
    system_template = f"""You are an expert assistant for UK private car insurance policy wordings.

    CORE RULES:
    1. Answer ONLY using the provided context.
    2. If the answer is not in the context, you MUST reply exactly: "{IDK_RESPONSE}"
    3. Do not fabricate information.
    4. Be concise (3–6 sentences).

    DISAMBIGUATION:
    - If the user asks about "theft" without specifics, assume THEFT OF THE VEHICLE unless they mention keys or personal belongings.

    FORMATTING:
    - End your answer with: Citations: [1], [2]
    - Do NOT include citations if the answer is "{IDK_RESPONSE}".
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"context": context, "question": question}).strip()