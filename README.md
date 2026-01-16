# UK Car Insurance Assistant (RAG Chatbot)

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about UK private car insurance policy wordings. This project uses **LangChain**, **Pinecone**, and **Ollama** to provide accurate, cited answers from PDF policy documents.

Developed for **Giga Academy Cohort IV - Project #4**.

## 🚀 Features

### Core Functionality (Must-Haves)
- **Document Ingestion Pipeline**: Loads, cleans, chunks, embeds, and indexes PDF policy documents.
- **Vector Search**: Uses Pinecone for high-speed similarity search (Top-K).
- **Grounded Answers**: Generates concise answers using **Qwen 2.5**, strictly grounded in retrieved context.
- **Citations**: Every answer includes specific source references (Document Name + Page Number).
- **"I Don't Know" Handling**: Safely admits when an answer is not found in the documents.

### Advanced Features (Nice-to-Haves)
- **🧠 Conversation Memory**: The bot remembers previous turns, allowing follow-up questions (e.g., "What is the excess?" -> "And for fire?").
- **🛡️ Guardrails & Safety**:
  - **Input Defense**: Detects and blocks prompt injection attacks (e.g., "Ignore your rules").
  - **Context Defense**: Filters out malicious instructions hidden within retrieved documents.

## 🛠️ Tech Stack
- **Framework**: Python 3.10+, LangChain
- **UI**: Streamlit
- **Vector DB**: Pinecone
- **LLM**: Ollama (Model: `qwen2.5:7b-instruct`)
- **Embeddings**: Ollama (Model: `mxbai-embed-large`)
- **PDF Processing**: PyPDFLoader

## 📂 Project Structure

```text
├── app/
│   └── main.py           # The Streamlit Chat UI & Application Logic
├── rag/
│   ├── ingestion.py      # Pipeline to load PDFs -> Pinecone
│   ├── retrieval.py      # Logic for searching the Vector Store
│   └── prompting.py      # LLM Chain, Memory, and Guardrails logic
├── documents/            # Place your PDF policy wordings here
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API Keys)
└── README.md             # Project documentation
```
