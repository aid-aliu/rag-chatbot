# 🚗 UK Private Car Insurance Assistant

**Live Demo:** [Click here to try the app](https://uk-insurance-bot.streamlit.app/)

An AI-powered RAG (Retrieval-Augmented Generation) application designed to answer questions about UK car insurance policies. It uses **OpenAI** for reasoning and **FAISS** for fast document retrieval, wrapped in a user-friendly **Streamlit** interface.

*(Add a screenshot of your chat interface here)*

## ✨ Features

* **💬 RAG Chatbot:** Ask natural language questions about insurance documents (e.g., *"Is theft covered if I leave my keys in the car?"*).
* **📚 Source Citations:** Every answer includes references to the specific PDF pages used.
* **📊 Observability Dashboard:** Track latency, token usage, and user feedback in real-time.
* **🛡️ Security:** Built-in prompt injection detection.
* **☁️ Hybrid Config:** Works locally (using `.env`) and on Streamlit Cloud (using Secrets).

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** OpenAI (GPT-4o-mini)
* **Vector Store:** FAISS (CPU)
* **Framework:** LangChain
* **Embeddings:** OpenAI `text-embedding-3-small`

---

## 🚀 Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/aid-aliu/rag-chatbot.git
cd rag-chatbot

```

### 2. Install dependencies

```bash
pip install -r requirements.txt

```

### 3. Setup Environment Variables

Create a `.env` file in the root directory and add your API key:

```ini
OPENAI_API_KEY="sk-proj-..."

```

### 4. Build the Knowledge Base

Place your PDF policy documents in a folder named `documents` at the root, then run:

```bash
python rag/ingest.py

```

*This parses the PDFs and creates the local `faiss_index` database.*

### 5. Run the App

```bash
streamlit run app/app.py

```

---

## 🌐 Deployment

This app is currently live on **Streamlit Community Cloud**.

To deploy your own version:

1. Push your code (including the `faiss_index` folder) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Deploy a new app pointing to `app/app.py`.
4. **Crucial Step:** In the App Settings, go to **Secrets** and add:
```toml
OPENAI_API_KEY = "sk-proj-..."

```

---

## 📂 Project Structure

```text
.
├── app/
│   ├── app.py             # Main Streamlit application
│   └── dashboard.py       # Analytics dashboard logic
├── rag/
│   ├── ingest.py          # Script to process PDFs -> FAISS index
│   ├── retrieval.py       # Logic for searching the vector DB
│   └── prompting.py       # LLM generation and prompt templates
├── documents/             # (Optional) Folder for raw PDFs
├── faiss_index/           # The generated vector database
├── requirements.txt       # Python dependencies
└── .gitignore             # Security rules

```

## 📈 Dashboard

The app includes a built-in "Observability Dashboard" tab where you can monitor:

* **Total Queries:** Number of user interactions.
* **Latency:** How long the AI takes to respond.
* **Token Usage:** Estimated cost tracking.
* **User Feedback:** Thumbs up/down scores.

---

## ⚠️ Important Notes

* **PDF Data:** The `documents` folder is ignored by Git to keep the repo light. The app relies on the `faiss_index` folder, which **must** be committed to GitHub for deployment.
* **Security:** Never commit your `.env` file. The `.gitignore` file included in this repo prevents accidental uploads.
