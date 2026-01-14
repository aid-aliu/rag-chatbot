from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama, OllamaEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    temperature=0,
)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

pc = Pinecone()
index = pc.Index('ragchatbot')

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index=index
)

vector_store.similarity_search("test", k=1)
