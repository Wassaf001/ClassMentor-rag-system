import os
import requests
import asyncio
import time
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from dataclasses import dataclass
from typing import List
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
os.environ['USER_AGENT'] = 'ClassMentor/1.0'

groq_api_key = os.getenv("groq_api_key")
gemini_api_key = os.getenv("GEMINI_API_KEY")

FAISS_INDEX_PATH = "../database/faiss/index.faiss"
PROCESSED_FOLDER = "../database/processed"

vector_store = None

@dataclass
class ContextWindow:
    max_tokens: int = 4000
    relevance_threshold: float = 0.7
    time_weight: float = 0.1

class EnhancedContextManager:
    def __init__(self, vector_store):
        self.memory = ConversationBufferWindowMemory(
            k=5, memory_key="chat_history", return_messages=True, output_key="answer"
        )
        self.retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=vector_store, decay_rate=0.01, k=3
        )
        self.context_window = ContextWindow()
        self.conversation_state = {}

    def get_relevant_context(self, query: str) -> List[Document]:
        current_context = self.retriever.get_relevant_documents(query)
        chat_history = self.memory.chat_memory.messages
        merged_context = self._merge_context(current_context, chat_history)
        return self._filter_by_relevance(merged_context, query)

    def _merge_context(self, current_context, chat_history):
        history_docs = [
            Document(page_content=msg.content, metadata={"timestamp": datetime.now().timestamp()})
            for msg in chat_history
        ]
        return self._remove_duplicates(current_context + history_docs)

    def _filter_by_relevance(self, docs: List[Document], query: str) -> List[Document]:
        scores = self._calculate_relevance_scores(docs, query)
        return self._trim_to_token_limit(
            [doc for doc, score in zip(docs, scores) if score >= self.context_window.relevance_threshold]
        )

    def _calculate_relevance_scores(self, docs: List[Document], query: str) -> List[float]:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
        query_embedding = embeddings.embed_query(query)
        doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]
        return [np.dot(query_embedding, doc_emb) for doc_emb in doc_embeddings]

    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        seen_content = set()
        return [doc for doc in docs if not (doc.page_content in seen_content or seen_content.add(doc.page_content))]

    def _trim_to_token_limit(self, docs: List[Document]) -> List[Document]:
        total_tokens, trimmed_docs = 0, []
        for doc in docs:
            token_count = len(doc.page_content.split())
            if total_tokens + token_count <= self.context_window.max_tokens:
                trimmed_docs.append(doc)
                total_tokens += token_count
        return trimmed_docs

def load_faiss_index():
    """Loads or creates the FAISS vector store."""
    global vector_store

    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading FAISS index...")
        vector_store = FAISS.load_local(
            folder_path="../database/faiss",
            index_name="index",
            embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key)
        )
    else:
        print("⚠️ FAISS index not found! Rebuilding...")
        processed_docs = []
        for filename in os.listdir(PROCESSED_FOLDER):
            if filename.endswith(".txt"):
                with open(os.path.join(PROCESSED_FOLDER, filename), 'r', encoding='utf-8') as f:
                    processed_docs.append(Document(page_content=f.read()))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(processed_docs)

        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key)
        vector_store = FAISS.from_texts(
            texts=[doc.page_content for doc in split_docs],
            embedding=embeddings_model,
            metadatas=[doc.metadata for doc in split_docs]
        )
        vector_store.save_local("../database/faiss")

load_faiss_index()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
<context>
{context}
</context>
Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = vector_store.as_retriever()

async def reciprocal_rank_fusion(results: list[list], k=60):
    """Combine retrieval results using Reciprocal Rank Fusion."""
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = str(doc)
            fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)
    return [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

async def rag_fusion_retrieval_chain(query):
    """Retrieve documents using RAG Fusion with reduced query variations."""
    query_variations = [query, f"{query} variation 1", f"{query} variation 2"]
    results = await asyncio.gather(*(asyncio.to_thread(retriever.get_relevant_documents, q) for q in query_variations))
    return (await reciprocal_rank_fusion(results))[:5]

async def process_query(user_query):
    """Process the user's query and return the response."""
    start_time = time.process_time()
    
    fused_contexts = await rag_fusion_retrieval_chain(user_query)
    response = document_chain.invoke({"context": fused_contexts, "input": user_query})

    elapsed_time = time.process_time() - start_time
    print("\n💡 **Response:**\n")
    print(response["answer"])
    print(f"\n⏳ Response time: {elapsed_time:.2f} seconds\n")

    print("\n📚 **Relevant Documents:**\n")
    for i, doc in enumerate(fused_contexts):
        print(f"🔹 Document #{i+1}\n{doc.page_content}\n{'-'*40}")

if __name__ == "__main__":
    user_query = input("📝 Enter your query: ")
    asyncio.run(process_query(user_query))
