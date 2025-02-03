import os
import streamlit as st
import requests
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import asyncio
import time
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

@dataclass
class ContextWindow:
    def __init__(self):
        self.max_tokens = 4000  
        self.relevance_threshold = 0.7 
        self.time_weight = 0.1  

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_url = os.getenv("GEMINI_API_URL")
gemini_api_key = os.getenv("GEMINI_API_KEY")
model = "models/text-embedding-004"

st.title("ClassMentor")

class EnhancedContextManager:
    def __init__(self, vector_store):
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        self.retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=vector_store,
            decay_rate=0.01,
            k=3
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
            Document(
                page_content=msg.content,
                metadata={"timestamp": datetime.now().timestamp()}
            ) for msg in chat_history
        ]
        all_docs = current_context + history_docs
        return self._remove_duplicates(all_docs)

    def _filter_by_relevance(self, docs: List[Document], query: str) -> List[Document]:
        scores = self._calculate_relevance_scores(docs, query)
        relevant_docs = [
            doc for doc, score in zip(docs, scores)
            if score >= self.context_window.relevance_threshold
        ]
        return self._trim_to_token_limit(relevant_docs)

    def update_conversation_state(self, query: str, response: str):
        self.memory.save_context({"input": query}, {"answer": response})
        self.conversation_state.update({
            "last_query": query,
            "last_response": response,
            "timestamp": datetime.now().timestamp()
        })

    def _calculate_relevance_scores(self, docs: List[Document], query: str) -> List[float]:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        query_embedding = embeddings.embed_query(query)
        doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]
        scores = [np.dot(query_embedding, doc_emb) for doc_emb in doc_embeddings]
        return scores

    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        seen_content = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs

    def _trim_to_token_limit(self, docs: List[Document]) -> List[Document]:
        total_tokens = 0
        trimmed_docs = []
        for doc in docs:
            token_count = len(doc.page_content.split())
            if total_tokens + token_count <= self.context_window.max_tokens:
                trimmed_docs.append(doc)
                total_tokens += token_count
        return trimmed_docs

def get_embeddings_from_gemini(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings using Google's text-embedding-004 model.
    Arguments:
        texts: List of strings to embed.
    Returns:
        List of embedding vectors.
    """
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=gemini_api_key
    )
    embeddings = [embeddings_model.embed_query(text) for text in texts]
    return embeddings

async def initialize_session_state():
    """Initialize loader, vector store, and other necessary items."""
    if "vector" not in st.session_state:
        st.session_state.loader = WebBaseLoader("https://paulgraham.com/greatwork.html")
        st.session_state.docs = await asyncio.to_thread(st.session_state.loader.load)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.documents = text_splitter.split_documents(st.session_state.docs[:5])
        document_texts = [doc.page_content for doc in st.session_state.documents]
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_api_key
        )
        embeddings = embeddings_model.embed_documents(document_texts)
        st.session_state.vector = FAISS.from_embeddings(embeddings, st.session_state.documents)

asyncio.run(initialize_session_state())

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)

prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vector.as_retriever()

async def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Combine multiple retrieval results using Reciprocal Rank Fusion asynchronously.
    Arguments:
        results: List of lists containing retrieved documents.
        k: A constant for rank score adjustment.
    Returns:
        List of reranked documents based on fused scores.
    """
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = str(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_results]

async def rag_fusion_retrieval_chain(query):
    """
    Retrieve documents using RAG Fusion with query variations asynchronously.
    Arguments:
        query: The user query.
    Returns:
        Fused and reranked document contexts.
    """
    query_variations = [f"{query} variation {i}" for i in range(1, 5)]
    tasks = [asyncio.to_thread(retriever.get_relevant_documents, q) for q in query_variations]
    results = await asyncio.gather(*tasks)
    fused_contexts = await reciprocal_rank_fusion(results)
    return fused_contexts[:5]

user_query = st.text_input("Input your question here:")

if user_query:
    async def process_query():
        """Process the user's query and return the response."""
        start_time = time.process_time()
        fused_contexts = await rag_fusion_retrieval_chain(user_query)
        response = document_chain.invoke({"context": fused_contexts, "input": user_query})
        elapsed_time = time.process_time() - start_time

        st.write(response["answer"])
        st.write(f"Response time: {elapsed_time:.2f} seconds")

        with st.expander("Relevant Documents from Similarity Search"):
            for i, doc in enumerate(fused_contexts):
                st.write(f"Source Document # {i+1}")
                st.write(doc.page_content)
                st.write("--------------------------------")

    asyncio.run(process_query())
