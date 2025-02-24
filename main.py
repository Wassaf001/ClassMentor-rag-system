import os
import asyncio
import time
import subprocess
from dotenv import load_dotenv
from datetime import datetime
from typing import List
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
os.environ['USER_AGENT'] = 'ClassMentor/1.0'

groq_api_key = os.getenv("groq_api_key")
gemini_api_key = os.getenv("GEMINI_API_KEY")

FAISS_INDEX_PATH = "../database/faiss/index.faiss"
PROCESSED_FOLDER = "../database/processed"

vector_store = None

def load_faiss_index():
    """Loads the FAISS vector store if it exists, otherwise triggers FAISS building."""
    global vector_store

    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading FAISS index...")
        vector_store = FAISS.load_local(
            folder_path="../database/faiss",
            index_name="index",
            embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key),
            allow_dangerous_deserialization=True
        )
    else:
        print("⚠️ FAISS index not found! Building FAISS first...")
        subprocess.run(["python", "build_faiss.py"], check=True) 
        print("🔄 Retrying FAISS loading...")
        load_faiss_index() 

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

async def process_query(user_query):
    """Process the user's query and return the response."""
    print(f"Processing query: {user_query}")
    start_time = time.process_time()
    
    relevant_docs = retriever.invoke(user_query)
    response = document_chain.invoke({"context": relevant_docs, "input": user_query})

    elapsed_time = time.process_time() - start_time
    print("\n💡 **Response:**\n")
    print(response)
    print(f"\n⏳ Response time: {elapsed_time:.2f} seconds\n")

    print("\n📚 **Relevant Documents:**\n")
    for i, doc in enumerate(relevant_docs):
        print(f"🔹 Document #{i+1}\n{doc.page_content}\n{'-'*40}")

if __name__ == "__main__":
    user_query = input("📝 Enter your query: ")
    asyncio.run(process_query(user_query))
