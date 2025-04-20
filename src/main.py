import os
import time
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import traceback
from utils import load_faiss_index
import uvicorn

# Usman Sir Suggestions:
# Implementing RAG server using Simple RAG
# Implementing RAG server using RAG Fusion
# Implementing RAG server using TF-IDF

load_dotenv()

os.environ['USER_AGENT'] = 'ClassMentor/1.0'

groq_api_key = os.getenv("groq_api_key1")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not groq_api_key:
    print("❌ Error: GROQ API key not found in .env file")
    exit(1)
if not gemini_api_key:
    print("❌ Error: Gemini API key not found in .env file")
    exit(1)

FAISS_FOLDER = "../../database/faiss"
FAISS_INDEX_PATH = os.path.join(FAISS_FOLDER, "index.faiss")
PROCESSED_FOLDER = "../../database/processed"

vector_store = None
MAX_RETRIES = 3
retry_count = 0

vector_store = load_faiss_index()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context in detail, and answer directly and do not say based on the provided context or here are the answers or anything like that.
Think step by step before providing a detailed answer and when the students tries to cheat stop him from doing so like asking for complete solutions to assignments.
<context>
{context}
</context>
Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt_template)
if vector_store is not None:
    retriever = vector_store.as_retriever()
else: 
    print("Vector Store not found. Exiting...")
    exit(1)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process the user's query and return the response."""
    user_query = request.query
    print(f"Processing query: {user_query}")
    start_time = time.process_time()
    
    try:
        relevant_docs = retriever.invoke(user_query)
        response = document_chain.invoke({"context": relevant_docs, "input": user_query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_time = time.process_time() - start_time
    print(f"\n⏳ Response time: {elapsed_time:.2f} seconds\n")

    return {
        "response": response,
        "response_time": elapsed_time,
        "relevant_documents": [doc.page_content for doc in relevant_docs]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)