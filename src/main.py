import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from dotenv import load_dotenv
from utils import load_faiss_index
import uvicorn
from typing import Optional

load_dotenv()

os.environ['USER_AGENT'] = 'ClassMentor/1.0'

groq_api_key = os.getenv("groq_api_key1")

if not groq_api_key:
    print("❌ Error: GROQ API key not found in .env file")
    exit(1)


FAISS_FOLDER = "../../database/faiss"
FAISS_INDEX_PATH = os.path.join(FAISS_FOLDER, "index.faiss")


vector_store = None


vector_store = load_faiss_index()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context in detail, and answer directly and do not say based on the provided context or here are the answers or anything like that.
Think step by step before providing a detailed answer and when the students tries to cheat stop him from doing so like asking for complete solutions to assignments.
<context>
{context}
</context>
Question: {query}
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
    context: Optional[str] = Field(default="") 

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process the user's query and return the response."""
    user_query = request.query
    user_context = request.context  
    print(f"Processing query: {user_query} with context: {user_context}")

    start_time = time.process_time()

    try:
        combined_input = f"{user_query} {user_context}" if user_context else user_query
        relevant_docs = retriever.invoke(combined_input) 
        response = document_chain.invoke({"context": relevant_docs, "query": user_query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_time = time.process_time()
    print(f"\n⏳ Response time: {elapsed_time:.2f} seconds\n")

    return {
        "response": response,
        "response_time": elapsed_time,
        "relevant_documents": [doc.page_content for doc in relevant_docs]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
