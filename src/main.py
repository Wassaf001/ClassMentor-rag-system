import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from dotenv import load_dotenv
from utils import faiss_utils
import uvicorn
from pathlib import Path
from typing import Optional

load_dotenv()

os.environ['USER_AGENT'] = 'ClassMentor/1.0'
groq_api_key = os.getenv("groq_api_key1")

if not groq_api_key:
    print("❌ Error: GROQ API key not found in .env file")
    exit(1)

FAISS_FOLDER = "../../database/faiss"
IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "database" / "images"
IMAGES_BASE_URL = "/images"

vector_store = faiss_utils.load_faiss_index()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context in detail. Do not say "based on the provided context" or anything similar.
Think step by step before providing a detailed answer.

<context>
{context}
</context>

Question: {query}
""")

document_chain = create_stuff_documents_chain(llm, prompt_template)

if vector_store is not None:
    retriever = vector_store.as_retriever()
else:
    print("❌ Vector Store not found. Exiting...")
    exit(1)

app = FastAPI()

app.mount(IMAGES_BASE_URL, StaticFiles(directory=str(IMAGES_DIR)), name="images")

class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = Field(default="") 

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process the user's query and return the response with embedded images."""
    user_query = request.query
    user_context = request.context
    print(f"Processing query: {user_query} with context: {user_context}")

    start_time = time.process_time()

    try:
        combined_input = f"{user_query} {user_context}" if user_context else user_query
        relevant_docs = retriever.invoke(combined_input)

        print(f"Retrieved {len(relevant_docs)} documents. Types:")
        docs_for_chain = []
        for doc in relevant_docs:
            if isinstance(doc, Document):
                text = doc.page_content
                image_tags = ""
                metadata = getattr(doc, 'metadata', {})
                image_filenames = metadata.get("image_urls", [])

                for filename in image_filenames:
                    if filename:
                         image_tags += f'\n<br><img src="{IMAGES_BASE_URL}/{filename}" alt="{filename}" style="max-width:100%;"><br>\n'

                modified_content = f"{text}{image_tags}"
                docs_for_chain.append(Document(page_content=modified_content, metadata=doc.metadata))

            else:
                print(f"Warning: Skipping non-Document object in retrieved results: {type(doc)}")
                pass 

        response = document_chain.invoke({"context": docs_for_chain, "query": user_query})


    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


    elapsed_time = time.process_time()
    print(f"\n⏳ Response time: {elapsed_time:.2f} seconds\n")

    return {
        "response": response,
        "response_time": elapsed_time
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
