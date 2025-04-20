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
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import traceback
from utils import load_faiss_index
import numpy as np
import uvicorn

# Load environment variables
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

# Paths
FAISS_FOLDER = "../../database/faiss"
FAISS_INDEX_PATH = os.path.join(FAISS_FOLDER, "index.faiss")
PROCESSED_FOLDER = "../../database/processed"

# Load vector store
vector_store = load_faiss_index()
if vector_store is None:
    print("Vector Store not found. Exiting...")
    exit(1)

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Create prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context in detail, and answer directly and do not say based on the provided context or here are the answers or anything like that.
Think step by step before providing a detailed answer and when the students tries to cheat stop him from doing so like asking for complete solutions to assignments.
<context>
{context}
</context>
Question: {input}
""")

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt_template)

# Create reciprocal rank fusion function
def reciprocal_rank_fusion(results_list, k=60):
    """
    Combines multiple search results using Reciprocal Rank Fusion
    
    Args:
        results_list: List of lists of retrieved documents
        k: Constant to prevent division by very small numbers
        
    Returns:
        Combined and re-ranked list of documents
    """
    # Create a dictionary to store document scores
    fused_scores = {}
    
    # Process each list of results
    for docs in results_list:
        # Process each document in the current result list
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata.get('source', '') + str(hash(doc.page_content))
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0}
            
            # Apply reciprocal rank fusion formula
            fused_scores[doc_id]["score"] += 1.0 / (rank + k)
    
    # Sort documents by their fused scores (descending)
    reranked_results = [
        item["doc"] 
        for item in sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    ]
    
    return reranked_results[:10]  # Return top 10 documents

# RAG Fusion implementation
class RAGFusion:
    def __init__(self, vector_store):
        # Create initial retrievers
        self.dense_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        
        # Get all documents for BM25
        all_docs = vector_store.similarity_search("", k=1000)  # Get as many docs as possible
        
        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        self.bm25_retriever.k = 20
        
        # Create query generation chain
        self.query_generator = ChatPromptTemplate.from_template("""
        Generate 3 different search queries related to the following question. 
        These should help retrieve relevant context to answer the question.
        Return only the queries, one per line, without any additional text or numbering.
        
        Original question: {question}
        """) | llm | StrOutputParser()
    
    async def retrieve(self, query):
        try:
            alternative_queries_text = await asyncio.to_thread(self.query_generator.invoke, {"question": query})
            alternative_queries = [q.strip() for q in alternative_queries_text.split('\n') if q.strip()]
            
            all_queries = [query] + alternative_queries
            print(f"Generated queries: {all_queries}")
            
            dense_results = []
            bm25_results = []
            
            for q in all_queries:
                dense_docs = await asyncio.to_thread(self.dense_retriever.invoke, q)
                bm25_docs = await asyncio.to_thread(self.bm25_retriever.get_relevant_documents, q)
                
                dense_results.append(dense_docs)
                bm25_results.append(bm25_docs)
            
            all_results = dense_results + bm25_results
            
            fused_docs = reciprocal_rank_fusion(all_results)
            return fused_docs
            
        except Exception as e:
            print(f"Error in query generation: {e}")
            # Fallback to standard retrieval
            dense_docs = await asyncio.to_thread(self.dense_retriever.invoke, query)
            return dense_docs

rag_fusion = RAGFusion(vector_store)

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
        relevant_docs = await rag_fusion.retrieve(user_query)
        
        response = await asyncio.to_thread(
            document_chain.invoke, 
            {"context": relevant_docs, "input": user_query}
        )
        
    except Exception as e:
        print(f"Error processing query: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    elapsed_time = time.process_time() - start_time
    print(f"\n⏳ Response time: {elapsed_time:.2f} seconds\n")
    
    return {
        "response": response,
        "response_time": elapsed_time,
        "relevant_documents": [doc.page_content for doc in relevant_docs],
        "method": "RAG Fusion"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  