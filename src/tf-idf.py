import os
import time
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from utils import load_faiss_index
import uvicorn

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
PROCESSED_FOLDER = "../../database/processed"

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

class TFIDFRetriever:
    def __init__(self, k=10):
        """
        Initialize the TF-IDF retriever.
        
        Args:
            k: Number of documents to retrieve
        """
        self.k = k
        self.documents = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2
        )
        self.document_vectors = None
        self._load_documents()
        
    def _load_documents(self):
        """Load documents from the FAISS index for TF-IDF processing"""
        try:
            vector_store = load_faiss_index()
            if vector_store is None:
                print("Vector Store not found. Cannot initialize TF-IDF retriever.")
                return
            all_docs = vector_store.similarity_search("", k=1000)
            self.documents = all_docs
            texts = [doc.page_content for doc in self.documents]
            self.document_vectors = self.vectorizer.fit_transform(texts)
            print(f"TF-IDF retriever initialized with {len(self.documents)} documents")
            
        except Exception as e:
            print(f"Error loading documents for TF-IDF: {e}")
            self.documents = []
    
    def get_relevant_documents(self, query):
        """
        Retrieve documents relevant to the query using TF-IDF similarity.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
        """
        if not self.documents or self.document_vectors is None:
            print("No documents available for TF-IDF retrieval")
            return []
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = np.argsort(similarities)[-self.k:][::-1]
        top_docs = [self.documents[i] for i in top_indices]
        
        return top_docs

tfidf_retriever = TFIDFRetriever(k=10)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process the user's query and return the response."""
    query = request.query
    print(f"Processing query: {query}")
    start_time = time.process_time()
    
    try:
        relevant_docs = await asyncio.to_thread(
            tfidf_retriever.get_relevant_documents,
            query
        )
        
        if not relevant_docs:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        response = await asyncio.to_thread(
            document_chain.invoke, 
            {"context": relevant_docs, "input": query}
        )
        
    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    elapsed_time = time.process_time() - start_time
    print(f"\n⏳ Response time: {elapsed_time:.2f} seconds\n")
    
    return {
        "response": response,
        "response_time": elapsed_time,
        "relevant_documents": [doc.page_content for doc in relevant_docs],
        "method": "TF-IDF RAG"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)  