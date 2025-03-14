import os
import asyncio
import time
import subprocess
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import traceback

load_dotenv()

os.environ['USER_AGENT'] = 'ClassMentor/1.0'

groq_api_key = os.getenv("groq_api_key")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not groq_api_key:
    print("❌ Error: GROQ API key not found in .env file")
    exit(1)
if not gemini_api_key:
    print("❌ Error: Gemini API key not found in .env file")
    exit(1)

FAISS_FOLDER = "../database/faiss"
FAISS_INDEX_PATH = os.path.join(FAISS_FOLDER, "index.faiss")
PROCESSED_FOLDER = "../database/processed"

vector_store = None
MAX_RETRIES = 3
retry_count = 0

def load_faiss_index():
    """Loads the FAISS vector store if it exists, otherwise triggers FAISS building."""
    global vector_store, retry_count

    if retry_count >= MAX_RETRIES:
        print("❌ Failed to load FAISS after multiple attempts. Exiting...")
        exit(1)

    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading FAISS index...")
        vector_store = FAISS.load_local(
            folder_path=FAISS_FOLDER,
            index_name="index",
            embeddings=GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=gemini_api_key
            ),
            allow_dangerous_deserialization=True
        )
    else:
        print("⚠️ FAISS index not found! Building FAISS first...")
        retry_count += 1

        try:
            result = subprocess.run(
                ["python", "build_faiss.py"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("📜 FAISS Build Output:\n", result.stdout)
            if result.stderr:
                print("⚠️ FAISS Build Errors:\n", result.stderr)

            print("🔄 Retrying FAISS loading...")
            load_faiss_index()
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running build_faiss.py: {e}")
            exit(1)

load_faiss_index()

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context and answer directly and do not say based on the provided context.
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
    print("\n🤖 Welcome to the RAG Query System!")
    print("----------------------------------------")
    
    if not os.path.exists(FAISS_FOLDER):
        print(f"❌ Error: FAISS directory not found at {FAISS_FOLDER}")
        exit(1)
        
    try:
        while True:
            user_query = input("📝 Enter your query (or type 'exit' to stop): ")
            if user_query.strip().lower() == 'exit':
                print("\n👋 Goodbye!")
                break
            elif user_query.strip():
                asyncio.run(process_query(user_query))
            else:
                print("⚠️ Query cannot be empty!")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("\nDetailed error:")
        print(traceback.format_exc())