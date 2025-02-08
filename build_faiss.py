import os
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
os.environ['USER_AGENT'] = 'ClassMentor/1.0'

gemini_api_key = os.getenv("GEMINI_API_KEY")

FAISS_FOLDER = "../database/faiss"
PROCESSED_FOLDER = "../database/processed"

def build_faiss():
    """Builds the FAISS index from processed documents."""
    print("📂 Processing documents to build FAISS...")
    
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

    vector_store.save_local(FAISS_FOLDER)
    print("✅ FAISS index built successfully!")

if __name__ == "__main__":
    build_faiss()
