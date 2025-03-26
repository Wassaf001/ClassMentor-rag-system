import os
import subprocess
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import traceback

load_dotenv()

os.environ['USER_AGENT'] = 'ClassMentor/1.0'
gemini_api_key = os.getenv("GEMINI_API_KEY")

FAISS_FOLDER = "../../database/faiss"
FAISS_INDEX_PATH = os.path.join(FAISS_FOLDER, "index.faiss")
PROCESSED_FOLDER = "../../database/processed"

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
        print("vector store", vector_store)
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
    return vector_store

def build_faiss():
    """Builds the FAISS index from processed documents."""
    all_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".txt")]
    if not all_files:
        print("⚠️ No processed documents found! Exiting FAISS build.")
        exit(1)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=gemini_api_key
    )

    processed_docs = []
    for filename in all_files:
        file_path = os.path.join(PROCESSED_FOLDER, filename)
        print(f"📖 Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            processed_docs.append(Document(page_content=f.read()))

    if processed_docs:
        split_docs = text_splitter.split_documents(processed_docs)
        print("🔨 Generating FAISS embeddings")
        try:
            vector_store = FAISS.from_texts(
                texts=[doc.page_content for doc in split_docs],
                embedding=embeddings_model,
                metadatas=[doc.metadata for doc in split_docs]
            )
            vector_store.save_local(FAISS_FOLDER, index_name="index")
            print("✅ FAISS index built successfully")
        except Exception as e:
            print(f"❌ Error during FAISS creation: {e}")
            print(traceback.format_exc())
            exit(1)