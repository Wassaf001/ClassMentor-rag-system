import os
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
PROCESSED_FOLDER = "../../database/processed"


def actual_build_faiss():
    """Builds the FAISS index from processed documents."""
    print("🚀 Starting FAISS index build process...")

    if not gemini_api_key:
        print("❌ GEMINI_API_KEY not found in environment. Build process cannot continue.")
        exit(1)

    if not os.path.isdir(PROCESSED_FOLDER):
        print(f"❌ Error: Processed documents folder not found at '{PROCESSED_FOLDER}'.")
        exit(1)

    all_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".txt")]
    if not all_files:
        print(f"⚠️ No processed .txt documents found in '{PROCESSED_FOLDER}'. FAISS index will not be built.")
        print("Exiting build process.")
        exit(1) 

    print(f"Found {len(all_files)} documents to process in '{PROCESSED_FOLDER}'.")

    os.makedirs(FAISS_FOLDER, exist_ok=True)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=gemini_api_key
        )
    except Exception as e:
        print(f"❌ Error initializing embeddings model: {e}")
        print(traceback.format_exc())
        exit(1)

    processed_docs_content = []
    doc_metadatas = []
    for filename in all_files:
        file_path = os.path.join(PROCESSED_FOLDER, filename)
        print(f"📖 Reading file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                processed_docs_content.append(content)
                image_folder = os.path.join(PROCESSED_FOLDER, "images")
                base_name = os.path.splitext(filename)[0]
                matching_images = [
                    img_name for img_name in os.listdir(image_folder)
                    if img_name.startswith(base_name)
                ] if os.path.exists(image_folder) else []

                doc_metadatas.append({
                    "source": filename,
                    "image_urls": matching_images  # 👈 Add this
                })
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {e}. Skipping this file.")

    if not processed_docs_content:
        print("⚠️ No documents successfully read. FAISS index not built.")
        exit(1) 
    print("Splitting documents into chunks...")
    documents_to_split = [Document(page_content=text, metadata=meta) for text, meta in zip(processed_docs_content, doc_metadatas)]
    split_docs = text_splitter.split_documents(documents_to_split)
    
    print(f"📄 Documents split into {len(split_docs)} chunks.")
    print("🔨 Generating FAISS embeddings and building index (this may take a while)...")

    try:
        vector_store = FAISS.from_documents(
            documents=split_docs,
            embedding=embeddings_model
        )
        
        vector_store.save_local(FAISS_FOLDER, index_name="index")
        print(f"✅ FAISS index built successfully and saved to '{FAISS_FOLDER}'.")
    except Exception as e:
        print(f"❌ Error during FAISS creation or saving: {e}")
        print(traceback.format_exc())
        exit(1) 

if __name__ == "__main__":
    actual_build_faiss()