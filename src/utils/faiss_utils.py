import os
import subprocess
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

os.environ['USER_AGENT'] = 'ClassMentor/1.0' 
gemini_api_key = os.getenv("GEMINI_API_KEY")

FAISS_FOLDER = "../../database/faiss" 
FAISS_INDEX_PATH = os.path.join(FAISS_FOLDER, "index.faiss")
BUILD_SCRIPT_NAME = "build_faiss.py"

MAX_BUILD_ATTEMPTS = 3
_vector_store_cache = None

def load_faiss_index(attempt=1):
    """
    Loads the FAISS vector store. If it doesn't exist or is corrupted,
    it triggers the build process and retries loading.
    """
    global _vector_store_cache

    if _vector_store_cache:
        print("🔄 Returning cached FAISS index...")
        return _vector_store_cache

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"🔄 Attempting to load FAISS index (Attempt {attempt})...")
        try:
            if not gemini_api_key:
                print("❌ GEMINI_API_KEY not found. Cannot load FAISS with GoogleGenerativeAIEmbeddings.")
                return None 

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=gemini_api_key
            )
            vector_store = FAISS.load_local(
                folder_path=FAISS_FOLDER,
                index_name="index",
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS index loaded successfully.")
            _vector_store_cache = vector_store
            return vector_store
        except Exception as e:
            print(f"❌ Error loading existing FAISS index: {e}")
            if attempt < MAX_BUILD_ATTEMPTS:
                print("Index might be corrupted. Attempting to rebuild...")
            else:
                print(f"❌ Failed to load FAISS index after {attempt} attempts and potential rebuilds. Exiting...")
                exit(1) 
    if attempt <= MAX_BUILD_ATTEMPTS:
        print(f"⚠️ FAISS index not found or loading failed. Triggering build (Attempt {attempt}/{MAX_BUILD_ATTEMPTS})...")
        try:
            result = subprocess.run(
                ["python", BUILD_SCRIPT_NAME],
                check=True,       
                capture_output=True, 
                text=True        
            )
            print("📜 FAISS Build Script Output:\n", result.stdout)
            if result.stderr:
                print("⚠️ FAISS Build Script Errors:\n", result.stderr)
            
            return load_faiss_index(attempt=attempt + 1)

        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {BUILD_SCRIPT_NAME}: {e}")
            print(f"📜 stdout from {BUILD_SCRIPT_NAME}:\n", e.stdout)
            print(f"stderr from {BUILD_SCRIPT_NAME}:\n", e.stderr)
            if attempt < MAX_BUILD_ATTEMPTS:
                print(f"Retrying build and load ({attempt + 1}/{MAX_BUILD_ATTEMPTS})...")
                return load_faiss_index(attempt=attempt + 1) 
            else:
                print(f"❌ Build process failed after {attempt} attempts. Exiting.")
                exit(1)
        except FileNotFoundError:
            print(f"❌ Error: The build script '{BUILD_SCRIPT_NAME}' was not found.")
            print("Ensure it's in the same directory as faiss_utils.py, or in your system's PATH,")
            print("or update BUILD_SCRIPT_NAME with the correct path.")
            exit(1)
    else:
        print(f"❌ Failed to load or build FAISS index after {MAX_BUILD_ATTEMPTS} attempts. Giving up.")
        return None

if __name__ == "__main__":
    print("--- Running faiss_utils.py directly ---")
    vector_db = load_faiss_index()
    if vector_db:
        print("\n🎉 FAISS Vector Store is ready for use in faiss_utils.py.")
    else:
        print("\n😭 Could not initialize FAISS Vector Store.")