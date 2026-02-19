import chromadb
from chromadb.utils import embedding_functions

# --- 1. Setup the Embedding Function ---
# This ensures every piece of text is automatically vectorized
model_name = "all-MiniLM-L6-v2"
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_name
)

def get_chroma():
    # --- 2. Initialize the Persistent Client ---
    # This replaces the old "duckdb" settings and saves data to a folder
    client = chromadb.PersistentClient(path="db/chroma_storage")

    # --- 3. Get or Create Collection ---
    # We link the embedding_func here so you don't have to manually vectorise text later
    collection = client.get_or_create_collection(
        name="genasium_vectors",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )

    return client, collection

# Initialize them for use in this script
chroma_client, vector_collection = get_chroma()