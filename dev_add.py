import sys
import os

# Make sure Python can find init_chroma
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from init_chroma import get_chroma

def add_test_data():
    # Connect using the new persistent Chroma client
    client, collection = get_chroma()

    # Sample documents
    documents = [
        "Experienced in building web apps with Next.js and Tailwind CSS",
        "Data analyst with 3 years of experience in Python and SQL",
        "Machine Learning engineer focused on NLP and Vector Databases"
    ]

    metadatas = [
        {"source": "resume"},
        {"source": "resume"},
        {"source": "resume"}
    ]

    ids = ["id1", "id2", "id3"]

    print("Adding data to ChromaDB...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    # Check count
    print(f"Success! Collection now has {collection.count()} items.")

if __name__ == "__main__":
    add_test_data()

