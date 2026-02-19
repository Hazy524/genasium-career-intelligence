import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from init_chroma import get_chroma

def search_chroma():
    # 1. Connect
    client, collection = get_chroma()

    # 2. Define a search term
    # Notice we don't use the word "React" or "Tailwind", 
    # but we ask for something related.
    query_text = "Who is the best person to build a website?"

    print(f"Searching for: '{query_text}'...")

    # 3. Query the collection
    # n_results=1 means "give me the single best match"
    results = collection.query(
        query_texts=[query_text],
        n_results=1
    )

    # 4. Show the results
    print("\n--- Search Results ---")
    for i, doc in enumerate(results['documents'][0]):
        print(f"Match: {doc}")
        print(f"ID: {results['ids'][0][i]}")
        print(f"Distance (Lower is better): {results['distances'][0][i]}")

if __name__ == "__main__":
    search_chroma()