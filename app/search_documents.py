import argparse
import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Configuration
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "documentation_embeddings"

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)


def main():
    # Read the search question from the terminal.
    parser = argparse.ArgumentParser(
        description="Search indexed documentation"
    )

    parser.add_argument(
        "question",
        type=str,
        help="The question to search for"
    )

    args = parser.parse_args()

    # Connect to our Qdrant database.
    client = QdrantClient(url=QDRANT_URL)

    # Load the same model used during indexing.
    print("Loading embedding model...", flush=True)

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    # Convert the question into an embedding.
    print("Generating query embedding...", flush=True)

    query_vector = model.encode_query(
        args.question,
        normalize_embeddings=True
    )

    # Search Qdrant using the generated vector.
    print("Searching Qdrant...", flush=True)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=3,
        with_payload=True
    )

    # Display the search results.
    print(f"\nQuestion: {args.question}")
    print("\nSearch Results:\n")

    if not results.points:
        print("No documents found.")
        return

    for rank, point in enumerate(results.points, start=1):
        payload = point.payload or {}

        print(f"Rank: {rank}")
        print(f"Document: {payload.get('title', 'Unknown')}")
        print(f"Score: {point.score:.4f}")
        print(f"Text: {payload.get('text', '')}")
        print("-" * 40)


if __name__ == "__main__":
    main()
