import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# Configuration
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "documentation_embeddings"

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL)

# Load our embedding model
model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

# Example documentation
documents = [
    {
        "id": 1,
        "title": "VPN Setup",
        "text": "Employees use a VPN to connect remotely."
    },
    {
        "id": 2,
        "title": "Password Policy",
        "text": "Passwords must meet security requirements."
    },
    {
        "id": 3,
        "title": "Employee Handbook",
        "text": "Employees can request paid time off."
    }
]

# Extract text for embedding
texts = [doc["text"] for doc in documents]

# Generate document embeddings
embeddings = model.encode_document(
    texts,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)

# Create Qdrant points
points = []

for doc, embedding in zip(documents, embeddings):
    point = models.PointStruct(
        id=doc["id"],
        vector=embedding.tolist(),
        payload={
            "title": doc["title"],
            "text": doc["text"]
        }
    )

    points.append(point)

# Insert the points into Qdrant
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True
)

print(f"Uploaded {len(points)} points to Qdrant.")
