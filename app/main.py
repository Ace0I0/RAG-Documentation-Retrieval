
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "documentation_embeddings"
QDRANT_URL = os.getenv(
    "QDRANT_URL", "http://localhost:6333"
)


# Application startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Qwen3 embedding model...", flush=True)

    app.state.model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    app.state.qdrant = QdrantClient(
        url=QDRANT_URL
    )

    print("Search API ready!", flush=True)

    yield

    app.state.qdrant.close()


app = FastAPI(lifespan=lifespan)


# Defines the expected request body
class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=3, ge=1, le=10)


@app.get("/")
def root():
    return {"message": "RAG Documentation Retrieval API"}


@app.get("/collections")
def get_collections():
    collections = app.state.qdrant.get_collections()

    return {
        "collections": [
            item.name for item in collections.collections
        ]
    }


@app.post("/search")
def search_documents(request: SearchRequest):

    # Convert the question into an embedding
    query_vector = app.state.model.encode_query(
        request.question,
        normalize_embeddings=True
    )

    # Search existing embeddings in Qdrant
    results = app.state.qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=request.limit,
        with_payload=True
    )

    # Format the results as JSON
    documents = []

    for point in results.points:
        payload = point.payload or {}

        documents.append({
            "id": point.id,
            "title": payload.get("title", "Unknown"),
            "text": payload.get("text", ""),
            "score": round(point.score, 4)
        })

    return {
        "question": request.question,
        "results": documents
    }
