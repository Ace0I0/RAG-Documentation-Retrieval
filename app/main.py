import json
import os
from urllib.request import urlopen
from urllib.error import URLError

from fastapi import FastAPI

app = FastAPI()

# Read the address from our environment.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

@app.get("/")
def root():
    return {"message": "RAG Documentation Retrieval API"}

@app.get("/collections")
def get_collections():
    try:
        # Request Qdrant's collections.
        with urlopen(
            f"{QDRANT_URL}/collections",
            timeout=5
        ) as response:
            data = json.load(response)

        # Extract collection names.
        collections = [item["name"]for item in data["result"]["collections"]]

        return {"collections": collections}

    except (URLError, TimeoutError, ValueError, KeyError):
        raise HTTPException(status_code=503, detail="Unable to retrieve Qdrant collections")

