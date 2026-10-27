from sentence_transformers import SentenceTransformer

# Load our embedding model
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")

# Documents we want to search
documents = [
    "Employees use a VPN to connect remotely.",
    "Passwords must meet security requirements.",
    "Employees can request paid time off."
]

# A question someone might ask
query = "How can I access the company network from home?"

# Convert documents into numerical vectors
document_vectors = model.encode_document(documents, normalize_embeddings=True)

# Convert the user's question into a vector
query_vector = model.encode_query(query, normalize_embeddings=True)

# Compare the query against each document
scores = document_vectors @ query_vector

print("Embedding dimensions:", len(query_vector))
print("\nSearch results:")

results = sorted(zip(documents, scores), key=lambda result: float(result[1]), reverse=True)

for document, score in results:
    print(f"{float(score):.4f} | {document}")
