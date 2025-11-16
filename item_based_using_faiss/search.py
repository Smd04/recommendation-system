import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index and metadata
index = faiss.read_index("products_faiss.index")
df = pd.read_pickle("products_metadata.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")  # M2 GPU

# Recommend function
def recommend(query, k=5):
    vec = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(np.array(vec), k)

    results = df.iloc[indices[0]].copy()
    results["score"] = distances[0]

    return results[["Product Name", "Category", "score", "combined_text"]]

# Example search
query = "cat"
print(recommend(query, k=5))
