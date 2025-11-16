from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import torch

# 1️⃣ Load dataset
ds = load_dataset("calmgoose/amazon-product-data-2020", split="train")
df = ds.to_pandas()

# 2️⃣ Fill missing values for relevant columns
cols = ["Product Name", "Category", "About Product", "Product Specification", "Technical Details"]
df = df[cols].fillna("")

# 3️⃣ Combine text for embedding
def combine_text(row):
    return (
        f"Product Name: {row['Product Name']}\n"
        f"Category: {row['Category']}\n"
        f"About Product: {row['About Product']}\n"
        f"Product Specification: {row['Product Specification']}\n"
        f"Technical Details: {row['Technical Details']}"
    )

df["combined_text"] = df.apply(combine_text, axis=1)

# 4️⃣ Save metadata (full dataset)
df.to_pickle("products_metadata.pkl")

# 5️⃣ Generate embeddings (GPU-enabled if MPS available)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")

embeddings = model.encode(
    df["combined_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# 6️⃣ Build FAISS index
embeddings = np.array(embeddings).astype("float32")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# 7️⃣ Save FAISS index
faiss.write_index(index, "products_faiss.index")

print("✅ Metadata + FAISS index built successfully!")
