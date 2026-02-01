import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

MODEL_DIR = "models/sentence_bert"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load cleaned data
resumes = pd.read_csv("data/processed/clean_resumes.csv")
jds = pd.read_csv("data/processed/clean_job_descriptions.csv")

# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Save model for reuse
model.save(MODEL_DIR)

# Encode texts
resume_embeddings = model.encode(
    resumes["clean_text"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

jd_embeddings = model.encode(
    jds["clean_text"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

# Example: match first resume with all JDs
similarities = cosine_similarity(
    resume_embeddings[0].reshape(1, -1),
    jd_embeddings
)[0]

top_matches = np.argsort(similarities)[::-1][:5]

print("\nTop 5 matching Job Descriptions:")
for idx in top_matches:
    print(f"Score: {similarities[idx]:.4f} | JD: {jds.iloc[idx]['clean_text'][:80]}...")
