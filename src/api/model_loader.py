import joblib
import os
from sentence_transformers import SentenceTransformer

# Load artifacts once (on startup)

tfidf_vectorizer = joblib.load("data/processed/tfidf_vectorizer.joblib")
resume_classifier = joblib.load("models/resume_classifier.joblib")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")