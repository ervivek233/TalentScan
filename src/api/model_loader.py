import joblib
from sentence_transformers import SentenceTransformer

# Load artifacts once (on startup)

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

MODEL_DIR = BASE_DIR / "resources" / "models"

tfidf_vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
resume_classifier = joblib.load(MODEL_DIR / "resume_classifier.joblib")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")