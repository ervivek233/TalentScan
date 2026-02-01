import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_PATH = "data/processed/clean_resumes.csv"
VECTORIZER_PATH = "data/processed/tfidf_vectorizer.joblib"
FEATURES_PATH = "data/processed/resume_features.joblib"

# Load data
df = pd.read_csv(INPUT_PATH)

X_text = df["clean_text"]
y = df["Category"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = vectorizer.fit_transform(X_text)

# Save artifacts
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump((X_tfidf, y), FEATURES_PATH)

print("TF-IDF features created and saved.")
print("Feature shape:", X_tfidf.shape)
