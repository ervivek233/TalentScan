from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from src.api.schemas import ResumeText, ResumeJDInput
from src.api.model_loader import (
    tfidf_vectorizer,
    resume_classifier,
    sentence_model
)

app = FastAPI(title="TalentScan Resume Analyzer.")
@app.get("/health")
def health():
    return {"status": "Ok"}

@app.post("/predict-category")
def predict_category(payload: ResumeText):
    vector = tfidf_vectorizer.transform([payload.text])
    prediction = resume_classifier.predict(vector)[0]
    return {"predicted_category": prediction}

@app.post("/match-jd")
def match_jd(payload: ResumeJDInput):
    resume_emb = sentence_model.encode([payload.resume_text])
    jd_emb = sentence_model.encode([payload.jd_text])

    score = cosine_similarity(resume_emb, jd_emb)[0][0]
    return {"semantic_similarity": round(float(score), 4)}

@app.post("/score-resume")
def score_resume(payload: ResumeJDInput):
    resume_emb = sentence_model.encode([payload.resume_text])
    jd_emb = sentence_model.encode([payload.jd_text])

    semantic = float(cosine_similarity(resume_emb, jd_emb)[0][0])

    final_score = round(semantic * 100, 2)  # placeholder, full logic later
    return {"resume_score": final_score}