from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from src.api.schemas import ResumeText, ResumeJDInput
from fastapi import UploadFile, File, HTTPException
from fastapi import UploadFile, File, Form, HTTPException
from src.api.analyzer import analyze_resume
from src.api.resume_parser import extract_text_from_pdf, extract_text_from_docx
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

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file.file)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file.file)
    else:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported"
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in resume"
        )

    return {
        "filename": file.filename,
        "extracted_text_preview": text[:500]
    }

@app.post("/analyze-resume")
async def analyze_resume_api(
    resume_file: UploadFile = File(...),
    jd_text: str = Form(...)
):
    try:
        result = analyze_resume(
            resume_file=resume_file.file,
            filename=resume_file.filename.lower(),
            jd_text=jd_text,
            tfidf_vectorizer=tfidf_vectorizer,
            resume_classifier=resume_classifier,
            sentence_model=sentence_model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
