from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

from src.preprocessing.text_cleaner import clean_text
from src.api.resume_parser import extract_text_from_pdf, extract_text_from_docx

# Load skills
skills_df = pd.read_csv("data/raw/skills.csv")
SKILLS = set(skills_df["skill"].str.lower())

def skill_match_score(resume_text: str, jd_text: str) -> float:
    resume_skills = {s for s in SKILLS if s in resume_text}
    jd_skills = {s for s in SKILLS if s in jd_text}

    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)

def resume_quality_score(resume_text: str) -> float:
    length = len(resume_text.split())
    if length < 50:
        return 0.3
    elif length < 150:
        return 0.6
    return 1.0

def analyze_resume(
    resume_file,
    filename: str,
    jd_text: str,
    tfidf_vectorizer,
    resume_classifier,
    sentence_model: SentenceTransformer
):
    # 1. Extract resume text
    if filename.endswith(".pdf"):
        raw_text = extract_text_from_pdf(resume_file)
    elif filename.endswith(".docx"):
        raw_text = extract_text_from_docx(resume_file)
    else:
        raise ValueError("Unsupported file type")

    # 2. Clean text
    clean_resume = clean_text(raw_text)
    clean_jd = clean_text(jd_text)

    # 3. Predict category
    vector = tfidf_vectorizer.transform([clean_resume])
    predicted_category = resume_classifier.predict(vector)[0]

    # 4. Semantic similarity
    resume_emb = sentence_model.encode([clean_resume])
    jd_emb = sentence_model.encode([clean_jd])
    semantic_score = float(cosine_similarity(resume_emb, jd_emb)[0][0])

    # 5. Skill + quality
    skill_score = skill_match_score(clean_resume, clean_jd)
    quality_score = resume_quality_score(clean_resume)

    # 6. Final score
    final_score = round(
        (0.5 * semantic_score + 0.3 * skill_score + 0.2 * quality_score) * 100,
        2
    )

    return {
        "predicted_category": predicted_category,
        "semantic_similarity": round(semantic_score, 4),
        "skill_match_score": round(skill_score, 4),
        "final_resume_score": final_score
    }
