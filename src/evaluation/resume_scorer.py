import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
resumes = pd.read_csv("data/processed/clean_resumes.csv")
jds = pd.read_csv("data/processed/clean_job_descriptions.csv")
skills_df = pd.read_csv("data/raw/skills.csv")

skills_list = set(skills_df["skill"].str.lower())

# Load Sentence-BERT
model = SentenceTransformer("all-MiniLM-L6-v2")

def skill_match_score(resume_text: str, jd_text: str) -> float:
    resume_skills = {s for s in skills_list if s in resume_text}
    jd_skills = {s for s in skills_list if s in jd_text}

    if not jd_skills:
        return 0.0

    return len(resume_skills & jd_skills) / len(jd_skills)

def resume_quality_score(resume_text: str) -> float:
    length = len(resume_text.split())

    if length < 50:
        return 0.3
    elif length < 150:
        return 0.6
    else:
        return 1.0

def final_resume_score(resume_text: str, jd_text: str) -> float:
    resume_emb = model.encode([resume_text])
    jd_emb = model.encode([jd_text])

    semantic_score = cosine_similarity(resume_emb, jd_emb)[0][0]
    skill_score = skill_match_score(resume_text, jd_text)
    quality_score = resume_quality_score(resume_text)

    final_score = (
        0.5 * semantic_score +
        0.3 * skill_score +
        0.2 * quality_score
    )

    return round(final_score * 100, 2)

# Demo run
sample_resume = resumes.iloc[0]["clean_text"]
sample_jd = jds.iloc[3]["clean_text"]

score = final_resume_score(sample_resume, sample_jd)
print("Sample Resume Text:", sample_resume)
print("Sample Job Description Text:", sample_jd)
print("Final Resume Score:", score)
