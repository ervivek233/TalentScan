import pandas as pd
from text_cleaner import clean_text

RAW_PATH = "data/raw/resumes.csv"
OUT_PATH = "data/processed/clean_resumes.csv"

df = pd.read_csv(RAW_PATH)

df["clean_text"] = df["Resume"].astype(str).apply(clean_text)

df = df[["Category", "clean_text"]]

df.to_csv(OUT_PATH, index=False)

print("Clean resumes saved.")
