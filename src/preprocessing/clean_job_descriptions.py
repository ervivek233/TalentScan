import pandas as pd
from text_cleaner import clean_text

RAW_PATH = "data/raw/job_descriptions.csv"
OUT_PATH = "data/processed/clean_job_descriptions.csv"

df = pd.read_csv(RAW_PATH)

df["clean_text"] = df.iloc[:, 0].astype(str).apply(clean_text)

df = df[["clean_text"]]

df.to_csv(OUT_PATH, index=False)

print("Clean job descriptions saved.")
