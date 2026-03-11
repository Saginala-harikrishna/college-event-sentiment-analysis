# predict_sentiment.py

# ======================================
# 1. Import Libraries
# ======================================

import pandas as pd
import pickle
import re
import os


# ======================================
# 2. Define File Paths (Cloud Safe)
# ======================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

INPUT_DATA = os.path.join(BASE_DIR, "data", "raw", "responses.csv")
OUTPUT_DATA = os.path.join(BASE_DIR, "data", "processed", "sentiment_results.csv")


# ======================================
# 3. Load Model and Vectorizer
# ======================================

print("Loading model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("Model loaded successfully")


# ======================================
# 4. Text Cleaning Function
# ======================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text


# ======================================
# 5. Load Google Form Responses
# ======================================

print("Loading responses...")

df = pd.read_csv(INPUT_DATA)

print("Dataset Loaded")
print(df.head())


# ======================================
# 6. Convert Day-wise Reviews to Rows
# ======================================

data_list = []

for _, row in df.iterrows():

    if "Day1 Review" in df.columns and pd.notna(row["Day1 Review"]):
        data_list.append({
            "day": "Day 1",
            "review": row["Day1 Review"]
        })

    if "Day2 Review" in df.columns and pd.notna(row["Day2 Review"]):
        data_list.append({
            "day": "Day 2",
            "review": row["Day2 Review"]
        })

    if "Day3 Review" in df.columns and pd.notna(row["Day3 Review"]):
        data_list.append({
            "day": "Day 3",
            "review": row["Day3 Review"]
        })


reviews_df = pd.DataFrame(data_list)

print("\nTransformed Dataset:")
print(reviews_df.head())


# ======================================
# 7. Clean Reviews
# ======================================

reviews_df["clean_review"] = reviews_df["review"].apply(clean_text)


# ======================================
# 8. Convert Text → Features
# ======================================

X = vectorizer.transform(reviews_df["clean_review"])


# ======================================
# 9. Predict Sentiment
# ======================================

predictions = model.predict(X)

reviews_df["sentiment"] = predictions


# ======================================
# 10. Ensure Processed Folder Exists
# ======================================

os.makedirs(os.path.dirname(OUTPUT_DATA), exist_ok=True)


# ======================================
# 11. Save Results
# ======================================

reviews_df.to_csv(OUTPUT_DATA, index=False)

print("\nSentiment predictions saved to:", OUTPUT_DATA)

print("\nSample Results:")
print(reviews_df.head())