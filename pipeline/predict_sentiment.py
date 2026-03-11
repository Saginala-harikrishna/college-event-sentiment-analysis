# predict_sentiment.py

# ===============================
# 1. Import Libraries
# ===============================

import pandas as pd
import pickle
import re


# ===============================
# 2. Load Model and Vectorizer
# ===============================

print("Loading model...")

with open("models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Model loaded successfully")


# ===============================
# 3. Text Cleaning Function
# ===============================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text


# ===============================
# 4. Load Google Form Responses
# ===============================

print("Loading responses...")

df = pd.read_csv("data/raw/responses.csv")

print(df.head())


# ===============================
# 5. Convert Day-wise Reviews to Rows
# ===============================

data_list = []

for index, row in df.iterrows():

    if pd.notna(row["Day1 Review"]):
        data_list.append({
            "day": "Day 1",
            "review": row["Day1 Review"]
        })

    if pd.notna(row["Day2 Review"]):
        data_list.append({
            "day": "Day 2",
            "review": row["Day2 Review"]
        })

    if pd.notna(row["Day3 Review"]):
        data_list.append({
            "day": "Day 3",
            "review": row["Day3 Review"]
        })


reviews_df = pd.DataFrame(data_list)

print("\nTransformed Dataset:")
print(reviews_df.head())


# ===============================
# 6. Clean Reviews
# ===============================

reviews_df["clean_review"] = reviews_df["review"].apply(clean_text)


# ===============================
# 7. Convert Text → Features
# ===============================

X = vectorizer.transform(reviews_df["clean_review"])


# ===============================
# 8. Predict Sentiment
# ===============================

predictions = model.predict(X)

reviews_df["sentiment"] = predictions


# ===============================
# 9. Save Processed Dataset
# ===============================

output_path = "data/processed/sentiment_results.csv"

reviews_df.to_csv(output_path, index=False)

print("\nSentiment predictions saved to:", output_path)

print("\nSample Results:")

print(reviews_df.head())