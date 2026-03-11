
import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')




print("Loading dataset...")

df = pd.read_csv("data/raw/IMDB_Dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())



stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", "", text)  # remove html tags

    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation & numbers

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)


print("Cleaning text data...")

df["clean_review"] = df["review"].apply(clean_text)

print(df[["review","clean_review"]].head())




print("Applying TF-IDF vectorization...")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_review"])

y = df["sentiment"]

print("Feature Matrix Shape:", X.shape)




print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)




print("Training Logistic Regression model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)




print("Evaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))




print("Saving model and vectorizer...")

with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully.")




print("\nTesting sample prediction...")

sample_review = "The event was amazing and I really enjoyed it"

clean_sample = clean_text(sample_review)

vector_sample = vectorizer.transform([clean_sample])

prediction = model.predict(vector_sample)

print("\nSample Review:", sample_review)

print("Predicted Sentiment:", prediction[0])