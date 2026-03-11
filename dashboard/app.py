import streamlit as st
import pandas as pd

st.title("College Event Sentiment Analysis Dashboard")

df = pd.read_csv("data/processed/sentiment_results.csv")

st.subheader("Feedback Dataset")

st.dataframe(df)