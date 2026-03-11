import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="College Event Sentiment Dashboard",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------

st.title("🎓 College Event Feedback Analytics")
st.markdown("Analyze student feedback from the **3-day college event** using NLP and Machine Learning.")

# -----------------------------
# File Path (Cloud Safe)
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "sentiment_results.csv")

# -----------------------------
# Load Data
# -----------------------------

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found. Please generate sentiment_results.csv first.")
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    return df


df = load_data()

if df.empty:
    st.stop()

# -----------------------------
# Sidebar Filters
# -----------------------------

st.sidebar.header("Dashboard Filters")

day_filter = st.sidebar.selectbox(
    "Select Day",
    ["All", "Day 1", "Day 2", "Day 3"]
)

sentiment_filter = st.sidebar.selectbox(
    "Select Sentiment",
    ["All", "positive", "negative"]
)

filtered_df = df.copy()

if day_filter != "All":
    filtered_df = filtered_df[filtered_df["day"] == day_filter]

if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df["sentiment"] == sentiment_filter]

# -----------------------------
# Metrics Section
# -----------------------------

st.subheader("📊 Feedback Summary")

total_reviews = len(filtered_df)
positive_reviews = (filtered_df["sentiment"] == "positive").sum()
negative_reviews = (filtered_df["sentiment"] == "negative").sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total Reviews", total_reviews)
col2.metric("Positive Reviews", positive_reviews)
col3.metric("Negative Reviews", negative_reviews)

st.markdown("---")

# -----------------------------
# Charts Section
# -----------------------------

col4, col5 = st.columns(2)

with col4:

    st.subheader("Sentiment Distribution")

    sentiment_counts = filtered_df["sentiment"].value_counts()

    if not sentiment_counts.empty:

        fig1 = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                "positive": "green",
                "negative": "red"
            }
        )

        st.plotly_chart(fig1, use_container_width=True)

with col5:

    st.subheader("Sentiment by Day")

    day_sentiment = filtered_df.groupby(["day", "sentiment"]).size().reset_index(name="count")

    if not day_sentiment.empty:

        fig2 = px.bar(
            day_sentiment,
            x="day",
            y="count",
            color="sentiment",
            barmode="group",
            color_discrete_map={
                "positive": "green",
                "negative": "red"
            }
        )

        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# -----------------------------
# Word Cloud Section
# -----------------------------

st.subheader("☁ Word Cloud of Feedback")

if len(filtered_df) > 0:

    text = " ".join(filtered_df["review"].astype(str))

    if text.strip() != "":

        wordcloud = WordCloud(
            width=1000,
            height=400,
            background_color="white",
            colormap="viridis"
        ).generate(text)

        fig, ax = plt.subplots()

        ax.imshow(wordcloud)
        ax.axis("off")

        st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Reviews Table
# -----------------------------

st.subheader("📋 Student Reviews")

st.dataframe(filtered_df, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------



