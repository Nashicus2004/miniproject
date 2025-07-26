# app.py
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Anime & Movie Recommender", layout="wide")

# Load dataset
CSV_PATH = "merged_anime_movie_dataset.csv"
df = pd.read_csv(CSV_PATH)

# Fill NaNs
df['description'] = df['description'].fillna("")
df['genre'] = df['genre'].fillna("")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df['description'])

# Genre list
genres_raw = df['genre'].astype(str)
genre_options = sorted(set(g.strip() for s in genres_raw for g in s.replace('|', ',').replace(';', ',').split(',')))
title_options = sorted(df['title'].dropna().unique())
mood_options = [
    "Action", "Romantic", "Sad", "Happy", "Dark", "Wholesome", "Funny", "Depressing",
    "Inspirational", "Chill", "Adventurous", "Mysterious", "Violent", "Light-hearted",
    "Tragic", "Philosophical", "Suspenseful", "Feel-good", "Thrilling", "Drama"
]

st.title("🎥 ML-powered Anime & Movie Recommender")

# UI Inputs
mood = st.selectbox("Mood", mood_options)
last_watched = st.selectbox("Last Watched Title", [""] + title_options)
selected_genres = st.multiselect("Select Genres", genre_options)
top_k = st.slider("How many results?", 1, 20, 5)

# Recommendation function using ML
def recommend_ml(mood, last_watched, genres, top_k):
    df_filtered = df.copy()

    # Filter by selected genres
    if genres:
        genre_mask = df_filtered['genre'].apply(lambda g: all(gen.lower() in g.lower() for gen in genres))
        df_filtered = df_filtered[genre_mask]

    # Filter by mood (search in description)
    if mood:
        mood_mask = df_filtered['description'].str.contains(mood, case=False)
        df_filtered = df_filtered[mood_mask]

    # ML Similarity based on description
    if last_watched:
        idx = df[df['title'] == last_watched].index
        if not idx.empty:
            idx = idx[0]
            cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            df_filtered['similarity'] = cosine_sim
            df_filtered = df_filtered.sort_values("similarity", ascending=False)
        else:
            df_filtered['similarity'] = 0  # fallback
    else:
        df_filtered['similarity'] = 0  # fallback

    return df_filtered.head(top_k)

# Show Recommendations
if st.button("🎬 Recommend!"):
    if not selected_genres:
        st.warning("Please select at least one genre.")
    else:
        results = recommend_ml(mood, last_watched, selected_genres, top_k)
        st.success(f"Top {top_k} Recommendations:")
        st.dataframe(results[["title", "type", "genre", "score", "description"]].style.format(subset=["description"], formatter=lambda d: d[:150] + "..."))
