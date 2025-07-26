# app.py
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Anime & Movie Recommender", layout="wide")

CSV_PATH = "merged_anime_movie_dataset.csv"
df = pd.read_csv(CSV_PATH)

# Process genres
genres_raw = df['genre'].dropna().astype(str)
genre_options = sorted(set(g.strip() for s in genres_raw for g in s.replace('|', ',').replace(';', ',').split(',')))
title_options = sorted(df['title'].dropna().unique())
mood_options = [
    "Action", "Romantic", "Sad", "Happy", "Dark", "Wholesome", "Funny", "Depressing",
    "Inspirational", "Chill", "Adventurous", "Mysterious", "Violent", "Light-hearted",
    "Tragic", "Philosophical", "Suspenseful", "Feel-good", "Thrilling", "Drama"
]

st.title("🎥 Anime & Movie Recommender")

# Inputs
mood = st.selectbox("Mood", mood_options)
last_watched = st.selectbox("Last Watched Title", [""] + title_options)
selected_genres = st.multiselect("Select Genres", genre_options)
top_k = st.slider("How many results?", 1, 20, 5)

def recommend(mood, last_watched, genres, top_k):
    genre_mask = df['genre'].fillna('').apply(lambda g: all(gen.lower() in g.lower() for gen in genres))
    df_filtered = df[genre_mask] if any(genre_mask) else df.copy()
    mood_mask = df_filtered['description'].fillna('').str.contains(mood, case=False)
    mood_matches = df_filtered[mood_mask]
    similarity_mask = df_filtered['title'].fillna('').str.contains(last_watched, case=False)
    similarity_matches = df_filtered[similarity_mask]
    combined = pd.concat([mood_matches, similarity_matches]).drop_duplicates()
    return combined.sort_values("score", ascending=False).head(top_k) if not combined.empty else df_filtered.sort_values("score", ascending=False).head(top_k)

# Recommendation button
if st.button("🎬 Recommend!"):
    if not selected_genres:
        st.warning("Please select at least one genre.")
    else:
        results = recommend(mood, last_watched, selected_genres, top_k)
        st.success(f"Top {top_k} Recommendations:")
        st.dataframe(results[["title", "type", "genre", "score", "description"]].style.format(subset=["description"], formatter=lambda d: d[:150] + "..."))
