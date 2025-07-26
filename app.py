import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Anime & Movie Recommender", layout="wide")

CSV_PATH = "merged_anime_movie_dataset.csv"
df = pd.read_csv(CSV_PATH)

df['description'] = df['description'].fillna("")
df['genre'] = df['genre'].fillna("")

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df['description'])

genres_raw = df['genre'].astype(str)
genre_options = sorted(set(g.strip() for s in genres_raw for g in s.replace('|', ',').replace(';', ',').split(',')))
title_options = sorted(df['title'].dropna().unique())
mood_options = [
    "Action", "Romantic", "Sad", "Happy", "Dark", "Wholesome", "Funny", "Depressing",
    "Inspirational", "Chill", "Adventurous", "Mysterious", "Violent", "Light-hearted",
    "Tragic", "Philosophical", "Suspenseful", "Feel-good", "Thrilling", "Drama"
]

st.title("🎥 ML-powered Anime & Movie Recommender")

mood = st.selectbox("Mood", mood_options)
last_watched = st.selectbox("Last Watched Title", [""] + title_options)
selected_genres = st.multiselect("Select Genres", genre_options)
top_k = st.slider("How many results?", 1, 20, 5)

def recommend_ml(mood, last_watched, genres, top_k):
    df_filtered = df.copy()

    if genres:
        genre_mask = df_filtered['genre'].apply(lambda g: all(gen.lower() in g.lower() for gen in genres))
        df_filtered = df_filtered[genre_mask]

    if mood:
        mood_mask = df_filtered['description'].str.contains(mood, case=False)
        df_filtered = df_filtered[mood_mask]

    similarity_series = pd.Series(0, index=df.index)

    if last_watched:
        idx = df[df['title'] == last_watched].index
        if not idx.empty:
            idx = idx[0]
            cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            similarity_series = pd.Series(cosine_sim, index=df.index)

    df_filtered = df_filtered.copy()
    df_filtered['similarity'] = similarity_series.loc[df_filtered.index]
    df_filtered = df_filtered.sort_values("similarity", ascending=False)

    if df_filtered.empty:
        fallback = df.copy()
        fallback['similarity'] = similarity_series
        fallback = fallback.sort_values("similarity", ascending=False)
        return fallback.head(top_k)

    return df_filtered.head(top_k)

if st.button("🎬 Recommend!"):
    if not selected_genres:
        st.warning("Please select at least one genre.")
    else:
        results = recommend_ml(mood, last_watched, selected_genres, top_k)
        if results.empty:
            st.info("No exact matches found. Showing top picks instead.")
            results = df.sort_values("score", ascending=False).head(top_k)
        try:
            st.success(f"Top {top_k} Recommendations:")
            st.dataframe(
                results[["title", "type", "genre", "score", "description"]]
                .style.format(subset=["description"], formatter=lambda d: str(d)[:150] + "..." if pd.notna(d) else "")
            )
        except Exception as e:
            st.error(f"Something went wrong while displaying the results: {e}")
