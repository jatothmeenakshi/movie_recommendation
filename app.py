import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_model():
    movies = pd.read_csv("https://raw.githubusercontent.com/LearnDataSci/articles/master/Python%20Pandas%20Tutorial%20A%20Complete%20Introduction%20for%20Beginners/IMDB-Movie-Data.csv")

    movies = movies.rename(columns={
        "Title": "title",
        "Genre": "genres",
        "Rating": "vote_average",
        "Votes": "popularity",
        "Description": "overview"
    })

    movies = movies[["title", "overview", "genres",
                     "vote_average", "popularity"]]
    movies["overview"] = movies["overview"].fillna("")
    movies["combined"] = movies["overview"]
    movies = movies.reset_index()

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["combined"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies["title"])

    return cosine_sim, movies, indices

def get_recommendations(title, num=10):
    cosine_sim, movies, indices = load_model()
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    result = movies[["title", "vote_average",
                     "popularity"]].iloc[movie_indices].copy()
    result["similarity"] = [round(i[1]*100, 1) for i in sim_scores]
    result.columns = ["Title", "Rating", "Popularity", "Match %"]
    result = result.reset_index(drop=True)
    result.index += 1
    return result

# Load model
cosine_sim, movies, indices = load_model()

# UI
st.title("🎬 Movie Recommendation System")
st.markdown("Select a movie and discover similar ones instantly!")
st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    movie_list = sorted(movies["title"].dropna().unique().tolist())
    selected_movie = st.selectbox(
        "Choose a movie you like:",
        movie_list,
        index=0
    )
with col2:
    num_recs = st.slider("How many?", 5, 15, 10)

st.divider()

if st.button("🎯 Get Recommendations", use_container_width=True):
    with st.spinner("Finding similar movies..."):
        recommendations = get_recommendations(selected_movie, num_recs)

    if recommendations is None:
        st.error("Movie not found!")
    else:
        st.subheader(f"Top {num_recs} movies similar to '{selected_movie}'")

        m1, m2, m3 = st.columns(3)
        m1.metric("Movies Analyzed", "1,000+")
        m2.metric("Top Match", f"{recommendations['Match %'].iloc[0]}%")
        m3.metric("Avg Rating",
                  f"{recommendations['Rating'].mean():.1f}/10")

        st.divider()

        cols = st.columns(2)
        for i, (_, row) in enumerate(recommendations.iterrows()):
            with cols[i % 2]:
                match_color = (
                    "🟢" if row["Match %"] > 15
                    else "🟡" if row["Match %"] > 8
                    else "🔴"
                )
                st.container(border=True).markdown(
                    f"**{i}. {row['Title']}**\n\n"
                    f"⭐ Rating: {row['Rating']}/10 &nbsp;&nbsp;"
                    f"{match_color} Match: {row['Match %']}%"
                )

        st.divider()
        st.subheader("📊 Full Results Table")
        st.dataframe(recommendations, use_container_width=True)