import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")
movies = pd.read_csv("https://raw.githubusercontent.com/LearnDataSci/articles/master/Python%20Pandas%20Tutorial%20A%20Complete%20Introduction%20for%20Beginners/IMDB-Movie-Data.csv")

print(f"Loaded! {len(movies)} movies found ✅")

# Rename columns to match
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

with open("cosine_sim.pkl", "wb") as f:
    pickle.dump(cosine_sim, f)
with open("movies.pkl", "wb") as f:
    pickle.dump(movies, f)
with open("indices.pkl", "wb") as f:
    pickle.dump(indices, f)

print("✅ All files saved!")