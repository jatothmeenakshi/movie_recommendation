import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Step 1: Loading dataset...")
movies = pd.read_csv("tmdb_5000_movies.csv")
print(f"Loaded! {len(movies)} movies found ✅")

print("Step 2: Preparing data...")
movies = movies[["title", "overview", "genres", "keywords", "popularity", "vote_average"]]
movies["overview"] = movies["overview"].fillna("")
movies["combined"] = movies["overview"]
movies = movies.reset_index()
print("Data ready! ✅")

print("Step 3: Building recommendation model...")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies["title"])
print("Model built! ✅")

print("Step 4: Saving files...")
with open("cosine_sim.pkl", "wb") as f:
    pickle.dump(cosine_sim, f)
with open("movies.pkl", "wb") as f:
    pickle.dump(movies, f)
with open("indices.pkl", "wb") as f:
    pickle.dump(indices, f)
print("All files saved! ✅")

print("")
print("DONE! Your 3 files are ready:")
print("  cosine_sim.pkl ✅")
print("  movies.pkl     ✅")
print("  indices.pkl    ✅")