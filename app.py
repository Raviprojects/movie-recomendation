"""
app.py
------
Flask backend for the Movie Recommendation System.

Routes:
    GET  /                     -> Home page (search + genre filter UI)
    POST /recommend            -> Search a movie, get content-based recommendations
    GET  /genre/<genre_name>   -> Browse movies by genre
    GET  /movie/<title>        -> Single movie details page

Model:
    Loads model/movie_recommendation.pkl (TF-IDF + cosine similarity,
    content-based). The loader is deliberately flexible about the pickle's
    internal shape -- see load_model() below and the README for details.

TMDB:
    get_movie_details(title) enriches each recommendation with a poster,
    overview, rating, release date, and genre pulled from the TMDB API.
    If the API key is missing or the request fails, a safe placeholder
    is returned instead of crashing the page.
"""

import os
import pickle

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "movie_recommendation.pkl")

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

DEFAULT_POSTER = "/static/images/default-poster.svg"

GENRES = [
    "Action", "Comedy", "Horror", "Sci-Fi",
    "Thriller", "Romance", "Animation", "Drama",
]

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model(path=MODEL_PATH):
    """
    Loads the trained recommendation model.

    Supports a few common pickle shapes so this works whether you built the
    model with model/build_model.py or brought your own file:

      1. dict with keys: "movies" (DataFrame), "cosine_sim" (2D array)
         -> used directly.
      2. dict with keys: "movies" (DataFrame), "tfidf_matrix" (sparse matrix)
         -> cosine similarity is computed on the fly.
      3. A bare pandas DataFrame with a text column ("tags" or "overview")
         -> TF-IDF + cosine similarity is computed on the fly.
      4. A (DataFrame, cosine_sim) tuple.

    Returns (movies_df, cosine_sim) or (None, None) if loading fails.
    """
    if not os.path.exists(path):
        print(f"[model] WARNING: {path} not found.")
        return None, None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        print(f"[model] ERROR loading pickle: {exc}")
        return None, None

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    movies_df = None
    cosine_sim = None

    if isinstance(data, dict):
        for key in ("movies", "df", "movies_df"):
            if data.get(key) is not None:
                movies_df = data[key]
                break

        cosine_sim = None
        for key in ("cosine_sim", "similarity"):
            if data.get(key) is not None:
                cosine_sim = data[key]
                break

        tfidf_matrix = data.get("tfidf_matrix")

        if cosine_sim is None and tfidf_matrix is not None:
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    elif isinstance(data, tuple) and len(data) == 2:
        movies_df, cosine_sim = data

    elif isinstance(data, pd.DataFrame):
        movies_df = data

    if movies_df is None:
        print("[model] WARNING: could not find a movies DataFrame in the pickle.")
        return None, None

    movies_df = movies_df.reset_index(drop=True)

    # Normalize column names we rely on.
    if "title" not in movies_df.columns:
        raise ValueError("Model DataFrame must contain a 'title' column.")
    if "genre" not in movies_df.columns:
        movies_df["genre"] = "Unknown"

    if cosine_sim is None:
        text_col = "tags" if "tags" in movies_df.columns else (
            "overview" if "overview" in movies_df.columns else None
        )
        if text_col is None:
            raise ValueError(
                "Model DataFrame needs a 'tags' or 'overview' text column "
                "when no precomputed similarity matrix is provided."
            )
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(movies_df[text_col].fillna(""))
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return movies_df, np.asarray(cosine_sim)


MOVIES_DF, COSINE_SIM = load_model()


# --------------------------------------------------------------------------
# Recommendation logic
# --------------------------------------------------------------------------

def find_movie_index(title):
    """Case-insensitive, partial-match lookup of a movie title in the dataset."""
    if MOVIES_DF is None:
        return None

    titles_lower = MOVIES_DF["title"].str.lower()
    query = title.strip().lower()

    exact = titles_lower[titles_lower == query]
    if not exact.empty:
        return exact.index[0]

    partial = titles_lower[titles_lower.str.contains(query, na=False, regex=False)]
    if not partial.empty:
        return partial.index[0]

    return None


def get_recommendations(title, top_n=8):
    """Returns (matched_title, [recommended_title, ...]) or (None, []) if not found."""
    idx = find_movie_index(title)
    if idx is None:
        return None, []

    matched_title = MOVIES_DF.loc[idx, "title"]
    sim_scores = list(enumerate(COSINE_SIM[idx]))
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    recommended_titles = []
    for movie_idx, score in sim_scores:
        if movie_idx == idx:
            continue
        recommended_titles.append(MOVIES_DF.loc[movie_idx, "title"])
        if len(recommended_titles) >= top_n:
            break

    return matched_title, recommended_titles


def get_movies_by_genre(genre_name, limit=12):
    if MOVIES_DF is None:
        return []
    mask = MOVIES_DF["genre"].str.lower() == genre_name.strip().lower()
    return MOVIES_DF.loc[mask, "title"].tolist()[:limit]


# --------------------------------------------------------------------------
# TMDB integration
# --------------------------------------------------------------------------

def _placeholder_details(title, genre="Unknown"):
    return {
        "title": title,
        "poster": DEFAULT_POSTER,
        "rating": "N/A",
        "overview": "No description available for this title.",
        "release_date": "N/A",
        "genre": genre,
    }


def get_movie_details(movie_name, fallback_genre="Unknown"):
    """
    Looks up a movie on TMDB and returns:
        { title, poster, rating, overview, release_date, genre }

    Falls back to a safe placeholder (default poster, "N/A" fields) if the
    API key is missing, the movie isn't found, or the request fails for any
    reason -- callers never need to handle exceptions themselves.
    """
    if not TMDB_API_KEY:
        return _placeholder_details(movie_name, fallback_genre)

    try:
        search_resp = requests.get(
            f"{TMDB_BASE_URL}/search/movie",
            params={"api_key": TMDB_API_KEY, "query": movie_name},
            timeout=6,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("results", [])

        if not results:
            return _placeholder_details(movie_name, fallback_genre)

        movie = results[0]
        movie_id = movie.get("id")

        genre_names = fallback_genre
        if movie_id:
            details_resp = requests.get(
                f"{TMDB_BASE_URL}/movie/{movie_id}",
                params={"api_key": TMDB_API_KEY},
                timeout=6,
            )
            if details_resp.ok:
                details = details_resp.json()
                genres = details.get("genres", [])
                if genres:
                    genre_names = ", ".join(g["name"] for g in genres)

        poster_path = movie.get("poster_path")
        poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else DEFAULT_POSTER

        return {
            "title": movie.get("title", movie_name),
            "poster": poster_url,
            "rating": movie.get("vote_average", "N/A"),
            "overview": movie.get("overview") or "No description available.",
            "release_date": movie.get("release_date") or "N/A",
            "genre": genre_names,
        }

    except requests.RequestException as exc:
        print(f"[TMDB] request failed for '{movie_name}': {exc}")
        return _placeholder_details(movie_name, fallback_genre)
    except Exception as exc:
        print(f"[TMDB] unexpected error for '{movie_name}': {exc}")
        return _placeholder_details(movie_name, fallback_genre)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    model_ready = MOVIES_DF is not None
    return render_template("index.html", genres=GENRES, model_ready=model_ready)


@app.route("/recommend", methods=["POST"])
def recommend():
    movie_name = request.form.get("movie", "").strip()

    if not movie_name:
        flash("Please enter a movie name to search.")
        return redirect(url_for("index"))

    if MOVIES_DF is None:
        flash("Recommendation model is not loaded. Please check the server logs.")
        return redirect(url_for("index"))

    matched_title, recommended_titles = get_recommendations(movie_name, top_n=8)

    if matched_title is None:
        flash(f'Movie not found. Try another movie.')
        return redirect(url_for("index"))

    # Look up genre from local dataset for better TMDB genre fallback.
    matched_row = MOVIES_DF[MOVIES_DF["title"] == matched_title].iloc[0]
    main_movie = get_movie_details(matched_title, fallback_genre=matched_row.get("genre", "Unknown"))

    recommended_movies = []
    for rec_title in recommended_titles:
        rec_row = MOVIES_DF[MOVIES_DF["title"] == rec_title].iloc[0]
        recommended_movies.append(
            get_movie_details(rec_title, fallback_genre=rec_row.get("genre", "Unknown"))
        )

    return render_template(
        "recommendations.html",
        mode="search",
        query=movie_name,
        main_movie=main_movie,
        recommended_movies=recommended_movies,
        genres=GENRES,
    )


@app.route("/genre/<genre_name>")
def genre_filter(genre_name):
    if genre_name not in GENRES:
        flash("Unknown genre selected.")
        return redirect(url_for("index"))

    if MOVIES_DF is None:
        flash("Recommendation model is not loaded. Please check the server logs.")
        return redirect(url_for("index"))

    titles = get_movies_by_genre(genre_name)

    if not titles:
        flash(f"No movies found for the '{genre_name}' genre yet.")
        return redirect(url_for("index"))

    movies = []
    for t in titles:
        row = MOVIES_DF[MOVIES_DF["title"] == t].iloc[0]
        movies.append(get_movie_details(t, fallback_genre=row.get("genre", genre_name)))

    return render_template(
        "recommendations.html",
        mode="genre",
        query=genre_name,
        main_movie=None,
        recommended_movies=movies,
        genres=GENRES,
    )


@app.route("/movie/<title>")
def movie_details(title):
    if MOVIES_DF is None or find_movie_index(title) is None:
        flash("Movie not found. Try another movie.")
        return redirect(url_for("index"))

    idx = find_movie_index(title)
    row = MOVIES_DF.loc[idx]
    details = get_movie_details(row["title"], fallback_genre=row.get("genre", "Unknown"))

    _, similar_titles = get_recommendations(row["title"], top_n=6)
    similar_movies = []
    for t in similar_titles:
        srow = MOVIES_DF[MOVIES_DF["title"] == t].iloc[0]
        similar_movies.append(get_movie_details(t, fallback_genre=srow.get("genre", "Unknown")))

    return render_template(
        "movie_details.html",
        movie=details,
        similar_movies=similar_movies,
        genres=GENRES,
    )


# --------------------------------------------------------------------------
# JSON API (optional, handy for the JS-driven search suggestions / AJAX use)
# --------------------------------------------------------------------------

@app.route("/api/recommend")
def api_recommend():
    movie_name = request.args.get("movie", "").strip()
    if not movie_name or MOVIES_DF is None:
        return jsonify({"error": "Movie not found. Try another movie."}), 404

    matched_title, recommended_titles = get_recommendations(movie_name, top_n=8)
    if matched_title is None:
        return jsonify({"error": "Movie not found. Try another movie."}), 404

    return jsonify({"matched_title": matched_title, "recommendations": recommended_titles})


if __name__ == "__main__":
    app.run(debug=True)
