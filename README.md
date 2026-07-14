# 🎬 Movie Recommendation AI

A Netflix + IMDb–styled, content-based movie recommendation website built with
Flask, TF-IDF + cosine similarity, and live posters/ratings from TMDB.

---

## Features

- Full-screen cinematic landing page with a glassmorphism search bar
- Content-based recommendations from a TF-IDF + cosine similarity model (`.pkl`)
- Genre filter (Action, Comedy, Horror, Sci-Fi, Thriller, Romance, Animation, Drama)
- Movie result cards with poster, title, genre, release date, and rating
- Live poster/overview/rating enrichment via the TMDB API
- Graceful error handling — "Movie not found" messages and a default poster
  when TMDB is unavailable
- Fully responsive, dark cinematic theme

---

## Project structure

```
MovieRecommendationSystem/
│
├── app.py                     # Flask app: routes, model loading, TMDB calls
├── requirements.txt
├── README.md
├── .env                        # TMDB_API_KEY (placeholder — add your real key)
│
├── model/
│   ├── movie_recommendation.pkl   # TF-IDF + cosine similarity model
│   └── build_model.py             # Rebuilds the .pkl from data/movies.csv
│
├── data/
│   └── movies.csv              # Sample dataset (40 movies) used by build_model.py
│
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/
│       ├── movie-background.jpg
│       └── default-poster.svg
│
└── templates/
    ├── index.html
    ├── recommendations.html
    └── movie_details.html
```

---

## Setup

1. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your TMDB API key.**
   Get a free key at https://www.themoviedb.org/settings/api, then edit `.env`:

   ```
   TMDB_API_KEY=your_real_key_here
   ```

   Without a key, the app still runs — recommendations work, but posters/ratings
   fall back to placeholder values instead of live TMDB data.

3. **Bring your own model, or use the sample one.**
   A working demo `.pkl` (built from `data/movies.csv`) is already included at
   `model/movie_recommendation.pkl`. To use **your own trained model**, copy your
   file to that same path — see "Bringing your own `.pkl`" below for the schema
   `app.py` expects.

4. **Run the app:**

   ```bash
   python app.py
   ```

   Visit http://127.0.0.1:5000

---

## Bringing your own `.pkl`

`app.py`'s `load_model()` function is intentionally flexible and accepts any of
these shapes:

| Shape | What it needs |
|---|---|
| `dict` | `"movies"`: a DataFrame with `title` (required) and `genre` columns, plus **either** `"cosine_sim"` (precomputed 2D similarity matrix) **or** `"tfidf_matrix"` (sparse matrix — similarity is computed on load) |
| `(DataFrame, cosine_sim)` tuple | Same DataFrame requirements as above |
| Bare `DataFrame` | Must include a `tags` or `overview` text column — TF-IDF and cosine similarity are computed automatically on startup |

If your `.pkl` uses different key names (e.g. `similarity` instead of
`cosine_sim`, or `df` instead of `movies`), `load_model()` already checks a few
common aliases — but if it still doesn't match, just rename the keys in your
pickle, or tweak the two `.get(...)` lines near the top of `load_model()` in
`app.py`.

To regenerate the sample model (e.g. after editing `data/movies.csv`):

```bash
python model/build_model.py
```

---

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Home page — search bar + genre dropdown |
| `/recommend` | POST | Takes `movie` form field, returns recommendations page |
| `/genre/<genre_name>` | GET | Browse movies by genre |
| `/movie/<title>` | GET | Single movie detail page with similar titles |
| `/api/recommend?movie=...` | GET | JSON recommendations (for AJAX/JS use) |

---

## Error handling

- Searching a movie that isn't in the dataset shows **"Movie not found. Try
  another movie."** and returns you to the home page.
- If TMDB is unreachable, missing a key, or returns no results, each movie
  card falls back to a default poster and `"N/A"` rating/date instead of
  crashing the page.

---

## Deploying

For production, run behind Gunicorn (already in `requirements.txt`) instead of
Flask's dev server:

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

Remember to set a real, random `FLASK_SECRET_KEY` in `.env` before deploying.

---

## Notes on the included assets

- `static/images/movie-background.jpg` is a generated abstract cinematic
  gradient (film-strip motif, spotlight glow, grain) so the project runs
  out of the box without any licensing concerns. Swap in your own background
  photo at the same path/filename any time.
- `data/movies.csv` and `model/movie_recommendation.pkl` are a small sample
  dataset so the recommender works immediately. Replace with your real
  trained `.pkl` for production use.
