import os
import pickle
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# ENV
# =========================
load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

if not OMDB_API_KEY:
    raise ValueError("OMDB_API_KEY is not set in environment variables")

OMDB_BASE = "http://www.omdbapi.com/"

# =========================
# FILE PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

# =========================
# GLOBAL VARIABLES
# =========================
df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

# =========================
# FASTAPI APP + LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_pickles_async()
    yield
    # Cleanup code can be added here later if needed


app = FastAPI(
    title="Movie Recommendation API",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELS
# =========================
class OMDBMovieCard(BaseModel):
    imdb_id: str
    title: str
    poster_url: Optional[str] = None
    release_year: Optional[str] = None
    imdb_rating: Optional[str] = None


class OMDBMovieDetail(BaseModel):
    imdb_id: str
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    genres: Optional[List[str]] = None
    imdb_rating: Optional[str] = None


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    omdb: Optional[OMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: Optional[OMDBMovieDetail] = None
    tfidf_recommendations: List[TFIDFRecItem]


# =========================
# HELPERS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


# =========================
# OMDB FUNCTIONS
# =========================
async def omdb_get(params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["apikey"] = OMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(OMDB_BASE, params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"OMDb request error: {repr(e)}"
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OMDb error {r.status_code}"
        )

    return r.json()


async def omdb_search_movie(title: str):
    data = await omdb_get({"t": title})
    if data.get("Response") == "False":
        return None
    return data


async def omdb_movie_details(title: str) -> OMDBMovieDetail:
    data = await omdb_search_movie(title)
    if not data:
        raise HTTPException(status_code=404, detail="Movie not found")

    return OMDBMovieDetail(
        imdb_id=data.get("imdbID"),
        title=data.get("Title"),
        overview=data.get("Plot"),
        release_date=data.get("Released"),
        poster_url=data.get("Poster"),
        genres=data.get("Genre", "").split(", "),
        imdb_rating=data.get("imdbRating")
    )


async def attach_omdb_card_by_title(title: str):
    try:
        data = await omdb_search_movie(title)
        if not data:
            return None
        return OMDBMovieCard(
            imdb_id=data.get("imdbID"),
            title=data.get("Title"),
            poster_url=data.get("Poster"),
            release_year=data.get("Year"),
            imdb_rating=data.get("imdbRating")
        )
    except Exception:
        return None


# =========================
# TF-IDF HELPERS
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}

    try:
        if isinstance(indices, dict):
            for k, v in indices.items():
                title_to_idx[_norm_title(k)] = int(v)
        else:
            for k, v in indices.items():
                title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError("indices.pkl must be dict or pandas Series-like")


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")

    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])

    raise HTTPException(status_code=404, detail=f"Title not found: '{title}'")


def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
            out.append((title_i, float(scores[int(i)])))
            if len(out) >= top_n:
                break
        except Exception:
            continue
    return out


# =========================
# LOAD PICKLES (Improved)
# =========================
async def load_pickles_async():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    print("=== Loading Pickle Files ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"BASE_DIR: {BASE_DIR}")

    try:
        with open(DF_PATH, "rb") as f:
            df = pickle.load(f)
        print(f"✓ df.pkl loaded | Shape: {df.shape}")

        with open(INDICES_PATH, "rb") as f:
            indices_obj = pickle.load(f)
        print("✓ indices.pkl loaded")

        with open(TFIDF_MATRIX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)
        print(f"✓ tfidf_matrix.pkl loaded | Shape: {getattr(tfidf_matrix, 'shape', 'N/A')}")

        with open(TFIDF_PATH, "rb") as f:
            tfidf_obj = pickle.load(f)
        print("✓ tfidf.pkl loaded")

        TITLE_TO_IDX = build_title_to_idx_map(indices_obj)
        print(f"✓ Title to index map built | Size: {len(TITLE_TO_IDX)}")

        if df is None or "title" not in df.columns:
            raise RuntimeError("df.pkl must contain 'title' column")

        print("✅ All models loaded successfully!")

    except Exception as e:
        print(f"❌ Failed to load pickle files: {type(e).__name__}: {e}")
        raise


# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"message": "Movie Recommendation API Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/movie", response_model=OMDBMovieDetail)
async def get_movie(title: str = Query(..., min_length=1)):
    return await omdb_movie_details(title)


@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(10, ge=1, le=30),
):
    details = await omdb_movie_details(query)

    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    tfidf_items = []
    for title, score in recs:
        card = await attach_omdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, omdb=card))

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items
    )