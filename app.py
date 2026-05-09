import requests
import streamlit as st

# =============================
# CONFIG
# =============================
API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CineVerse",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================
# CUSTOM CSS (JioHotstar Style)
# =============================
st.markdown(
    """
<style>

html, body, [class*="css"] {
    background-color: #0B0F1A;
    color: white;
    font-family: 'Inter', sans-serif;
}

/* Main container */
.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1600px;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Top Navbar */
.navbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding: 8px 0px 20px 0px;
}

.logo {
    font-size: 2rem;
    font-weight: 800;
    color: #00BFFF;
}

.subtext {
    color: #9CA3AF;
    font-size: 0.95rem;
}

/* Search Bar */
.stTextInput > div > div > input {
    background-color: #111827;
    color: white;
    border-radius: 14px;
    border: 1px solid #1F2937;
    padding: 14px;
    font-size: 16px;
}

/* Movie Card */
.movie-card {
    position: relative;
    border-radius: 18px;
    overflow: hidden;
    transition: 0.3s ease;
    background: #111827;
    margin-bottom: 14px;
}

.movie-card:hover {
    transform: scale(1.04);
}

.movie-title {
    font-size: 15px;
    font-weight: 600;
    margin-top: 8px;
    color: white;
    line-height: 1.2rem;
    height: 2.5rem;
    overflow:hidden;
}

.movie-rating {
    color: #00BFFF;
    font-size: 13px;
    font-weight: 600;
}

/* Buttons */
.stButton button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg,#00BFFF,#2563EB);
    color: white;
    border: none;
    padding: 10px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton button:hover {
    background: linear-gradient(90deg,#2563EB,#00BFFF);
    color: white;
}

/* Hero Banner */
.hero {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    padding: 40px;
    background: linear-gradient(
        120deg,
        rgba(0,0,0,0.9),
        rgba(0,0,0,0.4)
    ),
    url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=2070');
    background-size: cover;
    background-position: center;
    margin-bottom: 30px;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
}

.hero-desc {
    color: #D1D5DB;
    max-width: 650px;
    font-size: 1.05rem;
}

/* Section Title */
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 20px;
}

/* Details Container */
.details-box {
    background: #111827;
    border-radius: 20px;
    padding: 24px;
    border: 1px solid #1F2937;
}

/* Divider */
hr {
    border-color: #1F2937;
}

</style>
""",
    unsafe_allow_html=True,
)

# =============================
# SESSION STATE
# =============================
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

# =============================
# API HELPER
# =============================
@st.cache_data(ttl=60)
def api_get_json(path, params=None):

    try:
        r = requests.get(
            f"{API_BASE}{path}",
            params=params,
            timeout=25
        )

        if r.status_code >= 400:
            return None

        return r.json()

    except:
        return None


# =============================
# MOVIE GRID
# =============================
def movie_grid(movies, cols=6):

    if not movies:
        st.warning("No movies found.")
        return

    rows = (len(movies) + cols - 1) // cols

    idx = 0

    for _ in range(rows):

        colset = st.columns(cols)

        for c in range(cols):

            if idx >= len(movies):
                break

            movie = movies[idx]
            idx += 1

            title = movie.get("title")
            poster = movie.get("poster_url")
            rating = movie.get("imdb_rating", "N/A")

            with colset[c]:

                st.markdown("<div class='movie-card'>", unsafe_allow_html=True)

                if poster and poster != "N/A":
                    st.image(poster, use_container_width=True)
                else:
                    st.image(
                        "https://via.placeholder.com/300x450?text=No+Poster",
                        use_container_width=True
                    )

                st.markdown(
                    f"""
                    <div class='movie-title'>{title}</div>
                    <div class='movie-rating'>⭐ {rating}</div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button(
                    "View Details",
                    key=f"{title}_{idx}"
                ):
                    st.session_state.selected_movie = title
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)


# =============================
# NAVBAR
# =============================
st.markdown(
    """
<div class='navbar'>
    <div>
        <div class='logo'>🎬 CineVerse</div>
        <div class='subtext'>
            AI Powered Movie Recommendation Platform
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================
# HERO SECTION
# =============================
if not st.session_state.selected_movie:

    st.markdown(
        """
    <div class='hero'>
        <div class='hero-title'>
            Unlimited Movies, AI Recommendations & Entertainment
        </div>

        <br>

        <div class='hero-desc'>
            Discover trending movies, explore detailed information,
            and get personalized recommendations powered by Machine Learning.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# =============================
# HOME PAGE
# =============================
if not st.session_state.selected_movie:

    search = st.text_input(
        "",
        placeholder="🔍 Search movies like Interstellar, Avengers, Batman..."
    )

    if search.strip():

        data = api_get_json(
            "/movie/search",
            params={
                "query": search
            }
        )

        if not data:
            st.error("Movie not found.")
            st.stop()

        movie = data.get("movie_details")

        st.markdown(
            "<div class='section-title'>Search Result</div>",
            unsafe_allow_html=True
        )

        movie_grid(
            [
                {
                    "title": movie.get("title"),
                    "poster_url": movie.get("poster_url"),
                    "imdb_rating": movie.get("imdb_rating")
                }
            ],
            cols=5
        )

        tfidf_movies = []

        for item in data.get("tfidf_recommendations", []):

            omdb = item.get("omdb")

            if omdb:
                tfidf_movies.append(omdb)

        st.markdown(
            "<div class='section-title'>Recommended For You</div>",
            unsafe_allow_html=True
        )

        movie_grid(tfidf_movies, cols=6)

    else:

        trending_movies = [
            "Inception",
            "Interstellar",
            "Avengers",
            "Batman",
            "Titanic",
            "Joker",
            "Avatar",
            "Iron Man",
            "Doctor Strange",
            "The Dark Knight",
            "John Wick",
            "Gladiator"
        ]

        cards = []

        for title in trending_movies:

            data = api_get_json(
                "/movie",
                params={"title": title}
            )

            if data:
                cards.append(
                    {
                        "title": data.get("title"),
                        "poster_url": data.get("poster_url"),
                        "imdb_rating": data.get("imdb_rating")
                    }
                )

        st.markdown(
            "<div class='section-title'>🔥 Trending Movies</div>",
            unsafe_allow_html=True
        )

        movie_grid(cards, cols=6)

# =============================
# DETAILS PAGE
# =============================
else:

    if st.button("← Back to Home"):
        st.session_state.selected_movie = None
        st.rerun()

    data = api_get_json(
        "/movie/search",
        params={
            "query": st.session_state.selected_movie
        }
    )

    if not data:
        st.error("Movie details not found.")
        st.stop()

    movie = data.get("movie_details")

    left, right = st.columns([1, 2])

    with left:

        if movie.get("poster_url"):
            st.image(
                movie.get("poster_url"),
                use_container_width=True
            )

    with right:

        st.markdown(
            "<div class='details-box'>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"# {movie.get('title')}"
        )

        st.markdown(
            f"""
            ⭐ IMDb Rating: {movie.get('imdb_rating')}

            📅 Release Date: {movie.get('release_date')}

            🎭 Genres: {", ".join(movie.get('genres', []))}
            """
        )

        st.markdown("### Overview")

        st.write(movie.get("overview"))

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='section-title'>🎯 Recommended Movies</div>",
        unsafe_allow_html=True
    )

    rec_cards = []

    for item in data.get("tfidf_recommendations", []):

        omdb = item.get("omdb")

        if omdb:
            rec_cards.append(omdb)

    movie_grid(rec_cards, cols=6)