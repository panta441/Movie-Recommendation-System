# Imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from keybert import KeyBERT
import traceback
import subprocess
import sys

# --- SpaCy Model Check and Download ---
SPACY_MODEL_NAME = "en_core_web_lg"

def check_and_download_spacy_model(model_name):
    """Checks if a SpaCy model is installed and downloads it if not."""
    try:
        spacy.load(model_name)
        print(f"SpaCy model '{model_name}' already installed.")
    except OSError:
        print(f"SpaCy model '{model_name}' not found. Attempting download...")
        try:
            # Use sys.executable to ensure the command runs with the correct Python environment
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            print(f"Successfully downloaded '{model_name}'. Please restart the script if needed.")
        except subprocess.CalledProcessError:
            print(f"ERROR: Failed to download SpaCy model '{model_name}'.")
            print(f"Please install it manually by running: python -m spacy download {model_name}")
            sys.exit(1) # Exit if download fails, as the model is likely required
        except Exception as e:
             print(f"An unexpected error occurred during SpaCy model download: {e}")
             sys.exit(1)

# Run the check before trying to load the model definitively
check_and_download_spacy_model(SPACY_MODEL_NAME)

# Paths
MOVIES_CSV = "./ml-latest-small/movies.csv"
RATINGS_CSV = "./ml-latest-small/ratings.csv"
TAGS_CSV = "./ml-latest-small/tags.csv"
EMBEDDING_PATH = "movies_embeddings.npy"
METADATA_PATH = "movies_metadata.parquet"

# Load datasets
movies = pd.read_csv("./ml-latest-small/movies.csv")
ratings = pd.read_csv("./ml-latest-small/ratings.csv")
tags = pd.read_csv("./ml-latest-small/tags.csv")

# Bayesian Average Calculation
C = ratings.groupby('movieId')['rating'].count().mean()
m = ratings['rating'].mean()

bayesian_stats = ratings.groupby('movieId').agg(
    num_ratings=('rating', 'count'),
    sum_ratings=('rating', 'sum')
).reset_index()

bayesian_stats['bayesian_avg'] = (C * m + bayesian_stats['sum_ratings']) / (C + bayesian_stats['num_ratings'])

# Merge with movies
movies = movies.merge(bayesian_stats, on='movieId')

ratings = ratings.merge(
    bayesian_stats[['movieId', 'bayesian_avg']],
    on='movieId',
    how='left'
)

# Filter active users and movies
MIN_RATINGS_USER = 0
MIN_RATINGS_MOVIE = 0

user_counts = ratings['userId'].value_counts()
movie_counts = ratings['movieId'].value_counts()

filtered_ratings = ratings[
    (ratings['userId'].isin(user_counts[user_counts >= MIN_RATINGS_USER].index) &
    (ratings['movieId'].isin(movie_counts[movie_counts >= MIN_RATINGS_MOVIE].index)))
]

# Create utility matrix
utility_matrix = filtered_ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='bayesian_avg',
    # fill_value=0
)

filtered_movie_ids = utility_matrix.columns  # Movies that survived collaborative filtering
movies_filtered = movies[movies.movieId.isin(filtered_movie_ids)].reset_index(drop=True)
movies_filtered['processed_genres'] = movies_filtered['genres'].str.replace('|', ' ')


# 1. Load Pretrained Models
nlp = spacy.load("en_core_web_lg")  # SpaCy for NLP
kw_model = KeyBERT()  # KeyBERT for keyword extraction

# 2. Genre Mapping Dictionary (Expand as needed)
GENRE_MAPPING = {
    'comedy': ['funny', 'humorous', 'hilarious', 'sitcom'],
    'horror': ['scary', 'creepy', 'terrifying', 'ghost', 'zombie'],
    'action': ['fight', 'explosion', 'combat', 'chase', 'battle'],
    'romance': ['love', 'relationship', 'dating', 'couple']
}

# 3. Enhanced Feature Extraction Function
def extract_features(prompt: str) -> dict:
    """Hybrid feature extraction with pretrained models and rules"""
    doc = nlp(prompt.lower())

    # Part 1: Rule-based genre detection
    detected_genres = [
        genre for genre, keywords in GENRE_MAPPING.items()
        if any(keyword in prompt.lower() for keyword in keywords)
    ]

    # Part 2: KeyBERT keyword extraction
    keywords = kw_model.extract_keywords(
        prompt,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=5
    )
    key_phrases = [kw[0] for kw in keywords if kw[1] > 0.2]

    # Part 3: SpaCy entity recognition
    entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'WORK_OF_ART']]

    return {
        'genres': list(set(detected_genres)),
        'keywords': key_phrases + entities,
        'processed_text': ' '.join(detected_genres + key_phrases + entities)
    }

# 5. TF-IDF Vectorization with Expanded Vocabulary
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000
)

feature_matrix = tfidf.fit_transform(movies_filtered['genres'])

# 6. Recommendation Function with Hybrid Features
def get_recommendations(prompt: str, top_n=10):
    print(f"--- Inside LOGIC get_recommendations ---")
    print(f"Received prompt: {prompt}, top_n: {top_n}")
    # Extract features
    features = extract_features(prompt)

    # Transform to TF-IDF
    prompt_vector = tfidf.transform([features['processed_text']])

    # Calculate similarities
    similarities = cosine_similarity(prompt_vector, feature_matrix).flatten()

    # Get top matches
    similar_indices = similarities.argsort()[::-1][:top_n]

    return movies.iloc[similar_indices][['title', 'genres', 'bayesian_avg']]\
                .assign(similarity_score=similarities[similar_indices])

app = FastAPI()

class RecommendationQuery(BaseModel):
    prompt: str
    top_n: int = 10  # Default value

@app.post("/recommend")
async def handle_recommendation_request(query: RecommendationQuery): # Renamed endpoint handler
    try:
        # Now this correctly calls the logic function defined earlier
        recommendations = get_recommendations(query.prompt, query.top_n)
        return recommendations.to_dict(orient="records")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))