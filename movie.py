import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie dataset
movies_data = pd.read_csv('movies.csv')

# Select features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replace null values with an empty string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine selected features into a single string
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Vectorize the text data
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

def recommend_movies(movie_name):
    # Get a list of all movie titles
    list_of_all_titles = movies_data['title'].tolist()

    # Find close matches to the user's input
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0] if find_close_match else None

    if close_match:
        # Find the index of the closest match
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

        # Get similarity scores
        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        # Sort similar movies based on similarity scores
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Return top 10 recommended movies
        recommended_movies = []
        for i, movie in enumerate(sorted_similar_movies):
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            recommended_movies.append(f"{i + 1}. {title_from_index}")

        return recommended_movies[:10]
    else:
        return None

# Streamlit App
st.title('Movie Recommendation System')

# User Input
movie_name = st.text_input('Enter your favourite movie name:')
if movie_name:
    recommendations = recommend_movies(movie_name)
    if recommendations:
        st.subheader('Movies suggested for you:')
        for movie in recommendations:
            st.write(movie)
    else:
        st.warning('No close match found for the entered movie name.')
