import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load movie data (replace 'path/to/data.csv' with your actual data path)
data = pd.read_csv('path/to/data.csv')

# Create a function for cleaning movie titles (optional but recommended)
def clean_title(title):
    # Remove punctuation, lowercase, and split into words
    words = title.lower().replace(',', ' ').replace('.', ' ').split()
    # Remove stop words (optional but can improve similarity calculations)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Add title cleaning step if desired:
if 'title' in data.columns:
    data['clean_title'] = data['title'].apply(clean_title)
    # Use the cleaned title column for recommendations
    movie_data = data[['clean_title', 'genres']]  # Select relevant columns for recommendations
else:
    # If no 'title' column exists, use other relevant columns (e.g., 'movie_id')
    movie_data = data[['genres']]  # Assuming 'genres' is available

# Function to calculate TF-IDF similarity between movies
def tfidf_similarity(movie1, movie2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([movie1['genres'], movie2['genres']])
    return (tfidf_matrix * tfidf_matrix.T).toarray()[0][1]

# Function to recommend movies based on a given movie and similarity threshold
def recommend_movies(movie_title, movie_data, n=5, similarity_threshold=0.7):
    # Find the index of the movie in the data
    movie_index = movie_data[movie_data['clean_title'] == movie_title].index[0]  # Using cleaned title if available

    # Calculate similarity scores for all movies with the given movie
    similarities = []
    for i in range(len(movie_data)):
        similarities.append(tfidf_similarity(movie_data.iloc[movie_index], movie_data.iloc[i]))

    # Sort movies based on similarity scores (descending order)
    movie_data['similarity'] = similarities
    sorted_data = movie_data.sort_values(by=['similarity'], ascending=False)

    # Recommend top n similar movies (excluding the original movie and filtering by threshold)
    recommendations = sorted_data[sorted_data['clean_title'] != movie_title][sorted_data['similarity'] >= similarity_threshold].head(n)['clean_title'].tolist()  # Using cleaned title if available
    return recommendations

# Example usage
movie_to_recommend = "The Shawshank Redemption"
recommendations = recommend_movies(movie_to_recommend, movie_data)

print(f"Recommendations for {movie_to_recommend}:")
if recommendations:
    for movie in recommendations:
        print(movie)
else:
    print("No movies found with sufficient similarity.")
