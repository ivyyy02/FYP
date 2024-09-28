# Import necessary libraries
import numpy as np  # Explicit numpy import to avoid issues with numpy.core.multiarray
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import joblib

# Load Dataset (Ensure numpy is initialized properly)
@st.cache_resource
def load_data():
    # Load your dataset (adjust the path if necessary)
    df = pd.read_pickle("final_df.pkl")
    return df

# Function to load the SVD model or train it if not saved
@st.cache_resource
def load_svd_model(trainset):
    if os.path.exists('svd_model.pkl'):
        return joblib.load('svd_model.pkl')
    else:
        svd_model = SVD()
        svd_model.fit(trainset)
        joblib.dump(svd_model, 'svd_model.pkl')
        return svd_model

# Function to load NearestNeighbors model or train it if not saved
@st.cache_resource
def load_nn_model(tfidf_matrix):
    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=15)
    nn.fit(tfidf_matrix)
    return nn

# Main function to run Streamlit app
def main():
    st.title("Hybrid Product Recommendation System")

    # Load dataset
    df = load_data()

    # Prepare the training data for collaborative filtering
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['Author_ID', 'Product_ID', 'Rating_Given']], reader)

    # Train/test split
    trainset, testset = train_test_split(data, test_size=0.25)

    # Load or train SVD model
    svd_model = load_svd_model(trainset)

    # TF-IDF vectorization for content-based filtering
    df['Text_Review'] = df['Text_Review'].apply(lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else tokens).fillna('')
    df['combined_features'] = df['Product_Name'] + ' ' + df['Brand_Name'] + ' ' + df['Primary_Category'] + ' ' + df['Text_Review'] + ' ' + df['Price'].astype(str)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

    # Load or train NearestNeighbors model
    nn_model = load_nn_model(tfidf_matrix)

    # Input for user and product ID for recommendations
    author_id = st.text_input('Enter User ID:')
    product_id = st.text_input('Enter Product ID:')

    # Collaborative Filtering Predictions
    if author_id:
        # Get CF-based recommendations
        predictions = svd_model.test(testset)
        top_n_recommendations_cf = get_top_n_recommendations(predictions)
        if author_id in top_n_recommendations_cf:
            st.write(f"Top Collaborative Filtering Recommendations for User {author_id}:")
            st.write(top_n_recommendations_cf[author_id])

    # Content-Based Filtering Predictions
    if product_id:
        st.write(f"Top Content-Based Recommendations for Product {product_id}:")
        recommendations = get_cb_recommendations(product_id, df, nn_model)
        st.write(recommendations)

# Helper functions for recommendations
def get_top_n_recommendations(predictions, n=5):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def get_cb_recommendations(product_id, df, nn_model, n=5):
    product_index = df[df['Product_ID'] == product_id].index[0]
    distances, indices = nn_model.kneighbors(tfidf_matrix[product_index], n_neighbors=n+1)
    top_n_similar_products = indices.flatten()[1:n+1]
    return df.iloc[top_n_similar_products][['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Rating_Given']]

if __name__ == '__main__':
    main()
