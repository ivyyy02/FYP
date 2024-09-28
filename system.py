import os
import pandas as pd
import joblib
import streamlit as st
from surprise import Dataset, Reader, SVD
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Function to load the dataset from a pickle file
@st.cache_resource
def load_data():
    # Load the dataset from the pickle file
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

# Function to get top N recommendations for a specific user (Collaborative Filtering)
def get_top_n_recommendations(predictions, df, n=5):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Sort and display relevant product information for each recommended product
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    # Return product information for the user
    product_info = []
    for product_id, rating in top_n[uid]:
        product_info.append(df[df['Product_ID'] == product_id][['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category']].iloc[0].to_dict())

    return product_info

# Function to get Content-Based recommendations using Nearest Neighbors
def get_cb_recommendations(product_id, df, nn, tfidf_matrix, n=5):
    # Get index of the given product
    product_index = df[df['Product_ID'] == product_id].index[0]
    
    # Find the top N nearest neighbors (similar products)
    distances, indices = nn.kneighbors(tfidf_matrix[product_index], n_neighbors=n+1)  # +1 to skip the first product (itself)
    
    # Get top N similar products
    top_n_similar_products = indices.flatten()[1:n+1]  # Skip the first as it's the product itself

    return df.iloc[top_n_similar_products][['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category']]

# Hybrid recommendation system that combines Collaborative and Content-Based Filtering
def hybrid_recommendations(user_id, product_id, df, predictions, nn, tfidf_matrix, n=5):
    # Get CF-based recommendations
    cf_recommendations = get_top_n_recommendations(predictions, df, n=n)

    # Get CB-based recommendations for a given product
    cb_recommendations = get_cb_recommendations(product_id, df, nn, tfidf_matrix, n=n)

    # Combine CF and CB recommendations, removing duplicates
    combined_recommendations = pd.concat([pd.DataFrame(cf_recommendations), cb_recommendations]).drop_duplicates(subset='Product_ID')

    return combined_recommendations

def main():
    st.title("Hybrid Product Recommendation System")

    # Load the dataset
    df = load_data()

    # Prepare the training data for collaborative filtering
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['Author_ID', 'Product_ID', 'Rating_Given']], reader)
    trainset = data.build_full_trainset()

    # Load or train the SVD model
    svd_model = load_svd_model(trainset)

    # Make predictions on the test set (you can use train_test_split for real cases)
    testset = trainset.build_testset()
    predictions = svd_model.test(testset)

    # Content-Based Filtering setup
    # Prepare combined features for text and metadata (Content-Based)
    df['Price'] = df['Price'].astype(str)
    df['combined_features'] = df['Product_Name'] + ' ' + df['Brand_Name'] + ' ' + df['Primary_Category'] + ' ' + df['Text_Review'] + ' ' + df['Price']

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

    # Initialize NearestNeighbors for content-based filtering
    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    nn.fit(tfidf_matrix)

    # Streamlit input options
    user_id = st.text_input("Enter User ID:")
    product_id = st.text_input("Enter Product ID:")

    if user_id and product_id:
        # Display hybrid recommendations
        recommendations = hybrid_recommendations(user_id, product_id, df, predictions, nn, tfidf_matrix, n=5)
        st.write("Top 5 hybrid recommendations:")
        st.write(recommendations)

if __name__ == "__main__":
    main()
