# Import necessary libraries
# Import necessary libraries
import numpy as np  # Standard numpy import
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
    
# Collaborative Filtering (CF) - Part 1
@st.cache
def collaborative_filtering(df):
    reader = Reader(rating_scale=(1, 5))  # Ratings are between 1 and 5
    data = Dataset.load_from_df(df[['Author_ID', 'Product_ID', 'Rating_Given']], reader)

    # Train and test split
    trainset = data.build_full_trainset()
    svd_model = SVD()
    svd_model.fit(trainset)
    
    return svd_model

# Content-Based Filtering (CB) - Part 2
@st.cache
def content_based_filtering(df):
    # Handle missing values
    df['Text_Review'] = df['Text_Review'].apply(lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else tokens).fillna('')
    df['combined_features'] = df['Product_Name'] + ' ' + df['Brand_Name'] + ' ' + df['Primary_Category'] + ' ' + df['Text_Review'] + ' ' + df['Price'].astype(str)

    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

    # NearestNeighbors Model
    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=15)
    nn.fit(tfidf_matrix)
    
    return nn, tfidf_matrix

# Hybrid Recommendation System - Part 3
def hybrid_recommendations(user_id, product_id, df, svd_model, nn, n=5):
    # CF Recommendations
    cf_recommendations = get_top_n_recommendations_cf(svd_model, user_id, df, n)

    # CB Recommendations
    cb_recommendations = get_cb_recommendations(product_id, df, nn, n * 2)

    # Combine CF and CB results
    combined_recommendations = list(set(cf_recommendations + list(cb_recommendations['Product_ID'])))
    recommended_products = df[df['Product_ID'].isin(combined_recommendations)].drop_duplicates(subset='Product_ID')

    # Sort and return
    return recommended_products.sort_values(by='Loves_Count_Product', ascending=False)[['Product_ID', 'Product_Name', 'Price', 'Primary_Category', 'Rating_Given']].head(n)

# Function to get CF Recommendations
def get_top_n_recommendations_cf(svd_model, user_id, df, n=5):
    # Get all product_ids
    product_ids = df['Product_ID'].unique()

    # Make predictions for all products for the user
    predictions = [svd_model.predict(user_id, product_id) for product_id in product_ids]
    
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Return the top N recommended product IDs
    return [pred.iid for pred in predictions[:n]]

# Function to get CB Recommendations
def get_cb_recommendations(product_id, df, nn, n=5):
    # Get index of the given product
    product_index = df[df['Product_ID'] == product_id].index[0]
    
    # Find the top N nearest neighbors (similar products)
    distances, indices = nn.kneighbors(tfidf_matrix[product_index], n_neighbors=n+1)  # +1 because the product itself will be included
    
    # Get the top N similar products
    top_n_similar_products = indices.flatten()[1:n+1]  # Skip the first as it will be the product itself

    return df.iloc[top_n_similar_products][['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Rating_Given']]

# Main Streamlit App Code
def main():
    st.title("Hybrid Product Recommendation System")

    # Load dataset
    df = load_data()

    # Get collaborative and content-based models
    svd_model = collaborative_filtering(df)
    nn, tfidf_matrix = content_based_filtering(df)

    # Input user_id and product_id
    user_id = st.text_input('Enter User ID:', '1312882358')
    product_id = st.text_input('Enter Product ID:', 'P421996')

    if st.button('Get Recommendations'):
        # Get hybrid recommendations
        recommendations = hybrid_recommendations(user_id, product_id, df, svd_model, nn, n=5)

        # Display recommendations
        st.write("Top 5 Hybrid Recommendations:")
        st.dataframe(recommendations)

# Run the Streamlit app
if __name__ == '__main__':
    main()
