# Import necessary libraries
import pandas as pd
import gzip
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Load the compressed pickle file (final_df.pkl.gz)
@st.cache_data
def load_data():
    with gzip.open('final_df.pkl.gz', 'rb') as f:
        df = pickle.load(f)
    return df

# Load the data
df = load_data()

### Step 1: Content-Based Filtering (CB) with Enhanced Feature Engineering
# First, handle missing values by filling NaN with empty strings in relevant columns
df['Product_Name'] = df['Product_Name'].fillna('')
df['Brand_Name'] = df['Brand_Name'].fillna('')
df['Primary_Category'] = df['Primary_Category'].fillna('')
df['Text_Review'] = df['Text_Review'].apply(lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else tokens).fillna('')

# Convert 'Price' column to string before concatenating
df['Price'] = df['Price'].astype(str)

# Enhance combined features by incorporating more descriptive product attributes
df['combined_features'] = (
    df['Product_Name'] + ' ' +
    df['Brand_Name'] + ' ' +
    df['Primary_Category'] + ' ' +
    df['Text_Review'] + ' ' +
    df['Price']
)

# Initialize a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000, min_df=2, max_df=0.7)

# Create a matrix of TF-IDF features
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Initialize NearestNeighbors with cosine similarity
nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=15)  # Increased n_neighbors for better diversity
nn.fit(tfidf_matrix)

# Function to get product recommendations based on Content-Based Filtering using Nearest Neighbors
def get_cb_recommendations(product_id, df, nn, n=5):
    # Get index of the given product
    product_index = df[df['Product_ID'] == product_id].index[0]
    
    # Find the top N nearest neighbors (similar products)
    distances, indices = nn.kneighbors(tfidf_matrix[product_index], n_neighbors=n+1)  # +1 because the product itself will be included
    
    # Get the top N similar products
    top_n_similar_products = indices.flatten()[1:n+1]  # Skip the first as it will be the product itself

    return df.iloc[top_n_similar_products][['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Rating_Given']]

# Streamlit Interface
st.title("Product Recommendation System")

# Input for the product ID
product_id = st.text_input("Enter Product ID:", value="P421996")

# Number of recommendations
num_recommendations = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=5)

# Button to get recommendations
if st.button("Get Recommendations"):
    try:
        recommendations = get_cb_recommendations(product_id, df, nn, n=num_recommendations)
        st.write(f"Top {num_recommendations} similar products to {product_id}:")
        st.dataframe(recommendations)
    except Exception as e:
        st.error(f"Error occurred: {e}")
