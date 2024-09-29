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

# Group by 'Product_ID' and concatenate the reviews
df_grouped = df.groupby(['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 
                         'Average_Rating_Product', 'Loves_Count_Product'])['Text_Review'].apply(lambda x: ' | '.join(x.astype(str))).reset_index()

# Convert the numeric columns to strings before concatenating
df_grouped['Price'] = df_grouped['Price'].astype(str)
df_grouped['Average_Rating_Product'] = df_grouped['Average_Rating_Product'].astype(str)
df_grouped['Loves_Count_Product'] = df_grouped['Loves_Count_Product'].astype(str)

# Enhance combined features by incorporating more descriptive product attributes
df_grouped['combined_features'] = (
    df_grouped['Product_Name'] + ' ' +
    df_grouped['Brand_Name'] + ' ' +
    df_grouped['Primary_Category'] + ' ' +
    df_grouped['Text_Review'] + ' ' +
    df_grouped['Price'] + ' ' +
    df_grouped['Average_Rating_Product'] + ' ' +
    df_grouped['Loves_Count_Product']
)

# Initialize a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000, min_df=2, max_df=0.7)

# Create a matrix of TF-IDF features
tfidf_matrix = tfidf_vectorizer.fit_transform(df_grouped['combined_features'])

# Initialize NearestNeighbors with cosine similarity
nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)  # Increased n_neighbors for better diversity
nn.fit(tfidf_matrix)

# Function to get product recommendations based on Content-Based Filtering using Nearest Neighbors
def get_cb_recommendations(product_id, df_grouped, nn, n=5):
    # Check if product_id exists in the dataset
    if product_id not in df_grouped['Product_ID'].values:
        raise ValueError("Product ID not found in the dataset.")
    
    # Get index of the given product
    product_index = df_grouped[df_grouped['Product_ID'] == product_id].index[0]
    
    # Find the top N nearest neighbors (similar products)
    distances, indices = nn.kneighbors(tfidf_matrix[product_index], n_neighbors=n+1)  # +1 because the product itself will be included
    
    # Get the top N similar products
    top_n_similar_products = indices.flatten()[1:n+1]  # Skip the first as it will be the product itself
    return df.iloc[top_n_similar_products][['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Average_Rating_Product', 'Loves_Count_Product']]

# Streamlit Interface
st.title("Product Recommendation System")

# Provide a list of available Product_IDs for user selection
available_product_ids = df_grouped['Product_ID'].unique().tolist()

# Create an input field for users to input any Product ID
product_id = st.selectbox("Select a Product ID:", available_product_ids)

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
