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

# Reduce the data size 
df = df.sample(frac=0.5) 

# Handle missing values by filling NaN with empty strings in relevant columns
df['Product_Name'] = df['Product_Name'].fillna('')
df['Brand_Name'] = df['Brand_Name'].fillna('')
df['Primary_Category'] = df['Primary_Category'].fillna('')
df['Text_Review'] = df['Text_Review'].apply(lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else tokens).fillna('')
df['Price'] = df['Price'].astype(str)

# Group by 'Product_ID' and concatenate the reviews
df_grouped = df.groupby(['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 
                         'Average_Rating_Product', 'Loves_Count_Product'])['Text_Review'].apply(lambda x: ' | '.join(x)).reset_index()

# Enhance combined features by incorporating more descriptive product attributes (excluding Ingredients)
df_grouped['combined_features'] = (
    df_grouped['Product_Name'] + ' ' +
    df_grouped['Brand_Name'] + ' ' +
    df_grouped['Primary_Category'] + ' ' +
    df_grouped['Text_Review'] + ' ' +
    df_grouped['Price'])

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000, min_df=2, max_df=0.7)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_grouped['combined_features'])

# Initialize NearestNeighbors with cosine similarity and increased neighbors
nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)  
nn.fit(tfidf_matrix)

# Function to get product recommendations based on Content-Based Filtering using Nearest Neighbors
def get_cb_recommendations(product_id, df_grouped, nn, n=5):
    # Get index of the given product
    product_index = df_grouped[df_grouped['Product_ID'] == product_id].index[0]
    
    # Find the top N nearest neighbors (similar products)
    distances, indices = nn.kneighbors(tfidf_matrix[product_index], n_neighbors=n+1)  # +1 because the product itself will be included
    
    # Get the top N similar products
    top_n_similar_products = indices.flatten()[1:n+1]  # Skip the first as it will be the product itself

    # Filter out the product itself from the recommendations
    similar_products = df_grouped.iloc[top_n_similar_products]
    similar_products = similar_products[similar_products['Product_ID'] != product_id]

    return similar_products[['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 
                             'Primary_Category', 'Average_Rating_Product', 'Loves_Count_Product']]


# Streamlit interface
st.title("Product Recommendation System")

# Dropdown for product selection
product_id = st.selectbox("Select a Product ID:", df_grouped['Product_ID'].unique())

# Number of recommendations to display
num_recommendations = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=5)

# Button to get recommendations
if st.button("Get Recommendations"):
    try:
        recommendations = get_cb_recommendations(product_id, df_grouped, nn, n=num_recommendations)
        st.write(f"Top {num_recommendations} similar products to {product_id}:")
        st.dataframe(recommendations)
    except Exception as e:
        st.error(f"Error occurred: {e}")
