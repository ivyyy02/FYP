# Import necessary libraries
import pandas as pd
import gzip
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Load the compressed pickle file
@st.cache_data
def load_data():
    with gzip.open('final_df.pkl.gz', 'rb') as f:
        df = pickle.load(f)
    # Reduce the data size
    df = df.sample(frac=0.5)  
    return df
df = load_data()

# Handle missing values by filling NaN with empty strings in relevant columns
df['Product_Name'] = df['Product_Name'].fillna('')
df['Brand_Name'] = df['Brand_Name'].fillna('')
df['Primary_Category'] = df['Primary_Category'].fillna('')
df['Text_Review'] = df['Text_Review'].apply(lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else tokens).fillna('')
df['Price'] = df['Price'].astype(str)

# Group by 'Product_ID' and concatenate the reviews
df_grouped = df.groupby(['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 
                         'Average_Rating_Product', 'Loves_Count_Product'])['Text_Review'].apply(lambda x: ' | '.join(x)).reset_index()

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
    df_grouped['Loves_Count_Product'])

# Cache the vectorizer and model fitting to avoid recomputing every time
@st.cache_data
def vectorize_and_fit(df_grouped):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, min_df=2, max_df=0.7)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_grouped['combined_features'])

    # Initialize NearestNeighbors with cosine similarity
    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
    nn.fit(tfidf_matrix)

    return tfidf_matrix, nn

# Vectorize and fit
tfidf_matrix, nn = vectorize_and_fit(df_grouped)

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
    top_n_similar_products = indices.flatten()[1:]  # Skip the first as it will be the product itself

    # Filter out the product itself from recommendations, just in case
    top_n_similar_products = [i for i in top_n_similar_products if df_grouped.iloc[i]['Product_ID'] != product_id]
    
    # Ensure we still return 'n' recommendations, in case the product itself was included
    top_n_similar_products = top_n_similar_products[:n]

    return df_grouped.iloc[top_n_similar_products][['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Average_Rating_Product', 'Loves_Count_Product']]

# Streamlit Interface
st.title("Product Recommendation System")

# Provide a list of available Product_IDs and their corresponding names for user selection
available_products = df_grouped[['Product_ID', 'Product_Name']].drop_duplicates()

# Create a new list of options where each option is a string combining the Product ID and Product Name
available_product_options = [f"{row['Product_ID']} - {row['Product_Name']}" for _, row in available_products.iterrows()]

# Store the current selected Product_ID
selected_product_option = st.selectbox("Select a Product ID:", available_product_options)

# Extract the Product_ID from the selected option (since we combined Product_ID and Product_Name in the dropdown)
selected_product_id = selected_product_option.split(' - ')[0]

# Number of recommendations
num_recommendations = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=5)

# Button to get recommendations
if st.button("Get Recommendations"):
    try:
        recommendations = get_cb_recommendations(selected_product_id, df_grouped, nn, n=num_recommendations)
        st.write(f"Top {num_recommendations} similar products to {selected_product_id}:")
        st.dataframe(recommendations)
    except Exception as e:
        st.error(f"Error occurred: {e}")


