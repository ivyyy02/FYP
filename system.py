import streamlit as st
import pandas as pd
import requests
import io
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to download from Google Drive
def download_file_from_url(url):
    r = requests.get(url, allow_redirects=True)
    return io.BytesIO(r.content)

# Google Drive link to the pickle file
file_url = 'https://drive.google.com/uc?export=download&id=1xF-HjQEaYO102fH8VxjiISD83Pi94cbH'

# Load the dataframe from Google Drive
@st.cache
def load_data():
    file_bytes = download_file_from_url(file_url)
    df = pd.read_pickle(file_bytes)
    return df

# Load the data
df = load_data()

### Collaborative Filtering using implicit ALS ###
# Prepare the dataset for implicit's Alternating Least Squares (ALS) model
def prepare_sparse_matrix(df):
    # Creating a sparse matrix of users, products, and ratings for the ALS model
    users = list(df['Author_ID'].astype('category').cat.codes)
    products = list(df['Product_ID'].astype('category').cat.codes)
    ratings = df['Rating_Given'].astype(float)

    # Create a sparse matrix in the format [users, items, ratings]
    sparse_matrix = sparse.coo_matrix((ratings, (users, products)))
    return sparse_matrix, df['Author_ID'].astype('category'), df['Product_ID'].astype('category')

# Prepare sparse matrix
user_item_matrix, user_cat, item_cat = prepare_sparse_matrix(df)

# Train ALS model using implicit
model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=15)
model.fit(user_item_matrix.T)  # Transposed so that it aligns users with rows and products with columns

# Function to get top N recommendations for a specific user (Collaborative Filtering using implicit ALS)
def get_top_n_recommendations_als(user_id, n=5):
    try:
        user_idx = user_cat.get_loc(user_id)  # Map user_id to ALS matrix row
        recommended_items = model.recommend(user_idx, user_item_matrix.T, N=n, filter_already_liked_items=False)
        recommended_product_ids = [item_cat.categories[item[0]] for item in recommended_items]
        return df[df['Product_ID'].isin(recommended_product_ids)][['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Rating_Given']]
    except KeyError:
        return pd.DataFrame()

### Content-Based Filtering setup ###
df['Product_Name'] = df['Product_Name'].fillna('')
df['Brand_Name'] = df['Brand_Name'].fillna('')
df['Primary_Category'] = df['Primary_Category'].fillna('')
df['Text_Review'] = df['Text_Review'].apply(lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else tokens).fillna('')
df['Price'] = df['Price'].astype(str)
df['combined_features'] = df['Product_Name'] + ' ' + df['Brand_Name'] + ' ' + df['Primary_Category'] + ' ' + df['Text_Review'] + ' ' + df['Price']

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000, min_df=2, max_df=0.7)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=15)
nn.fit(tfidf_matrix)

def get_cb_recommendations(product_id, df, nn, n=5):
    product_index = df[df['Product_ID'] == product_id].index[0]
    distances, indices = nn.kneighbors(tfidf_matrix[product_index], n_neighbors=n+1)
    top_n_similar_products = indices.flatten()[1:n+1]
    return df.iloc[top_n_similar_products][['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Rating_Given']]

# Hybrid Recommendation System
def hybrid_recommendations(user_id, product_id, df, nn, n=5):
    cf_recommendations = get_top_n_recommendations_als(user_id, n)
    cb_recommendations = get_cb_recommendations(product_id, df, nn, n * 2)
    
    # Combine CF and CB recommendations, removing duplicates
    combined_recommendations = pd.concat([cf_recommendations, cb_recommendations]).drop_duplicates(subset='Product_ID')

    # Sort recommendations by 'Loves_Count_Product' or any other relevant metric for variety
    recommended_products = combined_recommendations.sort_values(by='Loves_Count_Product', ascending=False)
    relevant_columns = ['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Rating_Given', 'Loves_Count_Product']
    return recommended_products[relevant_columns].head(n)

# Streamlit User Inputs
st.title("Hybrid Product Recommendation System")
author_id = st.text_input("Enter User ID:", value="7746509195")
product_id = st.text_input("Enter Product ID:", value="P421996")

if st.button("Get Recommendations"):
    try:
        recommended_products = hybrid_recommendations(author_id, product_id, df, nn, n=5)
        st.write(f"Top 5 hybrid recommendations for user {author_id} based on product {product_id}:")
        st.dataframe(recommended_products)
    except Exception as e:
        st.error(f"Error occurred: {e}")
