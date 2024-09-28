import streamlit as st
import pandas as pd
import gzip
import pickle
from lightfm import LightFM
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sparse

# Load the compressed pickle file (final_df.pkl.gz)
@st.cache_data
def load_data():
    with gzip.open('final_df.pkl.gz', 'rb') as f:
        df = pickle.load(f)
    return df

# Load the data
df = load_data()

# Prepare the data for LightFM (Collaborative Filtering)
def prepare_sparse_matrix(df):
    user_item_matrix = df.pivot(index='Author_ID', columns='Product_ID', values='Rating_Given').fillna(0)
    sparse_user_item = sparse.csr_matrix(user_item_matrix.values)
    return sparse_user_item, user_item_matrix

sparse_user_item, user_item_matrix = prepare_sparse_matrix(df)

# Train the LightFM model
def train_lightfm_model(sparse_user_item):
    model = LightFM(loss='warp')  # Using Weighted Approximate-Rank Pairwise loss
    model.fit(sparse_user_item, epochs=30, num_threads=4)
    return model

lightfm_model = train_lightfm_model(sparse_user_item)

# Function to get top N LightFM recommendations for a specific user
def get_lightfm_recommendations(user_id, model, user_item_matrix, df, n=5):
    user_idx = list(user_item_matrix.index).index(user_id)
    scores = model.predict(user_idx, np.arange(sparse_user_item.shape[1]))
    top_indices = scores.argsort()[::-1][:n]
    recommended_products = df[df['Product_ID'].isin(user_item_matrix.columns[top_indices])]
    return recommended_products[['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Rating_Given']]

# Content-Based Filtering setup
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
def hybrid_recommendations(user_id, product_id, lightfm_model, nn, df, n=5):
    # Get LightFM recommendations
    lightfm_recommendations = get_lightfm_recommendations(user_id, lightfm_model, user_item_matrix, df, n=n)
    
    # Get Content-Based recommendations
    cb_recommendations = get_cb_recommendations(product_id, df, nn, n=n)
    
    # Combine LightFM and CB recommendations
    combined_recommendations = pd.concat([lightfm_recommendations, cb_recommendations]).drop_duplicates(subset='Product_ID')
    
    return combined_recommendations.head(n)

# Streamlit User Inputs
st.title("Hybrid Product Recommendation System")
author_id = st.text_input("Enter User ID:", value="7746509195")
product_id = st.text_input("Enter Product ID:", value="P421996")

if st.button("Get Recommendations"):
    try:
        recommended_products = hybrid_recommendations(author_id, product_id, lightfm_model, nn, df, n=5)
        st.write(f"Top 5 hybrid recommendations for user {author_id} based on product {product_id}:")
        st.dataframe(recommended_products)
    except Exception as e:
        st.error(f"Error occurred: {e}")
