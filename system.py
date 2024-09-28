import streamlit as st
import pandas as pd
import requests
import io
from lightfm import LightFM
from lightfm.data import Dataset as LightFMDataset
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

### Step 1: Collaborative Filtering Using `lightfm`
# Preparing data for LightFM
lightfm_data = LightFMDataset()
lightfm_data.fit(df['Author_ID'].unique(), df['Product_ID'].unique())

# Building interaction matrix for training
(interactions, _) = lightfm_data.build_interactions(
    [(row['Author_ID'], row['Product_ID']) for index, row in df.iterrows()]
)

# Initialize and train the LightFM model
model = LightFM(loss='warp')
model.fit(interactions, epochs=10, num_threads=2)

# Function to get top N collaborative filtering recommendations using LightFM
def get_cf_recommendations(model, user_id, lightfm_data, n=5):
    user_index = lightfm_data.mapping()[0][user_id]
    scores = model.predict(user_index, np.arange(interactions.shape[1]))
    top_items = np.argsort(-scores)[:n]
    
    product_mapping = {v: k for k, v in lightfm_data.mapping()[2].items()}
    recommended_product_ids = [product_mapping[i] for i in top_items]
    return recommended_product_ids

### Step 2: Content-Based Filtering (CB)
# Handle missing values
df['Product_Name'] = df['Product_Name'].fillna('')
df['Brand_Name'] = df['Brand_Name'].fillna('')
df['Primary_Category'] = df['Primary_Category'].fillna('')
df['Text_Review'] = df['Text_Review'].apply(lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else tokens).fillna('')
df['Price'] = df['Price'].astype(str)

# Create combined features for TF-IDF
df['combined_features'] = df['Product_Name'] + ' ' + df['Brand_Name'] + ' ' + df['Primary_Category'] + ' ' + df['Text_Review'] + ' ' + df['Price']

# Vectorize the text
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000, min_df=2, max_df=0.7)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Initialize Nearest Neighbors for content-based filtering
nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=15)
nn.fit(tfidf_matrix)

# Function to get content-based recommendations
def get_cb_recommendations(product_id, df, nn, n=5):
    product_index = df[df['Product_ID'] == product_id].index[0]
    distances, indices = nn.kneighbors(tfidf_matrix[product_index], n_neighbors=n+1)
    top_n_similar_products = indices.flatten()[1:n+1]
    return df.iloc[top_n_similar_products][['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Rating_Given']]

### Step 3: Hybrid Recommendation System
def hybrid_recommendations(user_id, product_id, df, model, lightfm_data, nn, n=5):
    # Collaborative Filtering recommendations
    try:
        cf_recommendations = get_cf_recommendations(model, user_id, lightfm_data, n)
    except KeyError:
        cf_recommendations = []
    
    # Content-Based Filtering recommendations
    cb_recommendations = get_cb_recommendations(product_id, df, nn, n * 2)
    cb_recommendations_list = list(cb_recommendations['Product_ID'])

    # Combine CF and CB recommendations
    combined_recommendations = list(set(cf_recommendations + cb_recommendations_list))

    # Get the recommended products
    recommended_products = df[df['Product_ID'].isin(combined_recommendations)].drop_duplicates(subset='Product_ID')
    recommended_products = recommended_products.sort_values(by='Loves_Count_Product', ascending=False)
    relevant_columns = ['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Rating_Given', 'Loves_Count_Product']
    
    return recommended_products[relevant_columns].head(n)

# Streamlit User Inputs
st.title("Hybrid Product Recommendation System")
author_id = st.text_input("Enter User ID:", value="7746509195")
product_id = st.text_input("Enter Product ID:", value="P421996")

# Get hybrid recommendations when button is clicked
if st.button("Get Recommendations"):
    try:
        recommended_products = hybrid_recommendations(author_id, product_id, df, model, lightfm_data, nn, n=5)
        st.write(f"Top 5 hybrid recommendations for user {author_id} based on product {product_id}:")
        st.dataframe(recommended_products)
    except Exception as e:
        st.error(f"Error occurred: {e}")
