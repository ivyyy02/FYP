import streamlit as st
import pandas as pd
import gzip
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the compressed pickle file (final_df.pkl.gz)
@st.cache_data
def load_data():
    with gzip.open('final_df.pkl.gz', 'rb') as f:
        df = pickle.load(f)
    return df

# Load the data
df = load_data()

# Prepare the data for the Surprise model (Collaborative Filtering)
def prepare_surprise_data(df):
    reader = Reader(rating_scale=(1, 5))  # Adjust scale based on your rating
    data = Dataset.load_from_df(df[['Author_ID', 'Product_ID', 'Rating_Given']], reader)
    return data

# Load Surprise data
data = prepare_surprise_data(df)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Train the SVD model
model = SVD()
model.fit(trainset)

# Function to get top N SVD recommendations for a specific user
def get_svd_recommendations(user_id, model, df, n=5):
    product_ids = df['Product_ID'].unique()
    recommendations = []
    
    for product_id in product_ids:
        pred = model.predict(user_id, product_id)
        recommendations.append((product_id, pred.est))

    # Sort recommendations by estimated rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_n_products = recommendations[:n]

    # Get product details from the original DataFrame
    recommended_products = df[df['Product_ID'].isin([prod[0] for prod in top_n_products])]
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
def hybrid_recommendations(user_id, product_id, model, nn, df, n=5):
    # Get SVD recommendations
    svd_recommendations = get_svd_recommendations(user_id, model, df, n=n)
    
    # Get Content-Based recommendations
    cb_recommendations = get_cb_recommendations(product_id, df, nn, n=n)
    
    # Combine SVD and CB recommendations
    combined_recommendations = pd.concat([svd_recommendations, cb_recommendations]).drop_duplicates(subset='Product_ID')
    
    return combined_recommendations.head(n)

# Streamlit User Inputs
st.title("Hybrid Product Recommendation System")
author_id = st.text_input("Enter User ID:", value="7746509195")
product_id = st.text_input("Enter Product ID:", value="P421996")

if st.button("Get Recommendations"):
    try:
        recommended_products = hybrid_recommendations(author_id, product_id, model, nn, df, n=5)
        st.write(f"Top 5 hybrid recommendations for user {author_id} based on product {product_id}:")
        st.dataframe(recommended_products)
    except Exception as e:
        st.error(f"Error occurred: {e}")
