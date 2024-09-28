# Import necessary libraries
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Title for the Streamlit app
st.title("Hybrid Product Recommendation System")

# Load the dataset from a pickle file
@st.cache
def load_data():
    df = pd.read_pickle('final_df.pkl')  # Replace with your pickle file path
    return df

df = load_data()

# Prepare the dataset for Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['Author_ID', 'Product_ID', 'Rating_Given']], reader)
trainset, testset = train_test_split(data, test_size=0.25)

# Initialize and train the SVD model for collaborative filtering
svd_model = SVD()
svd_model.fit(trainset)

# Make predictions on the test set
predictions = svd_model.test(testset)

# Function to get top N recommendations for a specific user (Collaborative Filtering)
def get_top_n_recommendations(predictions, n=5):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n_recommendations_cf = get_top_n_recommendations(predictions, n=5)

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
def hybrid_recommendations(user_id, product_id, df, predictions, nn, n=5):
    if user_id in top_n_recommendations_cf:
        cf_recommendations = [iid for iid, _ in top_n_recommendations_cf[user_id]]
    else:
        cf_recommendations = []

    cb_recommendations = get_cb_recommendations(product_id, df, nn, n * 2)
    cb_recommendations_list = list(cb_recommendations['Product_ID'])
    combined_recommendations = list(set(cf_recommendations + cb_recommendations_list))

    recommended_products = df[df['Product_ID'].isin(combined_recommendations)].drop_duplicates(subset='Product_ID')
    recommended_products = recommended_products.sort_values(by='Loves_Count_Product', ascending=False)
    relevant_columns = ['Product_ID', 'Product_Name', 'Brand_Name', 'Price', 'Primary_Category', 'Rating_Given', 'Loves_Count_Product']
    return recommended_products[relevant_columns].head(n)

# Streamlit User Inputs
author_id = st.text_input("Enter User ID:", value="7746509195")
product_id = st.text_input("Enter Product ID:", value="P421996")

if st.button("Get Recommendations"):
    recommended_products = hybrid_recommendations(author_id, product_id, df, predictions, nn, n=5)
    st.write(f"Top 5 hybrid recommendations for user {author_id} based on product {product_id}:")
    st.dataframe(recommended_products)
