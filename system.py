import joblib
import os
import streamlit as st  # <-- Add this line to import Streamlit

# Function to load the SVD model or train it if not saved
@st.cache
def load_svd_model(trainset):
    if os.path.exists('svd_model.pkl'):
        return joblib.load('svd_model.pkl')
    else:
        svd_model = SVD()
        svd_model.fit(trainset)
        joblib.dump(svd_model, 'svd_model.pkl')
        return svd_model

# Function to load content-based filtering models or train them if not saved
@st.cache
def load_content_based_models(df):
    if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('nn_model.pkl'):
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        nn = joblib.load('nn_model.pkl')
    else:
        # Handle missing values
        df['Text_Review'] = df['Text_Review'].apply(lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else tokens).fillna('')
        df['combined_features'] = df['Product_Name'] + ' ' + df['Brand_Name'] + ' ' + df['Primary_Category'] + ' ' + df['Text_Review'] + ' ' + df['Price'].astype(str)

        # Train models and save them
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

        nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=15)
        nn.fit(tfidf_matrix)

        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
        joblib.dump(nn, 'nn_model.pkl')

    return tfidf_vectorizer, nn
    
# Main Streamlit App Code
def main():
    st.title("Hybrid Product Recommendation System")

    # Load dataset
    df = load_data()

    # Prepare the training data for collaborative filtering
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['Author_ID', 'Product_ID', 'Rating_Given']], reader)
    trainset = data.build_full_trainset()

    # Load models (collaborative and content-based)
    svd_model = load_svd_model(trainset)
    tfidf_vectorizer, nn = load_content_based_models(df)

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
