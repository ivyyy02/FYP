import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the Logistic Regression model and TF-IDF vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer_lr.pkl')

# Streamlit UI
st.title("Product Recommendation Sentiment Analysis")

st.write("Enter a product review below and the model will predict if the product is recommended or not.")

# Input text box for review
user_input = st.text_area("Type your product review here:")

if st.button("Predict"):
    if user_input:
        # Vectorize the input text
        user_input_vectorized = vectorizer.transform([user_input])

        # Make prediction using the Logistic Regression model
        prediction = model.predict(user_input_vectorized)
        prediction_prob = model.predict_proba(user_input_vectorized)

        # Show the results
        if prediction == 1:
            st.success(f"The model predicts that this product is **Recommended** with a probability of {prediction_prob[0][1]:.2f}")
        else:
            st.error(f"The model predicts that this product is **Not Recommended** with a probability of {prediction_prob[0][0]:.2f}")
    else:
        st.warning("Please enter a review text.")
