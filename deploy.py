
import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved Logistic Regression model and TF-IDF vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer_lr.pkl')

# Function to predict whether a product is recommended
def predict_is_recommended(text_review):
    # Vectorize the input review using the loaded vectorizer
    transformed_review = vectorizer.transform([text_review])
    
    # Predict using the loaded logistic regression model
    prediction = model.predict(transformed_review)
    
    # Return 'Yes' if recommended, 'No' otherwise
    return 'Yes' if prediction == 1 else 'No'

# Streamlit web app layout
st.title("Product Recommendation Prediction")

# Input text box for user review
text_review = st.text_area("Enter the product review:")

# Predict button
if st.button('Predict'):
    # Ensure the review is not empty
    if text_review.strip() != "":
        # Make the prediction
        result = predict_is_recommended(text_review)
        # Display the result
        st.write(f"Is this product recommended? **{result}**")
    else:
        st.write("Please enter a product review to get a prediction.")
