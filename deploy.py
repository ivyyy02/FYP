# import streamlit as st
# import pickle  # <-- Use pickle instead of joblib
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load the saved Logistic Regression model and TF-IDF vectorizer using pickle
# with open('logistic_regression_model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# with open('tfidf_vectorizer_lr.pkl', 'rb') as vectorizer_file:
#     vectorizer = pickle.load(vectorizer_file)

# # Function to predict whether a product is recommended
# def predict_is_recommended(text_review):
#     # Vectorize the input review using the loaded vectorizer
#     transformed_review = vectorizer.transform([text_review])
    
#     # Predict using the loaded logistic regression model
#     prediction = model.predict(transformed_review)
    
#     # Return 'Yes' if recommended, 'No' otherwise
#     return 'Yes' if prediction == 1 else 'No'

# # Streamlit web app layout
# st.title("Product Recommendation Prediction")

# # Input text box for user review
# text_review = st.text_area("Enter the product review:")

# # Predict button
# if st.button('Predict'):
#     # Ensure the review is not empty
#     if text_review.strip() != "":
#         # Make the prediction
#         result = predict_is_recommended(text_review)
#         # Display the result
#         st.write(f"Is this product recommended? **{result}**")
#     else:
#         st.write("Please enter a product review to get a prediction.")

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
