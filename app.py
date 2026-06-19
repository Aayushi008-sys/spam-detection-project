import streamlit as st
import joblib
import re
import string

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# UI Title
st.title("📧 Spam Detection System")

st.write("Enter a message below and check whether it is SPAM or NOT.")

# Input box
message = st.text_area("Enter your message")

# Button
if st.button("Predict"):
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]

    if result == 1:
        st.error("🚨 This is SPAM")
    else:
        st.success("✅ This is NOT SPAM")