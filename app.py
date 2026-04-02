import streamlit as st
import joblib
import re
import string
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# Fix NLTK downloads (safe way)
# -------------------------------
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# -------------------------------
# Load model safely (works everywhere)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "sentiment_model.pkl")
tfidf_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)

# -------------------------------
# Text cleaning function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🛒 Flipkart Product Review Sentiment Analysis")

review = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:
        cleaned = clean_text(review)

        with st.spinner("Analyzing..."):
            vector = tfidf.transform([cleaned])
            prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ Positive Review 😊")
        else:
            st.error("❌ Negative Review 😞")
