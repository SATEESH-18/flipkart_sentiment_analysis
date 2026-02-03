import pandas as pd
import re
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# =========================
# Download NLTK resources
# =========================
nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# Load Dataset
# =========================
df = pd.read_csv("data.csv")

df = df[['Review text', 'Ratings']]
df.dropna(inplace=True)

# =========================
# Sentiment Label
# =========================
def label_sentiment(rating):
    return 1 if rating >= 4 else 0

df['sentiment'] = df['Ratings'].apply(label_sentiment)

# =========================
# Text Cleaning
# =========================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['clean_review'] = df['Review text'].apply(clean_text)

# =========================
# Train Test Split
# =========================
X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# TF-IDF
# =========================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =========================
# Model Training
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test_tfidf)
f1 = f1_score(y_test, y_pred)

print("✅ Training Completed")
print("🔥 F1 Score:", f1)

# =========================
# Save Model
# =========================
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("💾 Model saved as sentiment_model.pkl")
print("💾 Vectorizer saved as tfidf_vectorizer.pkl")
