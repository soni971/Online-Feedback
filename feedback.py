
# =========================
# Import Libraries
# =========================
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Udemy Course Predictor",
    page_icon="🎓",
    layout="wide"
)

# =========================
# Background Design
# =========================
page_bg = """
<style>
[data-testid="stAppViewContainer"]{
background-image: linear-gradient(to right, #141e30, #243b55);
color: white;
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

[data-testid="stSidebar"]{
background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
}

h1,h2,h3,h4,h5,p,label{
color:white;
}

.stButton>button{
background-color:#00c6ff;
color:white;
border-radius:10px;
height:3em;
width:100%;
font-size:18px;
}

.stTextInput>div>div>input{
background-color:#f1f1f1;
color:black;
}

.result-box{
padding:20px;
border-radius:15px;
background-color:rgba(255,255,255,0.1);
margin-top:20px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# =========================
# Title
# =========================
st.title("🎓 Udemy Course Category & Sentiment Predictor")

st.write("This app predicts course category and sentiment using Machine Learning.")

# =========================
# Load Dataset
# =========================
df = pd.read_csv("udemy_courses.csv")

# =========================
# Download Stopwords
# =========================
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# =========================
# Text Cleaning Function
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# Combine Columns
# =========================
df['text'] = df['course_title'] + " " + df['subject']

df['clean_text'] = df['text'].apply(clean_text)

# =========================
# Sentiment Function
# =========================
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive 😊"
    elif polarity < 0:
        return "Negative 😔"
    else:
        return "Neutral 😐"

df['sentiment'] = df['clean_text'].apply(get_sentiment)

# =========================
# Features and Labels
# =========================
X = df['clean_text']
y = df['subject']

# =========================
# TF-IDF Vectorizer
# =========================
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(X)

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Model Training
# =========================
model = MultinomialNB()

model.fit(X_train, y_train)

# =========================
# Accuracy
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# =========================
# Sidebar
# =========================
st.sidebar.title("📌 Project Info")

st.sidebar.info("""
This project uses:

✔ TF-IDF Vectorizer  
✔ Naive Bayes Algorithm  
✔ TextBlob Sentiment Analysis  
✔ Streamlit Frontend  
""")

st.sidebar.success(f"Model Accuracy: {accuracy:.2f}")

# =========================
# User Input
# =========================
st.subheader("✍ Enter Course Review / Course Text")

user_input = st.text_input(
    "Type here...",
    "Amazing course! I learned a lot and content is very useful"
)

# =========================
# Prediction Button
# =========================
if st.button("🔍 Predict"):

    # Clean Input
    clean_input = clean_text(user_input)

    # Vectorize
    vector_input = tfidf.transform([clean_input])

    # Predict Category
    category = model.predict(vector_input)[0]

    # Predict Sentiment
    sentiment = get_sentiment(user_input)

    # Show Results
    st.markdown(f"""
    <div class="result-box">
        <h2>📚 Predicted Category: {category}</h2>
        <h2>😊 Sentiment: {sentiment}</h2>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Dataset Preview
# =========================
st.subheader("📊 Dataset Preview")

st.dataframe(df.head())