import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Modi Sentiment Analysis",
    page_icon="🇮🇳",
    layout="centered"
)


# ---------------- BACKGROUND + CSS ----------------

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    *{
        font-family: 'Poppins', sans-serif;
    }


    .stApp{

        background-image:
        linear-gradient(
        rgba(0,0,0,0.65),
        rgba(0,0,0,0.65)
        ),
        url("https://images.unsplash.com/photo-1529107386315-e1a2ed48a620");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }


    .main-card{

        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(15px);

        padding: 35px;
        border-radius: 25px;

        border:1px solid rgba(255,255,255,0.3);

        box-shadow:
        0 8px 32px rgba(0,0,0,0.4);

    }


    h1{

        color:white;
        text-align:center;
        font-size:45px;
        font-weight:700;

    }


    p{

        color:white;
        text-align:center;
        font-size:18px;

    }


    label{

        color:white !important;
        font-size:18px !important;
        font-weight:600;

    }


    textarea{

        background:white !important;
        border-radius:15px !important;

        font-size:16px !important;

    }


    .stButton button{

        width:100%;

        background:
        linear-gradient(90deg,#ff9933,#ffffff,#138808);

        color:black;

        font-size:20px;

        font-weight:bold;

        border-radius:30px;

        padding:12px;

        border:none;

    }


    .stButton button:hover{

        transform:scale(1.05);

        transition:0.3s;

    }


    </style>
    """,
    unsafe_allow_html=True
)



# ---------------- LOAD MODEL ----------------


model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")



# ---------------- NLP CLEANING ----------------


lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))



def clean_text(text):

    text=text.lower()

    text=re.sub(r"http\S+|www\S+","",text)

    text=re.sub(r"[^a-zA-Z]"," ",text)


    words=text.split()


    words=[
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]


    return " ".join(words)



# ---------------- UI ----------------


st.markdown(
    """
    <div class="main-card">

    <h1>🇮🇳 Modi Reviews Sentiment Analysis</h1>

    <p>
    AI-powered NLP system to analyze public opinions
    and predict sentiment from reviews.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)



st.write("")



review = st.text_area(
    "Enter Review / Tweet",
    placeholder="Example: Modi government has done great work..."
)



st.write("")


if st.button("🔍 Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")

    else:

        cleaned = clean_text(review)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]


        st.write("Prediction:", prediction)


        if prediction == "Positive":

            st.success("😊 Positive Sentiment Detected")


        elif prediction == "Negative":

            st.error("😞 Negative Sentiment Detected")


        else:

            st.info("😐 Neutral Sentiment Detected")       
