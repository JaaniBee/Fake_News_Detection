import streamlit as st
import joblib

# Load model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("fake_news_model.pkl")

st.title("Fake News Detection 📰")

news = st.text_area("Enter news to check:")

if st.button("Check"):
    if news.strip() != "":
        news_vec = vectorizer.transform([news])  # Only transform
        prediction = model.predict(news_vec)[0]
        if prediction == 0:
            st.error("🛑 This news is Fake")
        else:
            st.success("✅ This news is Real")
    else:
        st.warning("Please enter some news text.")