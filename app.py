import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page title
st.title("📩 SMS Spam Detector")
st.write("Type a message below and let the model predict whether it's **SPAM** or **HAM**.")

# User input
message = st.text_area("Enter your SMS message:")

# Predict button
if st.button("Detect"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        msg_vector = vectorizer.transform([message.lower()])
        prediction = model.predict(msg_vector)[0]
        st.success(f"✅ This message is predicted as: **{prediction.upper()}**")
