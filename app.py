import streamlit as st
import pickle
import numpy as np
from keras.models import load_model

# load model and vectorizer
model = load_model("spam_model.keras")

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# UI
st.title("📧 Spam Detection AI")
st.write("Enter an email to check if it's Spam or Ham")

# input box
email = st.text_area("Email text")

# button
if st.button("Predict"):

    if email.strip() == "":
        st.warning("Please enter email text")
    else:

        # vectorize
        email_vect = vectorizer.transform([email]).toarray()

        # predict
        prediction = model.predict(email_vect)[0][0]

        # show result
        if prediction > 0.5:
            st.error(f"Spam detected (confidence {prediction:.2f})")
        else:
            st.success(f"Ham detected (confidence {1-prediction:.2f})")

        st.write("Raw score:", prediction)