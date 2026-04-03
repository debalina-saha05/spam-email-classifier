import streamlit as st
import pickle

# PAGE CONFIGURATION 
st.set_page_config(page_title="AI Email Classifier", layout="centered")

#LOAD THE TRAINED MODEL & VECTORIZER
@st.cache_resource # This keeps the model in memory so it's fast
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, tfidf

try:
    model, tfidf = load_model()
except FileNotFoundError:
    st.error("Model files not found! Please run 'python train.py' first.")

#USER INTERFACE
st.title("AI-Powered Email Spam Detector")
st.markdown("---")
st.write("This system uses **Machine Learning** to analyze email content and classify it as Spam or Safe.")

#Input area
email_input = st.text_area("Paste the email content below:", height=250, placeholder="Example: Congratulations! You've won a $1,000 gift card...")

if st.button("Analyze Email"):
    if email_input.strip() != "":
        # 1. Transform the input text using the saved vectorizer
        data = tfidf.transform([email_input])
        
        # 2. Make prediction
        prediction = model.predict(data)[0]
        
        # 3. Display Result
        st.markdown("### Result:")
        if prediction == 1:
            st.error("**ANALYSIS: THIS IS SPAM!**")
            st.info("Advice: Do not click any links or provide personal information.")
        else:
            st.success("**ANALYSIS: THIS IS A SAFE MESSAGE (HAM)**")
            st.info("This message appears to be legitimate.")
    else:
        st.warning("Please paste an email first!")

# FOOTER
st.markdown("---")
st.caption("Internship Project 2026")