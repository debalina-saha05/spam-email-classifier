import streamlit as st
import pickle
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="AI Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# 2. Load the trained model and vectorizer
# Using @st.cache_resource so it only loads once (saves memory)
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: 'model.pkl' or 'vectorizer.pkl' not found. Please upload them to GitHub.")
        return None, None

model, vectorizer = load_assets()

# 3. Sidebar for Professional Stats
with st.sidebar:
    st.title("📊 Model Insights")
    st.metric(label="Model Accuracy", value="96.68%")
    st.write("**Algorithm:** Multinomial Naive Bayes")
    st.write("**Method:** TF-IDF Vectorization")
    st.info("This project was built to demonstrate NLP classification using Scikit-Learn.")
    st.divider()

# 4. Main UI Header
st.title("📧 AI-Powered Spam Classifier")
st.markdown("""
    Welcome! This tool uses Machine Learning to detect if a message is **Spam** or **Ham (Safe)**.
    Simply paste a message below to analyze it.
""")

# 5. Interactive Sample Selection
sample_emails = {
    "Custom Text": "",
    "Spam Example 1": "WINNER! You have won a £1000 cash prize. Claim now by calling 0800-123-456. This is an urgent message!",
    "Spam Example 2": "URGENT! Your mobile number has been selected for a free gift. Text 'YES' to 60060 to receive your prize.",
    "Ham Example 1": "Hey, are we still meeting for the project discussion at 3 PM today? Let me know.",
    "Ham Example 2": "The weather looks great for the weekend. Do you want to go for a hike?"
}

selected_sample = st.selectbox("Select a sample message to test:", list(sample_emails.keys()))
user_input = st.text_area("Paste your email/message here:", value=sample_emails[selected_sample], height=150)

# 6. Prediction Logic
if st.button('Analyze Message'):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    elif model and vectorizer:
        # Preprocessing & Prediction
        data = [user_input]
        vectorized_data = vectorizer.transform(data)
        prediction = model.predict(vectorized_data)[0]
        
        # Calculate Probability (Confidence)
        proba = model.predict_proba(vectorized_data)[0]
        confidence = max(proba) * 100

        st.divider()
        
        # 7. Professional Result Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Result:")
            if prediction == 1:
                st.error("🚨 This is SPAM")
            else:
                st.success("✅ This is SAFE (Ham)")
        
        with col2:
            st.subheader("Confidence:")
            st.write(f"Level: **{confidence:.2f}%**")
            st.progress(int(confidence))

        # Additional explanation for recruiters
        if prediction == 1:
            st.info("💡 **Why Spam?** The model detected keywords and patterns commonly found in fraudulent messages.")
        else:
            st.info("💡 **Why Ham?** The message pattern closely matches typical conversational language.")
    else:
        st.error("Model assets could not be loaded.")

# 8. Footer
st.divider()
st.caption("2026 Internship Project")