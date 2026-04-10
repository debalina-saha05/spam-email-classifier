import streamlit as st
import pickle

# 1. Page Configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# 2. Load the trained model and vectorizer
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
    st.info("An intelligent assistant designed to protect users by automatically detecting phishing attempts and promotional junk in real-time.")
    st.divider()
    st.markdown("## **Project by**")
    st.write("Debalina Saha")

# 4. Main UI Header
st.title("📧 AI-Powered Spam Email Classifier")
st.markdown("""
    Welcome! This tool uses Machine Learning to detect if a message is **Spam** or **Ham (Safe)**.
    Simply paste a message below to analyze it.
""")

# 5. Professional Email Samples
sample_emails = {
    "Custom Text": "",
    
    "Spam: Phishing Attempt": """Subject: Urgent: Your account has been compromised!

Dear User,

Our security system detected an unauthorized login attempt from a new IP address. To protect your data, your account has been temporarily locked.

Please click the link below to verify your identity:
http://secure-login-verify-portal.com/update-security

Best,
Security Team""",

    "Spam: Prize Notification": """Subject: Final Notice: You have (1) pending payout of $500,000.00

Official Notification:

Your email address was selected as a winner in our Global Draw. To claim your prize, please provide your Full Name and Bank Details to our agent.

Reply directly to this email to start the process.

Best,
Claims Department""",

    "Ham: Meeting Invitation": """Subject: Schedule for Project Review Meeting

Hi Debalina,

I hope you're having a productive week. I've scheduled a brief meeting for Monday at 11:00 AM to review the progress on the Spam Classifier project.

Best,
George""",

    "Ham: Internship Feedback": """Subject: Feedback on your Machine Learning Task

Dear Debalina,

Great job on the latest update to your repository. The accuracy of 96.68% is very impressive, and the UI layout is clean and user-friendly.

Best,
The Engineering Team"""
}
selected_sample = st.selectbox(
    "Select a sample message to test:", 
    list(sample_emails.keys())
)
user_input = st.text_area(
    "Paste your email/message here:", 
    value=sample_emails[selected_sample], 
    height=200
)

# 6. Prediction Logic
if st.button('Analyze Message', type="primary"):
    text_to_analyze = user_input.strip()
    
    if text_to_analyze == "":
        st.warning("Please enter some text first!")
    elif model and vectorizer:
        # Preprocessing & Prediction
        vectorized_data = vectorizer.transform([text_to_analyze])
        prediction = model.predict(vectorized_data)[0]
        
        # Calculate Probability (Confidence)
        proba = model.predict_proba(vectorized_data)[0]
        confidence = max(proba) * 100

        st.divider()
        
        # 7. Professional Result Display 
        col1, col2 = st.columns(2)
        
        # Logic to determine colors and components based on prediction
        if prediction == 1:
            color = "#FF4B4B"  # Red for Spam
            result_func = st.error
            result_msg = "🚨 This is SPAM"
            info_msg = "💡 **Why Spam?** The model detected keywords and patterns commonly found in fraudulent messages."
        else:
            color = "#28A745"  # Green for Ham
            result_func = st.success
            result_msg = "✅ This is SAFE (Ham)"
            info_msg = "💡 **Why Ham?** The message pattern closely matches typical conversational language."
        
        with col1:
            st.subheader("Result:")
            result_func(result_msg) # Dynamically uses st.error or st.success
        
        with col2:
            st.subheader("Confidence:")
            st.write(f"Level: **{confidence:.2f}%**")
           
            st.markdown(f"""
                <div style="background-color: #dfe2e6; border-radius: 10px; height: 15px; width: 100%;">
                    <div style="background-color: {color}; width: {confidence}%; height: 15px; border-radius: 10px;"></div>
                </div>
                """, unsafe_allow_html=True)

        # 8. Dynamic Explanation
        st.info(info_msg)
        
    else:
        st.error("Model assets could not be loaded.")

# 9. Final Footer
st.divider()
st.markdown(
    f"<div style='text-align: center; color: #6c757d; font-size: 0.8rem;'>"
    f"Internship Project | 2026"
    f"</div>", 
    unsafe_allow_html=True
)