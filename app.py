import streamlit as st
import pickle

# 1. Page Configuration
st.set_page_config(
    page_title="AI Spam Classifier",
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
    st.divider()
    st.markdown("### **Project by**")
    st.write("Debalina Saha")
    st.info("This project was built to demonstrate NLP classification using Scikit-Learn.")

# 4. Main UI Header
st.title("📧 AI-Powered Spam Classifier")
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

# --- STABLE INPUT LOGIC ---
# Initialize session state for the text content
if 'text_content' not in st.session_state:
    st.session_state.text_content = ""

# Dropdown for samples
selected_sample = st.selectbox(
    "Select a sample message to test:", 
    list(sample_emails.keys()),
    key="sample_selector"
)

# Update the text area ONLY if the user picks a new sample from the dropdown
if st.session_state.get('last_selected') != selected_sample:
    st.session_state.text_content = sample_emails[selected_sample]
    st.session_state.last_selected = selected_sample

# The Text Area - This captures exactly what you type
user_input = st.text_area(
    "Paste your email/message here:", 
    value=st.session_state.text_content, 
    height=200,
    key="email_input_box"
)

# 6. Prediction Logic
if st.button('Analyze Message', type="primary"):
    # We strip whitespace and check if it's empty
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
st.caption("Internship Project | 2026")