import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("lr_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# ---- Custom CSS Styling (White + Decent Theme) ----
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h5 {
        color: #333333;
        text-align: center;
    }

    textarea {
        background-color: #f9f9f9 !important;
        color: #000000 !important;
        border: 1px solid #ccc !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }

    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        transition: 0.3s;
    }

    .stButton button:hover {
        background-color: #45a049;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
        text-align: center;
    }

    .fake {
        background-color: #ffe6e6;
        color: #cc0000;
        border-left: 6px solid #cc0000;
    }

    .real {
        background-color: #e6ffe6;
        color: #006600;
        border-left: 6px solid #009933;
    }
    </style>
""", unsafe_allow_html=True)

# ---- App Title and Subheading ----
st.title("📰 Fake News Detection")
st.markdown("<h5>🔍 Paste a news article below to check if it's <b>Real</b> or <b>Fake</b>.</h5>", unsafe_allow_html=True)

# ---- Custom Label and Text Area ----
st.markdown("<label style='color: #333; font-size: 16px;'>📝 Paste your news content below:</label>", unsafe_allow_html=True)
input_text = st.text_area(label="", height=200)

# ---- Predict Button ----
if st.button("🚀 Predict Now"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        input_vec = vectorizer.transform([input_text])
        prediction = model.predict(input_vec)[0]

        # ---- Display Result ----
        if prediction == 1:
            st.markdown("<div class='result-box fake'>🚫 This news is FAKE!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box real'>✅ This news is REAL!</div>", unsafe_allow_html=True)
