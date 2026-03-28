import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main {
        background-color: #0d0d0d;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 720px;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        letter-spacing: -0.5px;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        color: #f5f5f0;
        line-height: 1.15;
        margin-bottom: 0.3rem;
    }

    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }

    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #c8f135;
        margin-bottom: 0.6rem;
        margin-top: 1.6rem;
    }

    .result-card {
        border-radius: 16px;
        padding: 2rem 2.2rem;
        margin-top: 1.8rem;
        border: 1px solid #2a2a2a;
    }

    .result-stay {
        background: linear-gradient(135deg, #0f1f0a 0%, #0d1a0d 100%);
        border-color: #2e5c1a;
    }

    .result-churn {
        background: linear-gradient(135deg, #1f0a0a 0%, #1a0d0d 100%);
        border-color: #5c1a1a;
    }

    .result-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.4rem;
    }

    .result-verdict {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .verdict-stay { color: #7edb4b; }
    .verdict-churn { color: #f56060; }

    .result-prob {
        font-size: 0.95rem;
        color: #aaa;
        font-weight: 300;
    }

    .prob-bar-bg {
        background: #1e1e1e;
        border-radius: 999px;
        height: 8px;
        margin-top: 1.2rem;
        overflow: hidden;
    }

    .stButton>button {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        background-color: #c8f135;
        color: #0d0d0d;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        width: 100%;
        margin-top: 1.2rem;
        transition: background 0.2s;
    }

    .stButton>button:hover {
        background-color: #b5de1c;
        color: #0d0d0d;
    }

    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #ccc !important;
        font-size: 0.88rem !important;
    }

    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background-color: #1a1a1a !important;
        color: #f0f0f0 !important;
        border: 1px solid #2e2e2e !important;
        border-radius: 8px !important;
    }

    hr {
        border-color: #1f1f1f;
        margin: 1.5rem 0;
    }

    .footer {
        text-align: center;
        color: #444;
        font-size: 0.75rem;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = load_model('model.h5')
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, ohe_geo, le_gender, scaler


model, ohe_geo, le_gender, scaler = load_artifacts()

st.markdown('<div class="hero-title">🏦 Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enter customer details to predict churn probability using the trained ANN model.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Personal Info</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col3:
    age = st.slider("Age", 18, 92, 40)

st.markdown('<div class="section-label">Financial Profile</div>', unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
with col5:
    balance = st.number_input("Balance ($)", min_value=0.0, value=60000.0, step=500.0)

col6, col7 = st.columns(2)
with col6:
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0)
with col7:
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])

st.markdown('<div class="section-label">Account Details</div>', unsafe_allow_html=True)
col8, col9, col10 = st.columns(3)
with col8:
    tenure = st.slider("Tenure (years)", 0, 10, 3)
with col9:
    has_cr_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
with col10:
    is_active = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Predict Churn →"):
    geo_encoded = ohe_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

    input_df = pd.DataFrame([{
        'CreditScore': credit_score,
        'Gender': le_gender.transform([gender])[0],
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary
    }])

    input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)
    input_df = input_df[scaler.feature_names_in_]

    scaled = scaler.transform(input_df)
    prob = model.predict(scaled)[0][0]
    churns = prob > 0.5

    card_class = "result-churn" if churns else "result-stay"
    verdict_class = "verdict-churn" if churns else "verdict-stay"
    verdict_text = "Will Churn" if churns else "Will Stay"
    bar_color = "#f56060" if churns else "#7edb4b"
    bar_pct = round(prob * 100, 1)

    st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-label">Prediction Result</div>
            <div class="result-verdict {verdict_class}">{verdict_text}</div>
            <div class="result-prob">Churn probability: <strong>{bar_pct:.1f}%</strong></div>
            <div class="prob-bar-bg">
                <div style="height:100%; width:{bar_pct}%; background:{bar_color}; border-radius:999px; transition: width 0.5s ease;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">ANN · TensorFlow · Streamlit </div>', unsafe_allow_html=True)