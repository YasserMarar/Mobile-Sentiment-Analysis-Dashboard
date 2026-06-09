import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page Configuration
st.set_page_config(
    page_title="📊 Mobile Reviews Overview",
    page_icon="📊",
    layout="wide"
)


# 2. ML Pipeline Model Loader
@st.cache_resource
def load_ml_model():
    if os.path.exists("MobileReviews_model.pkl"):
        return joblib.load("MobileReviews_model.pkl")
    return None

model_pipeline = load_ml_model()

# 3. EXACT UNIQUE VALUES FROM YOUR DATASET (100% Match)
BRAND_MODEL_MAP = {
    'Realme': ['Realme 12 Pro', 'Realme Narzo 70'],
    'Google': ['Pixel 6', 'Pixel 7a', 'Pixel 8'],
    'Xiaomi': ['Redmi Note 13', 'Mi 13 Pro', 'Poco X6'],
    'Motorola': ['Edge 50', 'Moto G Power', 'Razr 40'],
    'Apple': ['iPhone 14', 'iPhone SE', 'iPhone 15 Pro', 'iPhone 13'],
    'OnePlus': ['OnePlus 12', 'OnePlus Nord 3', 'OnePlus 11R'],
    'Samsung': ['Galaxy Note 20', 'Galaxy S24', 'Galaxy A55', 'Galaxy Z Flip']
}

ALL_COUNTRIES = ['India', 'Brazil', 'UAE', 'Australia', 'Germany', 'UK', 'Canada', 'USA']
ALL_LANGUAGES = ['Hindi', 'Portuguese', 'English', 'German']
ALL_SOURCES = ['Amazon', 'Flipkart', 'AliExpress', 'BestBuy', 'eBay']

# --- SECTION 1: HEADER ---
st.title("🔮 Intelligent Review Sentiment Predictor")
st.markdown("Select your device configurations and review metrics to get instant sentiment classification.")
st.divider()

# --- SECTION 2: DYNAMIC DEVICE SELECTION (Outside Form to fix the bug) ---
st.markdown("##### 📱 Step 1: Choose Device Specs (Live Link)")
col_brand, col_model = st.columns(2)

with col_brand:
    user_brand = st.selectbox("Select Device Brand (7 Brands):", list(BRAND_MODEL_MAP.keys()))

with col_model:
    # This will now update instantly because it is outside the form barrier
    available_models = BRAND_MODEL_MAP[user_brand]
    user_model = st.selectbox("Select Specific Model (22 Models):", available_models)

st.divider()

# --- SECTION 3: REMAINING METRICS FORM ---
st.markdown("##### 📋 Step 2: Customer Demographics & Metrics")

with st.form("prediction_remaining_form"):
    c1, c2 = st.columns(2)
    
    with c1:
        user_price = st.number_input(
            label="Retail Price in USD ($):",
            min_value=100.0,
            max_value=2000.0,
            value=650.0,
            step=50.0
        )
        user_verified = st.selectbox("Verified Customer Purchase?:", [True, False])
        user_country = st.selectbox("Customer Country Location (8 Countries):", ALL_COUNTRIES)
        user_lang = st.selectbox("Review Text Language (4 Languages):", ALL_LANGUAGES)
        
    with c2:
        user_source = st.selectbox("E-Commerce Source Website (5 Sources):", ALL_SOURCES)
        user_age = st.slider("Customer Demographics Age:", min_value=18, max_value=70, value=30)
        user_length = st.slider("Review Length (Characters Count):", min_value=10, max_value=300, value=65)
        user_words = st.slider("Review Summary Word Count:", min_value=2, max_value=60, value=12)

    # Form Submission Button
    submit_button = st.form_submit_button(label="🔮 Run Sentiment Prediction Pipeline")

# --- SECTION 4: INFERENCE RUNNER ---
if submit_button:
    # Build payload structure identical to training dataset X matrix
    input_data = pd.DataFrame([{
        'age': user_age,
        'brand': user_brand,
        'model': user_model,
        'price_usd': user_price,
        'country': user_country,
        'language': user_lang,
        'verified_purchase': user_verified,
        'review_length': user_length,
        'word_count': user_words,
        'source': user_source
    }])
    
    st.subheader("📊 Model Inference Outcome")
    
    if model_pipeline is not None:
        try:
            # Predict using your trained CatBoost pipeline
            prediction = model_pipeline.predict(input_data)
            pred_label = prediction[0][0] if hasattr(prediction[0], '__len__') else prediction[0]
            
            if pred_label == 'Positive':
                st.success("😊 **Predicted Customer Sentiment: Positive**")
            elif pred_label == 'Neutral':
                st.warning("😐 **Predicted Customer Sentiment: Neutral**")
            else:
                st.error("😡 **Predicted Customer Sentiment: Negative**")
                
        except Exception as e:
            st.error(f"❌ Pipeline Processing Error: {e}")
    else:
        # Static smart fallback backup
        st.warning("⚠️ Model binary file (`MobileReviews_model.pkl`) not detected. Running mockup rule-base instead.")
        if user_length > 65 and user_verified:
            st.success("😊 **Predicted Customer Sentiment (Mockup): Positive**")
        elif user_length < 45:
            st.error("😡 **Predicted Customer Sentiment (Mockup): Negative**")
        else:
            st.warning("😐 **Predicted Customer Sentiment (Mockup): Neutral**")

# --- SECTION 5: FOOTER ---
st.divider()
st.caption("Deployment Dashboard Core Platform | Dynamic values verified and synchronized completely.")