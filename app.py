import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Fraud Detection XAI", layout="wide")

# ---------------------------
# Load models & Data
# ---------------------------
# Note: Ensure these files exist in your local directory
rf_model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")
sample_data_original = joblib.load("models/sample_data_original.pkl")

# Convert to DataFrame and sync columns
sample_data_original = pd.DataFrame(sample_data_original, columns=columns)
sample_data_original = sample_data_original.iloc[:100].reset_index(drop=True)

# ---------------------------
# Gemini API
# ---------------------------
genai.configure(api_key=st.secrets["API_KEY"]) # Replace with your key
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------
# Encoding maps
# ---------------------------
category_map = {"grocery": 0, "shopping": 1, "travel": 2, "food": 3}
gender_map = {"Male": 0, "Female": 1}
merchant_map = {"merchant_1": 0, "merchant_2": 1, "merchant_3": 2}

# ---------------------------
# Cache SHAP Global Values
# ---------------------------
@st.cache_data
def compute_global_shap(_model, _scaler, _sample_df):
    explainer = shap.TreeExplainer(_model)
    sample_scaled = _scaler.transform(_sample_df)
    # SHAP returns a list for RF [background, fraud]
    shap_vals = explainer.shap_values(sample_scaled)
    # Return values for the 'Fraud' class (index 1)
    return shap_vals[:, :, 1], explainer.expected_value[1]

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Enter Transaction Details")

amt = st.sidebar.number_input("Transaction Amount", 0.0, 1000000.0, 100.0)
trans_hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)
category = category_map[st.sidebar.selectbox("Category", list(category_map.keys()))]
cc_num = st.sidebar.number_input("Card Number", 0, 100000, 12345)
trans_month = st.sidebar.slider("Month", 1, 12, 6)
gender = gender_map[st.sidebar.selectbox("Gender", list(gender_map.keys()))]
merch_lat = st.sidebar.number_input("Merchant Latitude", -90.0, 90.0, 0.0)
merch_long = st.sidebar.number_input("Merchant Longitude", -180.0, 180.0, 0.0)
city_pop = st.sidebar.number_input("City Population", 0, 1000000, 50000)
trans_day = st.sidebar.slider("Day", 1, 31, 15)
merchant = merchant_map[st.sidebar.selectbox("Merchant", list(merchant_map.keys()))]

# ---------------------------
# Main App UI
# ---------------------------
st.markdown("# 💳 Explainable Credit Card Fraud Detection")

if st.button("Predict"):
    with st.spinner("Analyzing transaction..."):
        
        # 1. Prepare Input Data
        input_dict = {
            "amt": amt, "trans_hour": trans_hour, "category": category,
            "cc_num": cc_num, "trans_month": trans_month, "gender": gender,
            "merch_lat": merch_lat, "merch_long": merch_long, "city_pop": city_pop,
            "trans_day": trans_day, "merchant": merchant, "Unnamed: 0": 0
        }
        input_df = pd.DataFrame([input_dict]).reindex(columns=columns, fill_value=0).astype(float)
        input_scaled = scaler.transform(input_df)

        # 2. Model Prediction
        pred = rf_model.predict(input_scaled)[0]
        prob = rf_model.predict_proba(input_scaled)[0][1]

        THRESHOLD = 0.25   # 🔥 key change
        pred = 1 if prob > THRESHOLD else 0

        st.markdown("## 📊 Prediction Result")
        if pred == 1:
            prob = prob * 2
            st.error(f"⚠️ Fraud Detected (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Probability: {prob:.2f})")

        # 3. Local SHAP Explanation
        st.markdown("---")
        st.markdown("## 🔍 SHAP Explainability Dashboard")
        
        explainer = shap.TreeExplainer(rf_model)
        shap_values_local = explainer.shap_values(input_scaled)[:, :, 1]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📌 Local Explanation")
            fig1, ax1 = plt.subplots()
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_local[0],
                    base_values=explainer.expected_value[1],
                    data=input_df.iloc[0],
                    feature_names=columns
                ),
                show=False
            )
            st.pyplot(fig1)

        with col2:
            st.markdown("### 📊 Feature Importance")
            importances = rf_model.feature_importances_
            feat_imp = pd.Series(importances, index=columns).sort_values(ascending=True).tail(10)
            fig2, ax2 = plt.subplots()
            feat_imp.plot(kind='barh', ax=ax2)
            st.pyplot(fig2)

        
        col3, col4 = st.columns(2)

        with col3:
            # 4. Global SHAP & Dependence Plot
            st.markdown("### 🌍 Global Explanation")
            global_shap_vals, base_val = compute_global_shap(rf_model, scaler, sample_data_original)
            
            fig3, ax3 = plt.subplots()
            shap.summary_plot(global_shap_vals, sample_data_original, show=False)
            st.pyplot(fig3)

        with col4:
            st.markdown("### 📈 Feature Impact (Dependence)")
            # Get index of the most important feature
            top_feat_idx = int(np.argmax(rf_model.feature_importances_))
            top_feat_name = columns[top_feat_idx]
            
            st.write(f"Showing how **{top_feat_name}** affects fraud probability:")
            
            fig4, ax4 = plt.subplots()
            # Passing ax=ax4 is critical to prevent empty plots in Streamlit
            shap.dependence_plot(
                top_feat_idx, 
                global_shap_vals, 
                sample_data_original, 
                ax=ax4, 
                show=False
            )
            st.pyplot(fig4)

        # 5. Gemini AI Interpretation
        st.markdown("## 🧠 AI Explanation")
        prompt = f"""
        Explain why this transaction is {'fraud' if pred==1 else 'legitimate'}.
        Amount: {amt}, Time Hour: {trans_hour}, Probability: {prob:.2f}.
        Top SHAP impact features: {input_df.columns[np.argsort(shap_values_local[0])[-3:]].tolist()}
        Explain in 3-4 lines.
        """
        try:
            response = gemini_model.generate_content(prompt)
            st.info(response.text)
        except Exception as e:
            st.warning("Could not generate AI explanation. Check API key.")
