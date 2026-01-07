import streamlit as st
import joblib
import pickle
from PIL import Image
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    ohe = joblib.load("onehot_encoder.pkl")
    feature_cols = joblib.load("model_features.pkl")
    return model, ohe, feature_cols

model, ohe, feature_cols = load_model()

# Categorical columns (must match training)
cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border-left: 5px solid #10B981;
        color: #065F46;
    }
    .error-box {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 5px solid #EF4444;
        color: #7F1D1D;
    }
    .info-box {
        background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header Section
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<h1 class="main-header">🏦 Bank Marketing Campaign Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Predict customer subscription likelihood for term deposits</p>', unsafe_allow_html=True)
    
        # Image with caption - Centered
    try:
        image = Image.open('bank.jpg')
        col_img1, col_img2, col_img3 = st.columns([1, 3, 1])
        with col_img2:
            st.image(image, width=800, caption="Bank Marketing Campaign Analysis")
    except:
        col_img1, col_img2, col_img3 = st.columns([1, 3, 1])
        with col_img2:
            st.info("📷 Add a 'bank.jpg' image to the directory for better visual appeal")
    
    st.markdown("---")
    
    # Information Box
    with st.expander("ℹ️ About This Predictor", expanded=False):
        st.markdown("""
        This tool predicts whether a customer will subscribe to a term deposit based on:
        - **Demographic data** (age, job, education)
        - **Financial information** (balance, loans)
        - **Campaign history** (previous contacts, outcomes)
        - **Current campaign details** (contact type, month, duration)
        
        The model uses a Random Forest algorithm trained on historical bank marketing data.
        """)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Personal Information")
        
        # Personal info in a container
        with st.container():
            age = st.slider("**Age**", min_value=18, max_value=95, value=40, step=1)
            
            col1a, col1b = st.columns(2)
            with col1a:
                job = st.selectbox("**Job**", 
                    ['management', 'technician', 'entrepreneur', 'blue-collar', 
                     'unknown', 'retired', 'admin.', 'services', 'self-employed', 
                     'unemployed', 'housemaid', 'student'])
                
                marital = st.selectbox("**Marital Status**", 
                    ["divorced", "married", "single"])
            
            with col1b:
                education = st.selectbox("**Education**", 
                    ["primary", "secondary", "tertiary", "unknown"])
        
        st.markdown("### 💰 Financial Information")
        with st.container():
            balance = st.number_input("**Average Yearly Balance ($)**", 
                min_value=-10000, max_value=100000, value=2000, step=100)
            
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                default = st.selectbox("**Credit Default?**", ["no", "yes"])
            with col2b:
                housing = st.selectbox("**Housing Loan?**", ["no", "yes"])
            with col2c:
                loan = st.selectbox("**Personal Loan?**", ["no", "yes"])
    
    with col2:
        st.markdown("### 📞 Contact Information")
        with st.container():
            col3a, col3b = st.columns(2)
            with col3a:
                contact = st.selectbox("**Contact Type**", 
                    ["cellular", "telephone", "unknown"])
            with col3b:
                month = st.selectbox("**Contact Month**", 
                    ["jan", "feb", "mar", "apr", "may", "jun", 
                     "jul", "aug", "sep", "oct", "nov", "dec"])
            
            day = st.slider("**Contact Day of Month**", 
                min_value=1, max_value=31, value=15, step=1)
        
        st.markdown("### 📊 Campaign Details")
        with st.container():
            campaign = st.slider("**Number of Contacts During Campaign**", 
                min_value=1, max_value=50, value=3, step=1)
            
            pdays = st.slider("**Days Since Last Contact**", 
                min_value=-1, max_value=900, value=-1, step=1,
                help="-1 means client was not previously contacted")
            
            previous = st.slider("**Previous Contacts**", 
                min_value=0, max_value=50, value=0, step=1)
            
            poutcome = st.selectbox("**Previous Campaign Outcome**", 
                ["failure", "unknown", "success", "other"])
    
    st.markdown("---")
    
    # Prediction Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🚀 Predict Subscription Likelihood", use_container_width=True)
    
    if predict_button:
        with st.spinner('Analyzing customer data...'):
            # Process inputs
            default_val = 1 if default == "yes" else 0
            housing_val = 1 if housing == "yes" else 0
            loan_val = 1 if loan == "yes" else 0
            
            # Create input dataframe
            input_df = pd.DataFrame([{
                'age': age,
                'job': job,
                'marital': marital,
                'education': education,
                'default': default_val,
                'balance': balance,
                'housing': housing_val,
                'loan': loan_val,
                'contact': contact,
                'day': day,
                'month': month,
                'campaign': campaign,
                'pdays': pdays,
                'previous': previous,
                'poutcome': poutcome
            }])
            
            # Encode categorical variables
            encoded = ohe.transform(input_df[cat_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=ohe.get_feature_names_out(cat_cols)
            )
            
            # Combine and align features
            final_input = pd.concat(
                [input_df.drop(columns=cat_cols), encoded_df],
                axis=1
            )
            final_input = final_input.reindex(columns=feature_cols, fill_value=0)
            
            # Make prediction
            pred = model.predict(final_input)[0]
            prob = model.predict_proba(final_input)[0][1]
            
            # Display results
            st.markdown("---")
            
            # Metrics row
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(label="Prediction", 
                         value="SUBSCRIBE" if pred == 1 else "NOT SUBSCRIBE",
                         delta="High Probability" if prob > 0.7 else "Medium Probability" if prob > 0.4 else "Low Probability")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_metric2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(label="Confidence Score", 
                         value=f"{prob:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_metric3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                risk_level = "Low Risk" if prob < 0.3 else "Medium Risk" if prob < 0.6 else "High Risk"
                st.metric(label="Campaign Risk Level", value=risk_level)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Probability visualization
            st.markdown("### 📈 Subscription Probability")
            st.progress(prob)
            
            # Detailed prediction box
            if pred == 1:
                st.markdown(f"""
                <div class="prediction-box success-box">
                    ✅ <span style="font-size: 1.5rem;">LIKELY TO SUBSCRIBE</span><br>
                    <span style="font-size: 1.1rem; font-weight: normal;">
                    Probability: {prob:.2%} | Confidence: {'Very High' if prob > 0.8 else 'High' if prob > 0.6 else 'Moderate'}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="info-box">🎯 **Recommendation:** This customer is a good candidate for focused follow-up. Consider offering personalized incentives to convert interest into subscription.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box error-box">
                    ❌ <span style="font-size: 1.5rem;">UNLIKELY TO SUBSCRIBE</span><br>
                    <span style="font-size: 1.1rem; font-weight: normal;">
                    Probability: {prob:.2%} | Confidence: {'Very High' if prob < 0.2 else 'High' if prob < 0.4 else 'Moderate'}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="info-box">💡 **Recommendation:** Consider alternative approaches for this customer segment or focus resources on higher-probability candidates.</div>', unsafe_allow_html=True)
            
            # Key factors (if using a model that supports feature importance)
            st.markdown("### 🔍 Key Influencing Factors")
            col_factor1, col_factor2 = st.columns(2)
            
            with col_factor1:
                st.info("""
                **Factors that increase subscription likelihood:**
                - Previous campaign success
                - Higher education level
                - Professional occupations
                - Cellular contact method
                """)
            
            with col_factor2:
                st.info("""
                **Factors that decrease subscription likelihood:**
                - Multiple campaign contacts
                - Previous campaign failure
                - Having existing loans
                - Telephone contact method
                """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>"
        "Bank Marketing Predictor v1.0 • Data Science Model • For demonstration purposes"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()